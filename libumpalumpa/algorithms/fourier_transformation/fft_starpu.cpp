#include <libumpalumpa/data/starpu_utils.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/fft_starpu.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/fft_cpu.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/fft_cuda.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa {
namespace fourier_transformation {

  namespace {// to avoid poluting
    struct ExecuteArgs
    {
      Settings settings;
      const std::vector<std::unique_ptr<AFFT>> *algs;
    };

    void Codelet(void *buffers[], void *func_arg)
    {
      auto *outP = reinterpret_cast<AFFT::InputData::PayloadType *>(buffers[0]);
      auto out = AFFT::OutputData(std::move(*outP));
      auto *inP = reinterpret_cast<AFFT::InputData::PayloadType *>(buffers[1]);
      auto in = AFFT::InputData(std::move(*inP));
      auto *args = reinterpret_cast<ExecuteArgs *>(func_arg);
      auto &alg = args->algs->at(static_cast<size_t>(starpu_worker_get_id()));
      alg->Execute(out, in);
      alg->Synchronize();// this codelet is run asynchronously, but we have to wait till it's done
                         // to be able to use starpu task synchronization properly
    }

    struct InitArgs
    {
      const AFFT::OutputData &out;
      const AFFT::InputData &in;
      const Settings &settings;
      std::vector<std::unique_ptr<AFFT>> &algs;
    };

    void CpuInit(void *args)
    {
      auto *a = reinterpret_cast<InitArgs *>(args);
      auto alg = std::make_unique<FFTCPU>();
      if (alg->Init(a->out, a->in, a->settings)) {
        a->algs[static_cast<size_t>(starpu_worker_get_id())] = std::move(alg);
      }
    }

    void CudaInit(void *args)
    {
      auto *a = reinterpret_cast<InitArgs *>(args);
      std::vector<CUstream> stream = { starpu_cuda_get_local_stream() };
      auto alg = std::make_unique<FFTCUDA>(stream);
      if (alg->Init(a->out, a->in, a->settings)) {
        a->algs[static_cast<size_t>(starpu_worker_get_id())] = std::move(alg);
      }
    }
  }// namespace

  bool FFTStarPU::Init(const OutputData &out, const InputData &in, const Settings &s)
  {
    this->SetSettings(s);
    algs.clear();
    algs.resize(starpu_worker_get_count());
    InitArgs args = { out, in, s, algs };
    starpu_execute_on_each_worker(CpuInit, &args, STARPU_CPU);
    starpu_execute_on_each_worker(
      CudaInit, &args, STARPU_CUDA);// FIXME if one of the workers is not initialized, then we
                                    // should prevent starpu from running execute() on it
    spdlog::info("{} worker(s) initialized",
      std::count_if(algs.begin(), algs.end(), [](const auto &i) { return i != nullptr; }));
    return (algs.size()) > 0;
  }

  bool FFTStarPU::Execute(const OutputData &out, const InputData &in)
  {
    using Type = data::StarpuPayload<InputData::PayloadType::LDType>;
    auto o = StarpuOutputData(std::make_unique<Type>(out.data));
    auto i = StarpuInputData(std::make_unique<Type>(in.data));
    auto res = Execute(o, i);
    // assume that results are requested
    o.data->Unregister();
    return res;
  }

  bool FFTStarPU::Execute(const StarpuOutputData &out, const StarpuInputData &in)
  {
    struct starpu_task *task = starpu_task_create();
    task->handles[0] = out.data->GetHandle();
    task->handles[1] = in.data->GetHandle();
    task->workerids = CreateWorkerMask(task->workerids_len,
      algs);// FIXME bug in the StarPU? If the mask is completely 0, codelet is being invoked anyway
    task->cl_arg = new ExecuteArgs{ this->GetSettings(), &algs };
    task->cl_arg_size = sizeof(ExecuteArgs);
    task->cl = [] {
      static starpu_codelet c = {};
      c.where = STARPU_CUDA | STARPU_CPU;
      c.cpu_funcs[0] = Codelet;
      c.cuda_funcs[0] = Codelet;
      c.nbuffers = 2;
      c.modes[0] = STARPU_W;
      c.modes[1] = STARPU_R;
      return &c;
    }();

    task->name = this->taskName.c_str();
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit %s", this->taskName);
    return true;
  }


}// namespace fourier_transformation
}// namespace umpalumpa
