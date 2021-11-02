#include <libumpalumpa/algorithms/fourier_processing/fp_starpu.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_cpu.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_cuda.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::fourier_processing {
namespace {// to avoid poluting
  struct ExecuteArgs
  {
    Settings settings;
    const std::vector<std::unique_ptr<AFP>> *algs;
  };

  void Codelet(void *buffers[], void *func_arg)
  {
    auto *outP = reinterpret_cast<AFP::OutputData::DataType *>(buffers[0]);
    auto out = AFP::OutputData(std::move(*outP));
    auto *inD = reinterpret_cast<AFP::InputData::DataType *>(buffers[1]);
    auto *inF = reinterpret_cast<AFP::InputData::FilterType *>(buffers[2]);
    auto in = AFP::InputData(std::move(*inD), std::move(*inF));
    auto *args = reinterpret_cast<ExecuteArgs *>(func_arg);
    auto &alg = args->algs->at(static_cast<size_t>(starpu_worker_get_id()));
    std::ignore = alg->Execute(out, in);// we have no way of comunicate the result
    alg->Synchronize();// this codelet is run asynchronously, but we have to wait till it's done
                       // to be able to use starpu task synchronization properly
  }

  struct InitArgs
  {
    const AFP::OutputData &out;
    const AFP::InputData &in;
    const Settings &settings;
    std::vector<std::unique_ptr<AFP>> &algs;
  };

  void CpuInit(void *args)
  {
    auto *a = reinterpret_cast<InitArgs *>(args);
    auto alg = std::make_unique<FP_CPU>();
    if (alg->Init(a->out, a->in, a->settings)) {
      a->algs[static_cast<size_t>(starpu_worker_get_id())] = std::move(alg);
    }
  }

  void CudaInit(void *args)
  {
    auto *a = reinterpret_cast<InitArgs *>(args);
    std::vector<CUstream> stream = { starpu_cuda_get_local_stream() };
    auto alg = std::make_unique<FP_CUDA>(starpu_worker_get_id(), stream);
    if (alg->Init(a->out, a->in, a->settings)) {
      a->algs[static_cast<size_t>(starpu_worker_get_id())] = std::move(alg);
    }
  }
}// namespace

bool FPStarPU::Init(const StarpuOutputData &out, const StarpuInputData &in, const Settings &s)
{
  return AFP::Init(out.GetData()->GetPayload(),
    { in.GetData()->GetPayload(), in.GetFilter()->GetPayload() },
    s);
}

bool FPStarPU::InitImpl()
{
  const auto &out = this->GetOutputRef();
  const auto &in = this->GetInputRef();
  const auto &s = this->GetSettings();
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

bool FPStarPU::ExecuteImpl(const OutputData &out, const InputData &in)
{
  bool res = false;
  if ((nullptr == outPtr) && (nullptr == inPtr)) {
    // input and output data are not Starpu Payloads
    using DType = data::StarpuPayload<InputData::DataType::LDType>;
    using FType = data::StarpuPayload<InputData::FilterType::LDType>;
    auto o = StarpuOutputData(std::make_unique<DType>(out.GetData()));
    auto i = StarpuInputData(
      std::make_unique<DType>(in.GetData()), std::make_unique<FType>(in.GetFilter()));
    res = ExecuteImpl(o, i);
    // assume that results are requested
    o.GetData()->Unregister();
  } else {
    // input and output are Starpu Payload, stored locally
    res = ExecuteImpl(*outPtr, *inPtr);
  }
  return res;
}

bool FPStarPU::Execute(const StarpuOutputData &out, const StarpuInputData &in)
{
  // store the reference to payloads locally
  outPtr = &out;
  inPtr = &in;
  // call normal execute with the normal Payloads to get all checks etc.
  auto res = AFP::Execute(
    out.GetData()->GetPayload(), { in.GetData()->GetPayload(), in.GetFilter()->GetPayload() });
  // cleanup
  outPtr = nullptr;
  inPtr = nullptr;
  return res;
}

bool FPStarPU::ExecuteImpl(const StarpuOutputData &out, const StarpuInputData &in)
{
  struct starpu_task *task = starpu_task_create();
  task->handles[0] = out.GetData()->GetHandle();
  task->handles[1] = in.GetData()->GetHandle();
  task->handles[2] = in.GetFilter()->GetHandle();
  task->workerids = CreateWorkerMask(task->workerids_len,
    algs);// FIXME bug in the StarPU? If the mask is completely 0, codelet is being invoked anyway
  task->cl_arg = new ExecuteArgs{ this->GetSettings(), &algs };
  task->cl_arg_size = sizeof(ExecuteArgs);
  task->cl = [] {
    static starpu_codelet c = {};
    c.where = STARPU_CUDA | STARPU_CPU;
    c.cpu_funcs[0] = Codelet;
    c.cuda_funcs[0] = Codelet;
    c.nbuffers = 3;
    c.modes[0] = STARPU_W;
    c.modes[1] = STARPU_R;
    c.modes[2] = STARPU_R;
    return &c;
  }();

  task->name = this->taskName.c_str();
  STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit %s", this->taskName);
  return true;
}
}// namespace umpalumpa::fourier_processing
