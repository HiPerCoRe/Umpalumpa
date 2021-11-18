#include <libumpalumpa/algorithms/fourier_processing/fp_starpu.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_cpu.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_cuda.hpp>
#include <libumpalumpa/utils/starpu.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::fourier_processing {
namespace {// to avoid poluting

  struct Args
  {
    const AFP::OutputData &out;
    const AFP::InputData &in;
    const Settings &settings;
    std::vector<std::unique_ptr<AFP>> &algs;
  };

  void Codelet(void *buffers[], void *func_arg)
  {
    using umpalumpa::utils::StarPUUtils;
    auto *args = reinterpret_cast<Args *>(func_arg);

    auto pOut = StarPUUtils::Assemble(args->out.GetData(), StarPUUtils::ReceivePDPtr(buffers[0]));
    auto out = AFP::OutputData(pOut);

    auto pInData = StarPUUtils::Assemble(args->in.GetData(), StarPUUtils::ReceivePDPtr(buffers[1]));
    auto pInFilter =
      StarPUUtils::Assemble(args->in.GetFilter(), StarPUUtils::ReceivePDPtr(buffers[2]));
    auto in = AFP::InputData(pInData, pInFilter);

    auto &alg = args->algs.at(static_cast<size_t>(starpu_worker_get_id()));
    std::ignore = alg->Execute(out, in);// we have no way of comunicate the result
    alg->Synchronize();// this codelet is run asynchronously, but we have to wait till it's done
                       // to be able to use starpu task synchronization properly
  }

  void CpuInit(void *args)
  {
    auto *a = reinterpret_cast<Args *>(args);
    auto alg = std::make_unique<FPCPU>();
    if (alg->Init(a->out, a->in, a->settings)) {
      a->algs[static_cast<size_t>(starpu_worker_get_id())] = std::move(alg);
    }
  }

  void CudaInit(void *args)
  {
    auto *a = reinterpret_cast<Args *>(args);
    std::vector<CUstream> stream = { starpu_cuda_get_local_stream() };
    auto alg = std::make_unique<FPCUDA>(starpu_worker_get_id(), stream);
    if (alg->Init(a->out, a->in, a->settings)) {
      a->algs[static_cast<size_t>(starpu_worker_get_id())] = std::move(alg);
    }
  }
}// namespace

bool FPStarPU::InitImpl()
{
  if (0 == starpu_worker_get_count()) {
    spdlog::warn("No workers available. Is StarPU properly initialized?");
  }
  noOfInitWorkers = 0;
  const auto &out = this->GetOutputRef();
  const auto &in = this->GetInputRef();
  const auto &s = this->GetSettings();
  algs.clear();
  algs.resize(starpu_worker_get_count());
  Args args = { out, in, s, algs };
  starpu_execute_on_each_worker(CpuInit, &args, STARPU_CPU);
  starpu_execute_on_each_worker(
    CudaInit, &args, STARPU_CUDA);
  noOfInitWorkers =
    std::count_if(algs.begin(), algs.end(), [](const auto &i) { return i != nullptr; });
  auto level = (0 == noOfInitWorkers) ? spdlog::level::warn : spdlog::level::info;
  spdlog::log(level, "{} worker(s) initialized", noOfInitWorkers);
  return noOfInitWorkers > 0;
}

bool FPStarPU::ExecuteImpl(const OutputData &out, const InputData &in)
{
  using utils::StarPUUtils;
  // we need at least one initialized worker, otherwise mask would be 0 and all workers
  // would be used
  if (noOfInitWorkers < 1) return false;
  struct starpu_task *task = starpu_task_create();
  task->handles[0] = *StarPUUtils::GetHandle(out.GetData().dataInfo);
  task->handles[1] = *StarPUUtils::GetHandle(in.GetData().dataInfo);
  task->handles[2] = *StarPUUtils::GetHandle(in.GetFilter().dataInfo);
  task->workerids = utils::StarPUUtils::CreateWorkerMask(task->workerids_len,
    algs);
  task->cl_arg = new Args{ out, in, this->GetSettings(), algs }; // FIXME memory leak
  task->cl_arg_size = sizeof(Args);
  // make sure we free the mask
  task->callback_func = [](void *) { /* empty on purpose */ };
  task->callback_arg = task->workerids;
  task->callback_arg_free = 1;
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
