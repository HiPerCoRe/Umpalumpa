#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_starpu.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::extrema_finder {

namespace {// to avoid poluting
  struct ExecuteArgs
  {
    Settings settings;
    const std::vector<std::unique_ptr<AExtremaFinder>> *algs;
  };

  void Codelet(void *buffers[], void *func_arg)
  {
    auto *args = reinterpret_cast<ExecuteArgs *>(func_arg);
    auto *valsP = reinterpret_cast<AExtremaFinder::OutputData::PayloadType *>(buffers[0]);
    auto *locsP = reinterpret_cast<AExtremaFinder::OutputData::PayloadType *>(buffers[1]);
    auto out = AExtremaFinder::OutputData(std::move(*valsP), std::move(*locsP));
    auto *inP = reinterpret_cast<AExtremaFinder::InputData::PayloadType *>(buffers[2]);
    auto in = AExtremaFinder::InputData(std::move(*inP));
    auto &alg = args->algs->at(static_cast<size_t>(starpu_worker_get_id()));
    std::ignore = alg->Execute(out, in);// we have no way of comunicate the result
    alg->Synchronize();// this codelet is run asynchronously, but we have to wait till it's done
                       // to be able to use starpu task synchronization properly
  }

  struct InitArgs
  {
    const AExtremaFinder::OutputData &out;
    const AExtremaFinder::InputData &in;
    const Settings &settings;
    std::vector<std::unique_ptr<AExtremaFinder>> &algs;
  };

  void CpuInit(void *args)
  {
    auto *a = reinterpret_cast<InitArgs *>(args);
    auto alg = std::make_unique<SingleExtremaFinderCPU>();
    if (alg->Init(a->out, a->in, a->settings)) {
      a->algs[static_cast<size_t>(starpu_worker_get_id())] = std::move(alg);
    }
  }

  void CudaInit(void *args)
  {
    auto *a = reinterpret_cast<InitArgs *>(args);
    std::vector<CUstream> stream = { starpu_cuda_get_local_stream() };
    auto alg = std::make_unique<SingleExtremaFinderCUDA>(starpu_worker_get_id(), stream);
    if (alg->Init(a->out, a->in, a->settings)) {
      a->algs[static_cast<size_t>(starpu_worker_get_id())] = std::move(alg);
    }
  }
}// namespace

bool SingleExtremaFinderStarPU::Init(const StarpuOutputData &out,
  const StarpuInputData &in,
  const Settings &s)
{
  return AExtremaFinder::Init({ out.GetValues()->GetPayload(), out.GetLocations()->GetPayload() },
    in.GetData()->GetPayload(),
    s);
}

bool SingleExtremaFinderStarPU::InitImpl()
{
  if (0 == starpu_worker_get_count()) {
    spdlog::warn("No workers available. Is StarPU properly initialized?");
  }
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

bool SingleExtremaFinderStarPU::ExecuteImpl(const OutputData &out, const InputData &in)
{
  bool res = false;
  if ((nullptr == outPtr) && (nullptr == inPtr)) {
    // input and output data are not Starpu Payloads
    using Type = data::StarpuPayload<InputData::PayloadType::LDType>;
    auto o = StarpuOutputData(
      std::make_unique<Type>(out.GetValues()), std::make_unique<Type>(out.GetLocations()));
    auto i = StarpuInputData(std::make_unique<Type>(in.GetData()));
    res = ExecuteImpl(o, i);
    // assume that results are requested
    o.GetValues()->Unregister();
    o.GetLocations()->Unregister();
  } else {
    // input and output are Starpu Payload, stored locally
    res = ExecuteImpl(*outPtr, *inPtr);
  }
  return res;
}

bool SingleExtremaFinderStarPU::Execute(const StarpuOutputData &out, const StarpuInputData &in)
{
  // store the reference to payloads locally
  outPtr = &out;
  inPtr = &in;
  // call normal execute with the normal Payloads to get all checks etc.
  auto res =
    AExtremaFinder::Execute({ out.GetValues()->GetPayload(), out.GetLocations()->GetPayload() },
      in.GetData()->GetPayload());
  // cleanup
  outPtr = nullptr;
  inPtr = nullptr;
  return res;
}

bool SingleExtremaFinderStarPU::ExecuteImpl(const StarpuOutputData &out, const StarpuInputData &in)
{
  auto createArgs = [this]() {
    // need to use malloc, because task can only call free
    auto *args = reinterpret_cast<ExecuteArgs *>(malloc(sizeof(ExecuteArgs)));
    args->algs = &this->algs;
    args->settings = this->GetSettings();
    return args;
  };
  struct starpu_task *task = starpu_task_create();
  task->handles[0] = out.GetValues()->GetHandle();
  task->handles[1] = out.GetLocations()->GetHandle();
  task->handles[2] = in.GetData()->GetHandle();
  task->workerids = utils::StarPUUtils::CreateWorkerMask(task->workerids_len,
    algs);// FIXME bug in the StarPU? If the mask is completely 0, codelet is being invoked anyway
  task->cl_arg = createArgs();
  task->cl_arg_size = sizeof(ExecuteArgs);
  task->cl_arg_free = 1;
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
    c.modes[1] = STARPU_W;
    c.modes[2] = STARPU_R;
    return &c;
  }();

  task->name = this->taskName.c_str();
  STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit %s", this->taskName);
  return true;
}
}// namespace umpalumpa::extrema_finder
