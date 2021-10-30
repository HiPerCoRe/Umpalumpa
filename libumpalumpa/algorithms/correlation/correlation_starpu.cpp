#include <libumpalumpa/algorithms/correlation/correlation_starpu.hpp>
#include <libumpalumpa/algorithms/correlation/correlation_cpu.hpp>
#include <libumpalumpa/algorithms/correlation/correlation_cuda.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::correlation {
namespace {// to avoid poluting
  struct ExecuteArgs
  {
    Settings settings;
    const std::vector<std::unique_ptr<ACorrelation>> *algs;
  };

  void Codelet(void *buffers[], void *func_arg)
  {
    auto *outP = reinterpret_cast<ACorrelation::InputData::PayloadType *>(buffers[0]);
    auto out = ACorrelation::OutputData(std::move(*outP));
    auto *inP1 = reinterpret_cast<ACorrelation::InputData::PayloadType *>(buffers[1]);
    auto *inP2 = reinterpret_cast<ACorrelation::InputData::PayloadType *>(buffers[2]);
    auto in = ACorrelation::InputData(std::move(*inP1), std::move(*inP2));
    auto *args = reinterpret_cast<ExecuteArgs *>(func_arg);
    auto &alg = args->algs->at(static_cast<size_t>(starpu_worker_get_id()));
    std::ignore = alg->Execute(out, in);// we have no way of comunicate the result
    alg->Synchronize();// this codelet is run asynchronously, but we have to wait till it's done
                       // to be able to use starpu task synchronization properly
  }

  struct InitArgs
  {
    const ACorrelation::OutputData &out;
    const ACorrelation::InputData &in;
    const Settings &settings;
    std::vector<std::unique_ptr<ACorrelation>> &algs;
  };

  void CpuInit(void *args)
  {
    auto *a = reinterpret_cast<InitArgs *>(args);
    auto alg = std::make_unique<Correlation_CPU>();
    if (alg->Init(a->out, a->in, a->settings)) {
      a->algs[static_cast<size_t>(starpu_worker_get_id())] = std::move(alg);
    }
  }

  void CudaInit(void *args)
  {
    auto *a = reinterpret_cast<InitArgs *>(args);
    std::vector<CUstream> stream = { starpu_cuda_get_local_stream() };
    auto alg = std::make_unique<Correlation_CUDA>(starpu_worker_get_id(), stream);
    if (alg->Init(a->out, a->in, a->settings)) {
      a->algs[static_cast<size_t>(starpu_worker_get_id())] = std::move(alg);
    }
  }
}// namespace

bool CorrelationStarPU::Init(const StarpuOutputData &out,
  const StarpuInputData &in,
  const Settings &s)
{
  return ACorrelation::Init(out.GetCorrelations()->GetPayload(),
    { in.GetData1()->GetPayload(), in.GetData2()->GetPayload() },
    s);
}

bool CorrelationStarPU::InitImpl()
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

bool CorrelationStarPU::ExecuteImpl(const OutputData &out, const InputData &in)
{
  bool res = false;
  if ((nullptr == outPtr) && (nullptr == inPtr)) {
    // input and output data are not Starpu Payloads
    using Type = data::StarpuPayload<InputData::PayloadType::LDType>;
    auto o = StarpuOutputData(std::make_unique<Type>(out.GetCorrelations()));
    auto i =
      StarpuInputData(std::make_unique<Type>(in.GetData1()), std::make_unique<Type>(in.GetData2()));
    res = ExecuteImpl(o, i);
    // assume that results are requested
    o.GetCorrelations()->Unregister();
  } else {
    // input and output are Starpu Payload, stored locally
    res = ExecuteImpl(*outPtr, *inPtr);
  }
  return res;
}

bool CorrelationStarPU::Execute(const StarpuOutputData &out, const StarpuInputData &in)
{
  // store the reference to payloads locally
  outPtr = &out;
  inPtr = &in;
  // call normal execute with the normal Payloads to get all checks etc.
  auto res = ACorrelation::Execute(out.GetCorrelations()->GetPayload(),
    { in.GetData1()->GetPayload(), in.GetData2()->GetPayload() });
  // cleanup
  outPtr = nullptr;
  inPtr = nullptr;
  return res;
}

bool CorrelationStarPU::ExecuteImpl(const StarpuOutputData &out, const StarpuInputData &in)
{
  struct starpu_task *task = starpu_task_create();
  task->handles[0] = out.GetCorrelations()->GetHandle();
  task->handles[1] = in.GetData1()->GetHandle();
  task->handles[2] = in.GetData2()->GetHandle();
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
}// namespace umpalumpa::correlation
