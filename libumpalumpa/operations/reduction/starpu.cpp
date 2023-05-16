#include <libumpalumpa/operations/reduction/starpu.hpp>
#include <libumpalumpa/operations/reduction/cpu.hpp>
#include <libumpalumpa/operations/reduction/cuda.hpp>
#include <libumpalumpa/utils/starpu.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::reduction {

namespace {// to avoid poluting
  struct Args
  {
    // we need to store local copies of the wrappers
    // as references might not be valid by the time the codelet is executed
    // FIXME this has to be refactored properly to work with MPI
    const Abstract::OutputData out;
    const Abstract::InputData in;
    const Settings settings;
    std::vector<Abstract *> &ops;
  };

  struct CodeletArgs
  {
    data::Payload<data::LogicalDescriptor> out;
    data::Payload<data::LogicalDescriptor> in;
    std::vector<Abstract *> *ops;
  };

  void Codelet(void *buffers[], void *func_arg)
  {
    using umpalumpa::utils::StarPUUtils;
    auto *args = reinterpret_cast<CodeletArgs *>(func_arg);

    auto pOut = StarPUUtils::Assemble(args->out, buffers[0]);
    auto out = Abstract::OutputData(pOut);

    auto pIn = StarPUUtils::Assemble(args->in, buffers[1]);
    auto in = Abstract::InputData(pIn);

    auto &op = args->ops->at(static_cast<size_t>(starpu_worker_get_id()));
    std::ignore = op->Execute(out, in);// we have no way of comunicate the result
    op->Synchronize();// this codelet is run asynchronously, but we have to wait till it's done
                       // to be able to use starpu task synchronization properly
  }
  void CpuInit(void *args)
  {
    auto *a = reinterpret_cast<Args *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *op = reinterpret_cast<CPU *>(a->ops.at(id));
    if (nullptr == op) { op = new CPU(); }
    if (!op->Init(a->out, a->in, a->settings)) {
      delete op;
      op = nullptr;
    }
    // update the vector
    a->ops.at(id) = op;
  }

  void CudaInit(void *args)
  {
    auto *a = reinterpret_cast<Args *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *op = reinterpret_cast<CUDA *>(a->ops.at(id));
    if (nullptr == op) {
      std::vector<CUstream> stream = { starpu_cuda_get_local_stream() };
      op = new CUDA(static_cast<int>(id), stream);
    }
    if (!op->Init(a->out, a->in, a->settings)) {
      delete op;
      op = nullptr;
    }
    // update the vector
    a->ops.at(id) = op;
  }

  template<typename T> void UniversalCleanup(void *args)
  {
    auto *vec = reinterpret_cast<std::vector<Abstract *> *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *op = reinterpret_cast<T *>(vec->at(id));
    if (nullptr != op) { op->Cleanup(); }
  }

  template<typename T> void DeleteOp(void *args)
  {
    auto *vec = reinterpret_cast<std::vector<Abstract *> *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *op = reinterpret_cast<T *>(vec->at(id));
    delete op;
  }
}// namespace

StarPU::~StarPU()
{
  if (!this->IsInitialized()) return;
  Synchronize();
  starpu_execute_on_each_worker(DeleteOp<CPU>, &ops, STARPU_CPU);
  starpu_execute_on_each_worker(DeleteOp<CUDA>, &ops, STARPU_CUDA);
}

void StarPU::Cleanup()
{
  if (!this->IsInitialized()) return;
  Synchronize();
  starpu_execute_on_each_worker(UniversalCleanup<CPU>, &ops, STARPU_CPU);
  starpu_execute_on_each_worker(UniversalCleanup<CUDA>, &ops, STARPU_CUDA);
  Abstract::Cleanup();
}

void StarPU::Synchronize()
{
  while (!taskQueue.empty()) {
    std::ignore = starpu_task_wait(taskQueue.front());
    taskQueue.pop();
  }
}

bool StarPU::InitImpl()
{
  if (0 == starpu_worker_get_count()) {
    spdlog::warn("No workers available. Is StarPU properly initialized?");
  }
  noOfInitWorkers = 0;
  const auto &out = this->GetOutputRef();
  const auto &in = this->GetInputRef();
  const auto &s = this->GetSettings();
  ops.resize(starpu_worker_get_count());
  Args args = { out, in, s, ops };
  starpu_execute_on_each_worker(CpuInit, &args, STARPU_CPU);
  starpu_execute_on_each_worker(CudaInit, &args, STARPU_CUDA);
  noOfInitWorkers =
    std::count_if(ops.begin(), ops.end(), [](const auto &i) { return i != nullptr; });
  auto level = (0 == noOfInitWorkers) ? spdlog::level::warn : spdlog::level::info;
  spdlog::log(level, "{} worker(s) initialized", noOfInitWorkers);
  return noOfInitWorkers > 0;
}

bool StarPU::ExecuteImpl(const OutputData &out, const InputData &in)
{
  using utils::StarPUUtils;
  // we need at least one initialized worker, otherwise mask would be 0 and all workers
  // would be used
  if (noOfInitWorkers < 1) return false;

  auto CreateArgs = [this, &out, &in]() {
    auto *a = reinterpret_cast<CodeletArgs *>(malloc(sizeof(CodeletArgs)));
    a->ops = &this->ops;
    memcpy(reinterpret_cast<void *>(&a->out), &out.GetData(), sizeof(a->out));
    memcpy(reinterpret_cast<void *>(&a->in), &in.GetData(), sizeof(a->in));
    return a;
  };

  auto *task = taskQueue.emplace(starpu_task_create());
  task->handles[0] = *StarPUUtils::GetHandle(out.GetData().dataInfo);
  task->handles[1] = *StarPUUtils::GetHandle(in.GetData().dataInfo);
  task->workerids = utils::StarPUUtils::CreateWorkerMask(task->workerids_len, ops);
  task->cl_arg = CreateArgs();
  task->cl_arg_size = sizeof(CodeletArgs);
  task->cl_arg_free = 1;
  // make sure we free the mask
  task->callback_func = [](void *) { /* empty on purpose */ };
  task->callback_arg = task->workerids;
  task->callback_arg_free = 1;
  task->detach = 0;// so that we can wait for it
  task->cl = [] {
    static starpu_codelet c = {};
    c.where = STARPU_CUDA | STARPU_CPU;
    c.cpu_funcs[0] = Codelet;
    c.cuda_funcs[0] = Codelet;
    c.cuda_flags[0] = STARPU_CUDA_ASYNC;
    c.nbuffers = 2;
    c.modes[0] = STARPU_RW;
    c.modes[1] = STARPU_R;
    c.model = [] {
      static starpu_perfmodel m = {};
      m.type = STARPU_HISTORY_BASED;
      m.symbol = "Reduction_StarPU";
      return &m;
    }();
    return &c;
  }();

  task->name = this->taskName.c_str();
  STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit %s", this->taskName);
  return true;
}
}// namespace umpalumpa::reduction
