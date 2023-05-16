#include <libumpalumpa/operations/extrema_finder/single_extrema_finder_starpu.hpp>
#include <libumpalumpa/operations/extrema_finder/single_extrema_finder_cpu.hpp>
#include <libumpalumpa/operations/extrema_finder/single_extrema_finder_cuda.hpp>
#include <libumpalumpa/utils/starpu.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::extrema_finder {

namespace {// to avoid poluting
  struct Args
  {
    // we need to store local copies of the wrappers
    // as references might not be valid by the time the codelet is executed
    // FIXME this has to be refactored properly to work with MPI
    const AExtremaFinder::OutputData out;
    const AExtremaFinder::InputData in;
    const Settings settings;
    std::vector<AExtremaFinder *> &ops;
  };

  struct CodeletArgs
  {
    data::Payload<data::LogicalDescriptor> vals;
    data::Payload<data::LogicalDescriptor> locs;
    data::Payload<data::LogicalDescriptor> in;
    std::vector<AExtremaFinder *> *ops;
  };

  void Codelet(void *buffers[], void *func_arg)
  {
    using umpalumpa::utils::StarPUUtils;
    auto *args = reinterpret_cast<CodeletArgs *>(func_arg);

    auto pVals = StarPUUtils::Assemble(args->vals, buffers[0]);
    auto pLocs = StarPUUtils::Assemble(args->locs, buffers[1]);
    auto out = AExtremaFinder::OutputData(pVals, pLocs);

    auto pIn = StarPUUtils::Assemble(args->in, buffers[2]);
    auto in = AExtremaFinder::InputData(pIn);

    auto &op = args->ops->at(static_cast<size_t>(starpu_worker_get_id()));
    std::ignore = op->Execute(out, in);// we have no way of comunicate the result
    op->Synchronize();// this codelet is run asynchronously, but we have to wait till it's done
                       // to be able to use starpu task synchronization properly
  }
  void CpuInit(void *args)
  {
    auto *a = reinterpret_cast<Args *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *op = reinterpret_cast<SingleExtremaFinderCPU *>(a->ops.at(id));
    if (nullptr == op) { op = new SingleExtremaFinderCPU(); }
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
    auto *op = reinterpret_cast<SingleExtremaFinderCUDA *>(a->ops.at(id));
    if (nullptr == op) {
      std::vector<CUstream> stream = { starpu_cuda_get_local_stream() };
      op = new SingleExtremaFinderCUDA(static_cast<int>(id), stream);
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
    auto *vec = reinterpret_cast<std::vector<AExtremaFinder *> *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *op = reinterpret_cast<T *>(vec->at(id));
    if (nullptr != op) { op->Cleanup(); }
  }

  template<typename T> void DeleteOp(void *args)
  {
    auto *vec = reinterpret_cast<std::vector<AExtremaFinder *> *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *op = reinterpret_cast<T *>(vec->at(id));
    delete op;
  }
}// namespace

SingleExtremaFinderStarPU::~SingleExtremaFinderStarPU()
{
  if (!this->IsInitialized()) return;
  Synchronize();
  starpu_execute_on_each_worker(DeleteOp<SingleExtremaFinderCPU>, &ops, STARPU_CPU);
  starpu_execute_on_each_worker(DeleteOp<SingleExtremaFinderCUDA>, &ops, STARPU_CUDA);
}

void SingleExtremaFinderStarPU::Cleanup()
{
  if (!this->IsInitialized()) return;
  Synchronize();
  starpu_execute_on_each_worker(UniversalCleanup<SingleExtremaFinderCPU>, &ops, STARPU_CPU);
  starpu_execute_on_each_worker(UniversalCleanup<SingleExtremaFinderCUDA>, &ops, STARPU_CUDA);
  AExtremaFinder::Cleanup();
}

void SingleExtremaFinderStarPU::Synchronize()
{
  while (!taskQueue.empty()) {
    std::ignore = starpu_task_wait(taskQueue.front());
    taskQueue.pop();
  }
}

bool SingleExtremaFinderStarPU::InitImpl()
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

bool SingleExtremaFinderStarPU::ExecuteImpl(const OutputData &out, const InputData &in)
{
  using utils::StarPUUtils;
  // we need at least one initialized worker, otherwise mask would be 0 and all workers
  // would be used
  if (noOfInitWorkers < 1) return false;

  auto CreateArgs = [this, &out, &in]() {
    auto *a = reinterpret_cast<CodeletArgs *>(malloc(sizeof(CodeletArgs)));
    a->ops = &this->ops;
    memcpy(reinterpret_cast<void *>(&a->vals), &out.GetValues(), sizeof(a->vals));
    memcpy(reinterpret_cast<void *>(&a->locs), &out.GetLocations(), sizeof(a->locs));
    memcpy(reinterpret_cast<void *>(&a->in), &in.GetData(), sizeof(a->in));
    return a;
  };

  auto *task = taskQueue.emplace(starpu_task_create());
  task->handles[0] = *StarPUUtils::GetHandle(out.GetValues().dataInfo);
  task->handles[1] = *StarPUUtils::GetHandle(out.GetLocations().dataInfo);
  task->handles[2] = *StarPUUtils::GetHandle(in.GetData().dataInfo);
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
    c.nbuffers = 3;
    c.modes[0] = STARPU_W;
    c.modes[1] = STARPU_W;
    c.modes[2] = STARPU_R;
    c.model = [] {
      static starpu_perfmodel m = {};
      m.type = STARPU_HISTORY_BASED;
      m.symbol = "SingleExtremaFinder_StarPU";
      return &m;
    }();
    return &c;
  }();

  task->name = this->taskName.c_str();
  STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit %s", this->taskName);
  return true;
}
}// namespace umpalumpa::extrema_finder
