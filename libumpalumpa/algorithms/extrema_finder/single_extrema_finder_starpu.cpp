#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_starpu.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda.hpp>
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
    std::vector<AExtremaFinder *> &algs;
  };

  struct CodeletArgs
  {
    data::Payload<data::LogicalDescriptor> vals;
    data::Payload<data::LogicalDescriptor> locs;
    data::Payload<data::LogicalDescriptor> in;
    std::vector<AExtremaFinder *> *algs;
  };

  void Codelet(void *buffers[], void *func_arg)
  {
    using umpalumpa::utils::StarPUUtils;
    auto *args = reinterpret_cast<CodeletArgs *>(func_arg);

    // FIXME if handle points to the void interface, we access illegal memory
    auto pVals = StarPUUtils::Assemble(args->vals, buffers[0]);
    auto pLocs = StarPUUtils::Assemble(args->locs, buffers[1]);
    auto out = AExtremaFinder::OutputData(pVals, pLocs);

    auto pIn = StarPUUtils::Assemble(args->in, buffers[2]);
    auto in = AExtremaFinder::InputData(pIn);

    auto &alg = args->algs->at(static_cast<size_t>(starpu_worker_get_id()));
    std::ignore = alg->Execute(out, in);// we have no way of comunicate the result
    alg->Synchronize();// this codelet is run asynchronously, but we have to wait till it's done
                       // to be able to use starpu task synchronization properly
  }
  void CpuInit(void *args)
  {
    auto *a = reinterpret_cast<Args *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *alg = reinterpret_cast<SingleExtremaFinderCPU *>(a->algs.at(id));
    if (nullptr == alg) { alg = new SingleExtremaFinderCPU(); }
    if (!alg->Init(a->out, a->in, a->settings)) {
      delete alg;
      alg = nullptr;
    }
    // update the vector
    a->algs.at(id) = alg;
  }

  void CudaInit(void *args)
  {
    auto *a = reinterpret_cast<Args *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *alg = reinterpret_cast<SingleExtremaFinderCUDA *>(a->algs.at(id));
    if (nullptr == alg) {
      std::vector<CUstream> stream = { starpu_cuda_get_local_stream() };
      alg = new SingleExtremaFinderCUDA(static_cast<int>(id), stream);
    }
    if (!alg->Init(a->out, a->in, a->settings)) {
      delete alg;
      alg = nullptr;
    }
    // update the vector
    a->algs.at(id) = alg;
  }

  template<typename T> void UniversalCleanup(void *args)
  {
    auto *vec = reinterpret_cast<std::vector<AExtremaFinder *> *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *alg = reinterpret_cast<T *>(vec->at(id));
    if (nullptr != alg) { alg->Cleanup(); }
  }

  template<typename T> void DeleteAlg(void *args)
  {
    auto *vec = reinterpret_cast<std::vector<AExtremaFinder *> *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *alg = reinterpret_cast<T *>(vec->at(id));
    delete alg;
  }
}// namespace

SingleExtremaFinderStarPU::~SingleExtremaFinderStarPU()
{
  if (!this->IsInitialized()) return;
  Synchronize();
  starpu_execute_on_each_worker(DeleteAlg<SingleExtremaFinderCPU>, &algs, STARPU_CPU);
  starpu_execute_on_each_worker(DeleteAlg<SingleExtremaFinderCUDA>, &algs, STARPU_CUDA);
}

void SingleExtremaFinderStarPU::Cleanup()
{
  if (!this->IsInitialized()) return;
  Synchronize();
  starpu_execute_on_each_worker(UniversalCleanup<SingleExtremaFinderCPU>, &algs, STARPU_CPU);
  starpu_execute_on_each_worker(UniversalCleanup<SingleExtremaFinderCUDA>, &algs, STARPU_CUDA);
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
  algs.resize(starpu_worker_get_count());
  Args args = { out, in, s, algs };
  starpu_execute_on_each_worker(CpuInit, &args, STARPU_CPU);
  starpu_execute_on_each_worker(CudaInit, &args, STARPU_CUDA);
  noOfInitWorkers =
    std::count_if(algs.begin(), algs.end(), [](const auto &i) { return i != nullptr; });
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
    a->algs = &this->algs;
    memcpy(reinterpret_cast<void *>(&a->vals), &out.GetValues(), sizeof(a->vals));
    memcpy(reinterpret_cast<void *>(&a->locs), &out.GetLocations(), sizeof(a->locs));
    memcpy(reinterpret_cast<void *>(&a->in), &in.GetData(), sizeof(a->in));
    return a;
  };

  auto *task = taskQueue.emplace(starpu_task_create());
  task->handles[0] = *StarPUUtils::GetHandle(out.GetValues().dataInfo);
  task->handles[1] = *StarPUUtils::GetHandle(out.GetLocations().dataInfo);
  task->handles[2] = *StarPUUtils::GetHandle(in.GetData().dataInfo);
  task->workerids = utils::StarPUUtils::CreateWorkerMask(task->workerids_len, algs);
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
