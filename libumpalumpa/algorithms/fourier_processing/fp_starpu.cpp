#include <libumpalumpa/algorithms/fourier_processing/fp_starpu.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_cpu.hpp>
#include <libumpalumpa/algorithms/fourier_processing/fp_cuda.hpp>
#include <libumpalumpa/utils/starpu.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::fourier_processing {
namespace {// to avoid poluting

  struct Args
  {
    // we need to store local copies of the wrappers
    // as references might not be valid by the time the codelet is executed
    // FIXME this has to be refactored properly to work with MPI
    const AFP::OutputData out;
    const AFP::InputData in;
    const Settings settings;
    std::vector<AFP *> &algs;
  };

  struct CodeletArgs
  {
    data::Payload<data::FourierDescriptor> out;
    data::Payload<data::FourierDescriptor> in;
    data::Payload<data::LogicalDescriptor> filter;
    std::vector<AFP *> *algs;
  };

  void Codelet(void *buffers[], void *func_arg)
  {
    using umpalumpa::utils::StarPUUtils;
    auto *args = reinterpret_cast<CodeletArgs *>(func_arg);

    auto pOut = StarPUUtils::Assemble(args->out, buffers[0]);
    auto out = AFP::OutputData(pOut);

    auto pInData = StarPUUtils::Assemble(args->in, buffers[1]);
    auto pInFilter = StarPUUtils::Assemble(args->filter, buffers[2]);
    auto in = AFP::InputData(pInData, pInFilter);

    auto &alg = args->algs->at(static_cast<size_t>(starpu_worker_get_id()));
    std::ignore = alg->Execute(out, in);// we have no way of comunicate the result
    alg->Synchronize();// this codelet is run asynchronously, but we have to wait till it's done
                       // to be able to use starpu task synchronization properly
  }

  void CpuInit(void *args)
  {
    auto *a = reinterpret_cast<Args *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *alg = reinterpret_cast<FPCPU *>(a->algs.at(id));
    if (nullptr == alg) { alg = new FPCPU(); }
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
    auto *alg = reinterpret_cast<FPCUDA *>(a->algs.at(id));
    if (nullptr == alg) {
      std::vector<CUstream> stream = { starpu_cuda_get_local_stream() };
      alg = new FPCUDA(static_cast<int>(id), stream);
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
    auto *vec = reinterpret_cast<std::vector<AFP *> *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *alg = reinterpret_cast<T *>(vec->at(id));
    if (nullptr != alg) { alg->Cleanup(); }
  }

  template<typename T> void DeleteAlg(void *args)
  {
    auto *vec = reinterpret_cast<std::vector<AFP *> *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *alg = reinterpret_cast<T *>(vec->at(id));
    delete alg;
  }
}// namespace

FPStarPU::~FPStarPU()
{
  if (!this->IsInitialized()) return;
  Synchronize();
  starpu_execute_on_each_worker(DeleteAlg<FPCPU>, &algs, STARPU_CPU);
  starpu_execute_on_each_worker(DeleteAlg<FPCUDA>, &algs, STARPU_CUDA);
}

void FPStarPU::Cleanup()
{
  if (!this->IsInitialized()) return;
  Synchronize();
  starpu_execute_on_each_worker(UniversalCleanup<FPCPU>, &algs, STARPU_CPU);
  starpu_execute_on_each_worker(UniversalCleanup<FPCUDA>, &algs, STARPU_CUDA);
  AFP::Cleanup();
}

void FPStarPU::Synchronize()
{
  while (!taskQueue.empty()) {
    std::ignore = starpu_task_wait(taskQueue.front());
    taskQueue.pop();
  }
}

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
  starpu_execute_on_each_worker(CudaInit, &args, STARPU_CUDA);
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

  auto CreateArgs = [this, &out, &in]() {
    auto *a = reinterpret_cast<CodeletArgs *>(malloc(sizeof(CodeletArgs)));
    a->algs = &this->algs;
    memcpy(reinterpret_cast<void *>(&a->out), &out.GetData(), sizeof(a->out));
    memcpy(reinterpret_cast<void *>(&a->in), &in.GetData(), sizeof(a->in));
    memcpy(reinterpret_cast<void *>(&a->filter), &in.GetFilter(), sizeof(a->filter));
    return a;
  };

  auto *task = taskQueue.emplace(starpu_task_create());
  task->handles[0] = *StarPUUtils::GetHandle(out.GetData().dataInfo);
  task->handles[1] = *StarPUUtils::GetHandle(in.GetData().dataInfo);
  task->handles[2] = *StarPUUtils::GetHandle(in.GetFilter().dataInfo);
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
    c.cuda_flags[0] = STARPU_CUDA_ASYNC;
    c.nbuffers = 3;
    c.modes[0] = STARPU_W;
    c.modes[1] = STARPU_R;
    c.modes[2] = STARPU_R;
    c.model = [] {
      static starpu_perfmodel m = {};
      m.type = STARPU_HISTORY_BASED;
      m.symbol = "FourierProcessing_StarPU";
      return &m;
    }();
    return &c;
  }();
  task->name = this->taskName.c_str();
  STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit %s", this->taskName);
  return true;
}
}// namespace umpalumpa::fourier_processing
