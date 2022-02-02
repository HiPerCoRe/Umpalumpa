#include <libumpalumpa/algorithms/fourier_transformation/fft_starpu.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/fft_cpu.hpp>
#include <libumpalumpa/algorithms/fourier_transformation/fft_cuda.hpp>
#include <libumpalumpa/utils/starpu.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::fourier_transformation {
namespace {// to avoid poluting
  struct Args
  {
    // we need to store local copies of the wrappers
    // as references might not be valid by the time the codelet is executed
    // FIXME this has to be refactored properly to work with MPI
    const AFFT::OutputData out;
    const AFFT::InputData in;
    const Settings settings;
    std::vector<AFFT *> &algs;
  };

  struct CodeletArgs
  {
    data::Payload<data::FourierDescriptor> out;
    data::Payload<data::FourierDescriptor> in;
    std::vector<AFFT *> *algs;
  };

  void Codelet(void *buffers[], void *func_arg)
  {
    using umpalumpa::utils::StarPUUtils;
    auto *args = reinterpret_cast<CodeletArgs *>(func_arg);

    auto pOut = StarPUUtils::Assemble(args->out, buffers[0]);
    auto out = AFFT::OutputData(pOut);

    auto pIn = StarPUUtils::Assemble(args->in, buffers[1]);
    auto in = AFFT::InputData(pIn);

    auto &alg = args->algs->at(static_cast<size_t>(starpu_worker_get_id()));
    std::ignore = alg->Execute(out, in);// we have no way of comunicate the result
    alg->Synchronize();// this codelet is run asynchronously, but we have to wait till it's done
                       // to be able to use starpu task synchronization properly
  }

  void CpuInit(void *args)
  {
    auto *a = reinterpret_cast<Args *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *alg = reinterpret_cast<FFTCPU *>(a->algs.at(id));
    if (nullptr == alg) { alg = new FFTCPU(); }
    if (!alg->Init(a->out, a->in, a->settings)) {
      delete alg;
      alg = nullptr;
    }
    // update the vector
    a->algs.at(id) = alg;
    // inform StarPU that allocation used some memory
    starpu_memory_allocate(
      starpu_worker_get_local_memory_node(), alg->GetUsedBytes(), STARPU_MEMORY_OVERFLOW);
  }

  void CudaInit(void *args)
  {
    auto *a = reinterpret_cast<Args *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *alg = reinterpret_cast<FFTCUDA *>(a->algs.at(id));
    if (nullptr == alg) {
      std::vector<CUstream> stream = { starpu_cuda_get_local_stream() };
      alg = new FFTCUDA(stream);
    }
    if (!alg->Init(a->out, a->in, a->settings)) {
      delete alg;
      alg = nullptr;
    }
    // update the vector
    a->algs.at(id) = alg;
    // inform StarPU that allocation used some memory
    starpu_memory_allocate(
      starpu_worker_get_local_memory_node(), alg->GetUsedBytes(), STARPU_MEMORY_OVERFLOW);
  }

  template<typename T> void UniversalCleanup(void *args)
  {
    auto *vec = reinterpret_cast<std::vector<AFFT *> *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *alg = reinterpret_cast<T *>(vec->at(id));
    if (nullptr != alg) {
      auto bytes = alg->GetUsedBytes();
      alg->Cleanup();
      starpu_memory_deallocate(starpu_worker_get_local_memory_node(), bytes);
    }
  }

  template<typename T> void DeleteAlg(void *args)
  {
    auto *vec = reinterpret_cast<std::vector<AFFT *> *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *alg = reinterpret_cast<T *>(vec->at(id));
    delete alg;
    vec->at(id) = nullptr;
  }
}// namespace

FFTStarPU::~FFTStarPU()
{
  if (!this->IsInitialized()) return;
  Synchronize();
  Cleanup();
  starpu_execute_on_each_worker(DeleteAlg<FFTCPU>, &algs, STARPU_CPU);
  starpu_execute_on_each_worker(DeleteAlg<FFTCUDA>, &algs, STARPU_CUDA);
}

void FFTStarPU::Cleanup()
{
  if (!this->IsInitialized()) return;
  Synchronize();
  starpu_execute_on_each_worker(UniversalCleanup<FFTCPU>, &algs, STARPU_CPU);
  starpu_execute_on_each_worker(UniversalCleanup<FFTCUDA>, &algs, STARPU_CUDA);
}

void FFTStarPU::Synchronize()
{
  while (!taskQueue.empty()) {
    std::ignore = starpu_task_wait(taskQueue.front());
    taskQueue.pop();
  }
}

bool FFTStarPU::InitImpl()
{
  if (0 == starpu_worker_get_count()) {
    spdlog::warn("No workers available. Is StarPU properly initialized?");
  }
  noOfInitWorkers = 0;
  const auto &out = this->GetOutputRef();
  const auto &in = this->GetInputRef();
  const auto &s = this->GetSettings();
  algs.resize(starpu_worker_get_count(), nullptr);
  Args args = { out, in, s, algs };
  // Allow for multithreading. Each Nth worker will use N threads
  auto cpuIDs = utils::StarPUUtils::GetCPUWorkerIDs(s.GetThreads());
  // FIXME this will not work on multi-memory nodes, because the
  // algs are not updated when returned from worker
  starpu_execute_on_specific_workers(
    CpuInit, &args, static_cast<unsigned>(cpuIDs.size()), cpuIDs.data(), "CPU worker Init");
  starpu_execute_on_each_worker(CudaInit, &args, STARPU_CUDA);
  noOfInitWorkers =
    std::count_if(algs.begin(), algs.end(), [](const auto &i) { return i != nullptr; });
  auto level = (0 == noOfInitWorkers) ? spdlog::level::warn : spdlog::level::info;
  spdlog::log(level, "{} worker(s) initialized", noOfInitWorkers);
  return noOfInitWorkers > 0;
}

bool FFTStarPU::ExecuteImpl(const OutputData &out, const InputData &in)
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
    return a;
  };

  auto *task = taskQueue.emplace(starpu_task_create());
  task->handles[0] = *StarPUUtils::GetHandle(out.GetData().dataInfo);
  task->handles[1] = *StarPUUtils::GetHandle(in.GetData().dataInfo);
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
    c.nbuffers = 2;
    c.modes[0] = STARPU_W;
    c.modes[1] = STARPU_R;
    c.model = [] {
      static starpu_perfmodel m = {};
      m.type = STARPU_HISTORY_BASED;
      m.symbol = "FFT_StarPU";
      return &m;
    }();
    return &c;
  }();

  task->name = this->taskName.c_str();
  STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit %s", this->taskName);
  return true;
}
}// namespace umpalumpa::fourier_transformation
