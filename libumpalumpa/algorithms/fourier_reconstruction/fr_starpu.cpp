#include <libumpalumpa/algorithms/fourier_reconstruction/fr_starpu.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/fr_cuda.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/fr_cpu.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/fr_starpu_kernels.hpp>
#include <libumpalumpa/utils/starpu.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa::fourier_reconstruction {
namespace {// to avoid poluting

  struct Args
  {
    // we need to store local copies of the wrappers
    // as references might not be valid by the time the codelet is executed
    // FIXME this has to be refactored properly to work with MPI
    const AFR::OutputData out;
    const AFR::InputData in;
    const Settings settings;
    std::vector<AFR *> &algs;
  };

  struct CodeletArgs
  {
    data::Payload<data::FourierDescriptor> fft;
    data::Payload<data::FourierDescriptor> volume;
    data::Payload<data::LogicalDescriptor> weight;
    data::Payload<data::LogicalDescriptor> traverseSpace;
    data::Payload<data::LogicalDescriptor> blobTable;
    std::vector<AFR *> *algs;
  };

  static starpu_codelet *GetInitCodelet()
  {
    static starpu_codelet c = {};
    c.where = STARPU_CPU | STARPU_CUDA;
    c.cpu_funcs[0] = InitCodeletCPU;
    c.cuda_funcs[0] = InitCodeletCUDA;
    c.cuda_flags[0] = STARPU_CUDA_ASYNC;
    c.nbuffers = 1;
    c.modes[0] = STARPU_W;
    c.name = "FourierReconstruction_Init";
    c.model = [] {
      static starpu_perfmodel m = {};
      m.type = STARPU_HISTORY_BASED;
      m.symbol = "FourierReconstruction_Init";
      return &m;
    }();
    return &c;
  }

  static starpu_codelet *GetSumCodelet()
  {
    static starpu_codelet c = {};
    c.where = STARPU_CPU | STARPU_CUDA;
    c.cpu_funcs[0] = SumCodeletCPU;
    c.cuda_funcs[0] = SumCodeletCUDA;
    c.cuda_flags[0] = STARPU_CUDA_ASYNC;
    c.nbuffers = 2;
    c.modes[0] = STARPU_RW;
    c.modes[1] = STARPU_R;
    c.name = "FourierReconstruction_Sum";
    c.model = [] {
      static starpu_perfmodel m = {};
      m.type = STARPU_HISTORY_BASED;
      m.symbol = "FourierReconstruction_Sum";
      return &m;
    }();
    return &c;
  }

  void Codelet(void *buffers[], void *func_arg)
  {
    using umpalumpa::utils::StarPUUtils;
    auto *args = reinterpret_cast<CodeletArgs *>(func_arg);

    auto pFFT = StarPUUtils::Assemble(args->fft, buffers[0]);
    auto pVolume = StarPUUtils::Assemble(args->volume, buffers[1]);
    auto pWeight = StarPUUtils::Assemble(args->weight, buffers[2]);
    auto pTraverseSpace = StarPUUtils::Assemble(args->traverseSpace, buffers[3]);
    auto pBlobTable = StarPUUtils::Assemble(args->blobTable, buffers[4]);

    auto out = AFR::OutputData(pVolume, pWeight);
    auto in = AFR::InputData(pFFT, pVolume, pWeight, pTraverseSpace, pBlobTable);

    auto &alg = args->algs->at(static_cast<size_t>(starpu_worker_get_id()));
    std::ignore = alg->Execute(out, in);// we have no way of comunicate the result
    alg->Synchronize();// this codelet is run asynchronously, but we have to wait till it's done
                       // to be able to use starpu task synchronization properly
  }

  void CpuInit(void *args)
  {
    auto *a = reinterpret_cast<Args *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *alg = reinterpret_cast<FRCPU *>(a->algs.at(id));
    if (nullptr == alg) { alg = new FRCPU(); }
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
    auto *alg = reinterpret_cast<FRCUDA *>(a->algs.at(id));
    if (nullptr == alg) {
      std::vector<CUstream> stream = { starpu_cuda_get_local_stream() };
      alg = new FRCUDA(static_cast<int>(id), stream);
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
    auto *vec = reinterpret_cast<std::vector<AFR *> *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *alg = reinterpret_cast<T *>(vec->at(id));
    if (nullptr != alg) { alg->Cleanup(); }
  }

  template<typename T> void DeleteAlg(void *args)
  {
    auto *vec = reinterpret_cast<std::vector<AFR *> *>(args);
    auto id = static_cast<size_t>(starpu_worker_get_id());
    auto *alg = reinterpret_cast<T *>(vec->at(id));
    delete alg;
  }


}// namespace

FRStarPU::~FRStarPU()
{
  if (!this->IsInitialized()) return;
  Cleanup();
  starpu_execute_on_each_worker(DeleteAlg<FRCPU>, &algs, STARPU_CPU);
  starpu_execute_on_each_worker(DeleteAlg<FRCUDA>, &algs, STARPU_CUDA);
}

void FRStarPU::Cleanup()
{
  if (!this->IsInitialized()) return;
  Synchronize();
  starpu_execute_on_each_worker(UniversalCleanup<FRCPU>, &algs, STARPU_CPU);
  starpu_execute_on_each_worker(UniversalCleanup<FRCUDA>, &algs, STARPU_CUDA);
}

void FRStarPU::Synchronize()
{
  while (!taskQueue.empty()) {
    std::ignore = starpu_task_wait(taskQueue.front());
    taskQueue.pop();
  }
}

bool FRStarPU::InitImpl()
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

bool FRStarPU::ExecuteImpl(const OutputData &out, const InputData &in)
{
  using utils::StarPUUtils;
  // we need at least one initialized worker, otherwise mask would be 0 and all workers
  // would be used
  if (noOfInitWorkers < 1) return false;

  auto CreateArgs = [this, &out, &in]() {
    auto *a = reinterpret_cast<CodeletArgs *>(malloc(sizeof(CodeletArgs)));
    a->algs = &this->algs;
    memcpy(reinterpret_cast<void *>(&a->fft), &in.GetFFT(), sizeof(a->fft));
    memcpy(reinterpret_cast<void *>(&a->volume), &in.GetVolume(), sizeof(a->volume));
    memcpy(reinterpret_cast<void *>(&a->weight), &in.GetWeight(), sizeof(a->weight));
    memcpy(reinterpret_cast<void *>(&a->traverseSpace),
      &in.GetTraverseSpace(),
      sizeof(a->traverseSpace));
    memcpy(reinterpret_cast<void *>(&a->blobTable), &in.GetBlobTable(), sizeof(a->blobTable));
    return a;
  };


  auto *task = taskQueue.emplace(starpu_task_create());
  task->handles[0] = *StarPUUtils::GetHandle(in.GetFFT().dataInfo);
  task->handles[1] = *StarPUUtils::GetHandle(in.GetVolume().dataInfo);
  starpu_data_set_reduction_methods(task->handles[1], GetSumCodelet(), GetInitCodelet());
  task->handles[2] = *StarPUUtils::GetHandle(in.GetWeight().dataInfo);
  starpu_data_set_reduction_methods(task->handles[2], GetSumCodelet(), GetInitCodelet());
  task->handles[3] = *StarPUUtils::GetHandle(in.GetTraverseSpace().dataInfo);
  task->handles[4] = *StarPUUtils::GetHandle(in.GetBlobTable().dataInfo);
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
    c.nbuffers = 5;
    c.modes[0] = STARPU_R;
    c.modes[1] = STARPU_REDUX;
    c.modes[2] = STARPU_REDUX;
    c.modes[3] = STARPU_R;
    c.modes[4] = STARPU_R;
    c.model = [] {
      static starpu_perfmodel m = {};
      m.type = STARPU_HISTORY_BASED;
      m.symbol = "FourierReconstruction_StarPU";
      return &m;
    }();
    return &c;
  }();
  task->name = this->taskName.c_str();


  STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit %s", this->taskName);
  return true;
}
}// namespace umpalumpa::fourier_reconstruction
