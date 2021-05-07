#include <libumpalumpa/data/starpu_utils.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_starpu.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa {
namespace extrema_finder {

  namespace {// to avoid poluting

    void cpu(void *buffers[], void *func_arg)
    {
      auto *settings = reinterpret_cast<Settings *>(func_arg);
      auto *vals =
        (settings->result == SearchResult::kValue)
          ? reinterpret_cast<umpalumpa::data::Payload<umpalumpa::data::LogicalDescriptor> *>(
            buffers[1])
          : nullptr;
      auto out = ResultData(vals, nullptr);
      auto *in = reinterpret_cast<umpalumpa::extrema_finder::SearchData *>(buffers[0]);
      auto idx = static_cast<size_t>(starpu_worker_get_devid(
        starpu_worker_get_id()));// we have N workers, but they might not be numbered from 0
      auto &alg = SingleExtremaFinderStarPU::Instance().GetCPUAlgorithms().at(idx);
      alg->Execute(out, *in, *settings);
    }

    void gpu(void *buffers[], void *func_arg)
    {
      spdlog::info("SingleExtremaFinderStarPU: Entering GPU codelet");
      auto *settings = reinterpret_cast<Settings *>(func_arg);
      auto *vals =
        (settings->result == SearchResult::kValue)
          ? reinterpret_cast<umpalumpa::data::Payload<umpalumpa::data::LogicalDescriptor> *>(
            buffers[1])
          : nullptr;
      auto out = ResultData(vals, nullptr);
      auto *in = reinterpret_cast<umpalumpa::extrema_finder::SearchData *>(buffers[0]);
      auto idx = static_cast<size_t>(starpu_worker_get_devid(
        starpu_worker_get_id()));// we have N workers, but they might not be numbered from 0
      auto &alg = SingleExtremaFinderStarPU::Instance().GetCUDAAlgorithms().at(idx);
      alg->Execute(out, *in, *settings);

      // FIXME call alg->synch()
    }
  }// namespace

  bool SingleExtremaFinderStarPU::Init(const ResultData &out,
    const SearchData &in,
    const Settings &settings)
  {
    // FIXME refactor to avoid code duplication
    cpuAlgs.clear();
    for (unsigned i = 0; i < starpu_cpu_worker_get_count(); ++i) {
      auto w = std::make_unique<SingleExtremaFinderCPU>();
      if (w->Init(out, in, settings)) { cpuAlgs.emplace_back(std::move(w)); }
    }
    spdlog::debug("Initialized {} CPU workers", cpuAlgs.size());

    cudaAlgs.clear();
    for (unsigned i = 0; i < starpu_cuda_worker_get_count(); ++i) {
      auto w = std::make_unique<SingleExtremaFinderCUDA>(i); // FIXME check that it's the right device index
      if (w->Init(out, in, settings)) { cudaAlgs.emplace_back(std::move(w)); }
    }
    spdlog::debug("Initialized {} CUDA workers", cudaAlgs.size());

    return cpuAlgs.size() > 0;
  }

void SingleExtremaFinderStarPU::Synchronize() {
  for (auto &a : cudaAlgs) {
    reinterpret_cast<SingleExtremaFinderCUDA*>(a.get())->Synchronize();
  }
}

  bool SingleExtremaFinderStarPU::Execute(const ResultData &out,
    const SearchData &in,
    const Settings &settings)
  {
    starpu_data_handle_t hIn = { 0 };
    starpu_payload_register(&hIn, STARPU_MAIN_RAM, const_cast<SearchData &>(in));
    starpu_data_set_name(hIn, in.description.c_str());

    starpu_data_handle_t hVal = { 0 };
    if (nullptr != out.values) {
      starpu_payload_register(&hVal, STARPU_MAIN_RAM, *out.values);
      starpu_data_set_name(hVal, out.values->description.c_str());
    }

    starpu_data_handle_t hLoc = { 0 };
    if (nullptr != out.locations) {
      starpu_payload_register(&hLoc, STARPU_MAIN_RAM, *out.locations);
      starpu_data_set_name(hLoc, out.locations->description.c_str());
    } else {
      starpu_void_data_register(&hLoc);
    }

    struct starpu_task *task = starpu_task_create();
    task->handles[0] = hIn;
    task->handles[1] = hVal;
    task->handles[2] = hLoc;
    task->cl_arg = new Settings(settings);
    task->cl_arg_size = sizeof(Settings);
    task->cl = [] {
      static starpu_codelet c;
      c.where = STARPU_CUDA | STARPU_CPU;
      c.cpu_funcs[0] = cpu;
      c.cuda_funcs[0] = gpu;
      c.nbuffers = 3;
      c.modes[0] = STARPU_R;
      c.modes[1] = STARPU_W;
      c.modes[2] = STARPU_W;
      return &c;
    }();

    task->name = this->taskName.c_str();
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit %s", this->taskName);
    starpu_data_unregister_submit(hIn);
    starpu_data_unregister(hVal); // This will move data to CPU -> there should be some better way how to do it
    starpu_data_unregister_submit(hLoc);
    return true;
  }
}// namespace extrema_finder
}// namespace umpalumpa
