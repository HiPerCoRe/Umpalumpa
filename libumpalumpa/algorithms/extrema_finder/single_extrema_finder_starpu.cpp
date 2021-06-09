#include <libumpalumpa/data/starpu_utils.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_starpu.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>

namespace umpalumpa {
namespace extrema_finder {

  namespace {// to avoid poluting
    struct ExecuteArgs
    {
      Settings settings;
      const std::vector<std::unique_ptr<AExtremaFinder>> *algs;
    };

    void Codelet(void *buffers[], void *func_arg)
    {
      auto *args = reinterpret_cast<ExecuteArgs *>(func_arg);
      auto *vals =
        (args->settings.result == SearchResult::kValue)
          ? reinterpret_cast<umpalumpa::data::Payload<umpalumpa::data::LogicalDescriptor> *>(
            buffers[1])
          : nullptr;
      auto out = ResultData(vals, nullptr);
      auto *in = reinterpret_cast<umpalumpa::extrema_finder::SearchData *>(buffers[0]);
      auto &alg = args->algs->at(static_cast<size_t>(starpu_worker_get_id()));
      alg->Execute(out, *in, args->settings);
      alg->Synchronize();// this codelet is run asynchronously, but we have to wait till it's done
                         // to be able to use starpu task synchronization properly
    }

    struct InitArgs
    {
      const ResultData &out;
      const SearchData &in;
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
      auto alg = std::make_unique<SingleExtremaFinderCUDA>(starpu_cuda_get_local_stream());
      if (alg->Init(a->out, a->in, a->settings)) {
        a->algs[static_cast<size_t>(starpu_worker_get_id())] = std::move(alg);
      }
    }
  }// namespace

  bool SingleExtremaFinderStarPU::Init(const ResultData &out,
    const SearchData &in,
    const Settings &settings)
  {
    algs.clear();
    algs.resize(starpu_worker_get_count());
    InitArgs args = { out, in, settings, algs };
    starpu_execute_on_each_worker(CpuInit, &args, STARPU_CPU);
    starpu_execute_on_each_worker(
      CudaInit, &args, STARPU_CUDA);// FIXME if one of the workers is not initialized, then we
                                    // should prevent starpu from running execute() on it
    spdlog::info("{} worker(s) initialized",
      std::count_if(algs.begin(), algs.end(), [](const auto &i) { return i != nullptr; }));
    return (algs.size()) > 0;
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
    task->workerids = CreateWorkerMask(task->workerids_len, algs); // FIXME bug in the StarPU? If the mask is completely 0, codelet is being invoked anyway
    task->cl_arg = new ExecuteArgs{ settings, &algs };
    task->cl_arg_size = sizeof(ExecuteArgs);
    task->cl = [] {
      static starpu_codelet c = {};
      c.where = STARPU_CUDA | STARPU_CPU;
      c.cpu_funcs[0] = Codelet;
      c.cuda_funcs[0] = Codelet;
      c.nbuffers = 3;
      c.modes[0] = STARPU_R;
      c.modes[1] = STARPU_W;
      c.modes[2] = STARPU_W;
      return &c;
    }();

    task->name = this->taskName.c_str();
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit %s", this->taskName);
    starpu_data_unregister_submit(hIn);// unregister data at leasure
    starpu_data_unregister(hVal);// copy results back to home node
    starpu_data_unregister(hLoc);// copy results back to home node
    return true;
  }
}// namespace extrema_finder
}// namespace umpalumpa
