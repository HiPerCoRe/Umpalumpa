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
      auto *valsP = (args->settings.result == SearchResult::kValue)
                      ? reinterpret_cast<AExtremaFinder::ResultData::type *>(buffers[1])
                      : nullptr;
      auto out = AExtremaFinder::ResultData(*valsP, std::nullopt);
      auto *inP = reinterpret_cast<AExtremaFinder::SearchData::type *>(buffers[0]);
      auto in = AExtremaFinder::SearchData(std::move(*inP));
      auto &alg = args->algs->at(static_cast<size_t>(starpu_worker_get_id()));
      alg->Execute(out, in, args->settings);
      alg->Synchronize();// this codelet is run asynchronously, but we have to wait till it's done
                         // to be able to use starpu task synchronization properly
    }

    struct InitArgs
    {
      const AExtremaFinder::ResultData &out;
      const AExtremaFinder::SearchData &in;
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
    using ResultType = data::StarpuPayload<ResultData::type::type>;
    auto oVal = std::make_unique<ResultType>(out.values.value());
    auto oLoc =
      std::make_unique<ResultType>(ResultType::PayloadType(out.values.value().info, "Locations"));
    auto o = StarpuResultData(std::move(oVal), std::move(oLoc));
    using LocalSearchType = data::StarpuPayload<SearchData::type::type>;
    auto i = StarpuSearchData(std::make_unique<LocalSearchType>(in.data));
    return Execute(o, i, settings);
  }

  bool SingleExtremaFinderStarPU::Execute(const StarpuResultData &out,
    const StarpuSearchData &in,
    const Settings &settings)
  {
    struct starpu_task *task = starpu_task_create();
    task->handles[0] = in.data->GetHandle();
    task->handles[1] = out.values.value()->GetHandle();
    task->handles[2] = out.locations.value()->GetHandle();
    task->workerids = CreateWorkerMask(task->workerids_len,
      algs);// FIXME bug in the StarPU? If the mask is completely 0, codelet is being invoked anyway
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
    return true;
  }


}// namespace extrema_finder
}// namespace umpalumpa
