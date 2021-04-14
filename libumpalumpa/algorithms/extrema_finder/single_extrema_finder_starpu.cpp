#include <libumpalumpa/data/starpu_utils.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_starpu.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cpu.hpp>
#include <libumpalumpa/utils/logger.hpp>

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
      auto prg = SingleExtremaFinderCPU();// FIXME the starpu instance has to have its own version
                                          // and call that one, otherwise if init() is used
      prg.Execute(out, *in, *settings);
    }
  }// namespace

  bool SingleExtremaFinderStarPU::Execute(const ResultData &out,
    const SearchData &in,
    const Settings &settings)
  {

    spdlog::info("Welcome to spdlog!");
    spdlog::error("Some error message with arg: {}", 1);
    
    spdlog::warn("Easy padding in numbers like {:08d}", 12);
    spdlog::critical("Support for int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}", 42);
    spdlog::info("Support for floats {:03.2f}", 1.23456);
    spdlog::info("Positional args are {1} {0}..", "too", "supported");
    spdlog::info("{:<30}", "left aligned");
    
    spdlog::set_level(spdlog::level::debug); // Set global log level to debug
    spdlog::debug("This message should be displayed..");    
    
    // change log pattern
    spdlog::set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [thread %t] %v");
    
    // Compile time log levels
    // define SPDLOG_ACTIVE_LEVEL to desired level
    SPDLOG_TRACE("Some trace message with param {}", 42);
    SPDLOG_DEBUG("Some debug message");

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
      c.where = STARPU_CPU;
      c.cpu_funcs[0] = cpu, c.nbuffers = 3;
      c.modes[0] = STARPU_R;
      c.modes[1] = STARPU_W;
      c.modes[2] = STARPU_W;
      return &c;
    }();

    task->name = this->taskName.c_str();
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit %s", this->taskName);
    starpu_data_unregister_submit(hIn);
    starpu_data_unregister_submit(hVal);
    starpu_data_unregister_submit(hLoc);
    return true;
  }
}// namespace extrema_finder
}// namespace umpalumpa
