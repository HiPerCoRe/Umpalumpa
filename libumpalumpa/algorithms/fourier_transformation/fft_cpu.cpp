#include <libumpalumpa/algorithms/fourier_transformation/fft_cpu.hpp>
#include <libumpalumpa/system_includes/spdlog.hpp>
#include <fftw3.h>
#include <mutex>

namespace umpalumpa {
namespace fourier_transformation {

  namespace {// to avoid poluting
    // FIXME for inverse transformation, we need to either copy data to aux array or
    // add some flag to settings stating that user is fine with data rewriting

    // Since planning is not thread safe, we need to have a single mutex for all instances
    static std::mutex mutex;

    template<typename F>
    auto PlanHelper(const AFFT::OutputData &out,
      const AFFT::InputData &in,
      const Settings &settings,
      F function)
    {
      auto &fd = in.data.info;
      auto n = std::array<int, 3>{ static_cast<int>(fd.GetPaddedSpatialSize().z),
        static_cast<int>(fd.GetPaddedSpatialSize().y),
        static_cast<int>(fd.GetPaddedSpatialSize().x) };

      int rank = ToInt(fd.GetPaddedSpatialSize().GetDim());
      size_t offset = 3 - static_cast<size_t>(rank);

      // void *in = nullptr;
      // void *out = settings.isInPlace() ? in : &m_mockOut;

      // no input-preserving algorithms are implemented for multi-dimensional c2r transforms
      // see http://www.fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags
      auto flags =
        FFTW_ESTIMATE | (settings.IsForward() ? FFTW_PRESERVE_INPUT : FFTW_DESTROY_INPUT);
      // if (!isDataAligned) { flags = flags | FFTW_UNALIGNED; }

      int idist;
      int odist;
      if (settings.IsForward()) {
        idist = static_cast<int>(fd.GetPaddedSpatialSize().single);
        odist = static_cast<int>(fd.GetFrequencySize().single);
        // We know that data type is either float or double (validated before)
      } else {
        idist = static_cast<int>(fd.GetPaddedFrequencySize().single);
        odist = static_cast<int>(fd.GetPaddedSpatialSize().single);
      }
      // set threads
      // fftw_plan_with_nthreads(threads);
      // fftwf_plan_with_nthreads(threads);
      
      // only one thread can create a plan
      std::lock_guard<std::mutex> lck(mutex);
      auto tmp = function(rank,
        &n[offset],
        static_cast<int>(in.data.info.GetPaddedSize().n),
        in.data.ptr,
        nullptr,
        1,
        idist,
        out.data.ptr,
        nullptr,
        1,
        odist,
        flags);
      return tmp;
    }

    bool IsDouble(const AFFT::OutputData &out, const AFFT::InputData &in, Direction d)
    {
      if (Direction::kForward == d) {
        return ((out.data.dataInfo.type == data::DataType::kComplexDouble)
                && (in.data.dataInfo.type == data::DataType::kDouble));
      }
      return ((out.data.dataInfo.type == data::DataType::kDouble)
              && (in.data.dataInfo.type == data::DataType::kComplexDouble));
    }

    bool IsFloat(const AFFT::OutputData &out, const AFFT::InputData &in, Direction d)
    {
      if (Direction::kForward == d) {
        return ((out.data.dataInfo.type == data::DataType::kComplexFloat)
                && (in.data.dataInfo.type == data::DataType::kFloat));
      }
      return ((out.data.dataInfo.type == data::DataType::kFloat)
              && (in.data.dataInfo.type == data::DataType::kComplexFloat));
    }

    struct StrategyFloat : public FFTCPU::Strategy
    {
      ~StrategyFloat() { fftwf_destroy_plan(plan); }

      static constexpr auto kStrategyName = "StrategyFloat";

      bool Init(const AFFT::OutputData &out,
        const AFFT::InputData &in,
        const Settings &settings) override final
      {
        bool canProcess = IsFloat(out, in, settings.GetDirection());
        if (!canProcess) return false;
        auto f = [&settings](int rank,
                   const int *n,
                   int batch,
                   void *ptrIn,
                   const int *,
                   int,
                   int idist,
                   void *ptrOut,
                   const int *,
                   int,
                   int odist,
                   unsigned flags) {
          if (settings.IsForward()) {
            return fftwf_plan_many_dft_r2c(rank,
              n,
              batch,
              reinterpret_cast<float *>(ptrIn),
              nullptr,
              1,
              idist,
              reinterpret_cast<fftwf_complex *>(ptrOut),
              nullptr,
              1,
              odist,
              flags);
          } else {
            return fftwf_plan_many_dft_c2r(rank,
              n,
              batch,
              reinterpret_cast<fftwf_complex *>(ptrIn),
              nullptr,
              1,
              idist,
              reinterpret_cast<float *>(ptrOut),
              nullptr,
              1,
              odist,
              flags);
          }
        };
        plan = PlanHelper(out, in, settings, f);
        return true;
      }

      std::string GetName() const override final { return kStrategyName; }

      bool Execute(const AFFT::OutputData &out,
        const AFFT::InputData &in,
        const Settings &s) override final
      {
        if (!in.data.IsValid() || in.data.IsEmpty() || !out.data.IsValid() || out.data.IsEmpty())
          return false;
        // FIXME Since FFTW is very picky about the data alignment etc.,
        // we currently internally reinitialize the algorithm each time Execute() is called
        // Based on the documentation, replanning for existing size should be fast, but
        // still this should be refactored in the future

        fftwf_destroy_plan(plan);// plan has to be valid because Init() should have been called
        this->Init(out, in, s);
        if (s.IsForward()) {
          fftwf_execute_dft_r2c(plan,
            reinterpret_cast<float *>(in.data.ptr),
            reinterpret_cast<fftwf_complex *>(out.data.ptr));
        } else {
          fftwf_execute_dft_c2r(plan,
            reinterpret_cast<fftwf_complex *>(in.data.ptr),
            reinterpret_cast<float *>(out.data.ptr));
        }
        return true;
      }

    private:
      fftwf_plan plan;
    };

    struct StrategyDouble : public FFTCPU::Strategy
    {

      ~StrategyDouble() { fftw_destroy_plan(plan); }

      static constexpr auto kStrategyName = "StrategyDouble";

      bool Init(const AFFT::OutputData &out,
        const AFFT::InputData &in,
        const Settings &settings) override final
      {
        bool canProcess = IsDouble(out, in, settings.GetDirection());
        if (!canProcess) return false;
        auto f = [&settings](int rank,
                   const int *n,
                   int batch,
                   void *ptrIn,
                   const int *,
                   int,
                   int idist,
                   void *ptrOut,
                   const int *,
                   int,
                   int odist,
                   unsigned flags) {
          if (settings.IsForward()) {
            return fftw_plan_many_dft_r2c(rank,
              n,
              batch,
              reinterpret_cast<double *>(ptrIn),
              nullptr,
              1,
              idist,
              reinterpret_cast<fftw_complex *>(ptrOut),
              nullptr,
              1,
              odist,
              flags);
          } else {
            return fftw_plan_many_dft_c2r(rank,
              n,
              batch,
              reinterpret_cast<fftw_complex *>(ptrIn),
              nullptr,
              1,
              idist,
              reinterpret_cast<double *>(ptrOut),
              nullptr,
              1,
              odist,
              flags);
          }
        };
        plan = PlanHelper(out, in, settings, f);
        return true;
      }

      std::string GetName() const override final { return kStrategyName; }

      bool Execute(const AFFT::OutputData &out,
        const AFFT::InputData &in,
        const Settings &s) override final
      {
        if (!in.data.IsValid() || in.data.IsEmpty() || !out.data.IsValid() || out.data.IsEmpty())
          return false;
        // FIXME Since FFTW is very picky about the data alignment etc.,
        // we currently internally reinitialize the algorithm each time Execute() is called
        // Based on the documentation, replanning for existing size should be fast, but
        // still this should be refactored in the future

        fftw_destroy_plan(plan);// plan has to be valid because Init() should have been called
        this->Init(out, in, s);
        if (s.IsForward()) {
          fftw_execute_dft_r2c(plan,
            reinterpret_cast<double *>(in.data.ptr),
            reinterpret_cast<fftw_complex *>(out.data.ptr));
        } else {
          fftw_execute_dft_c2r(plan,
            reinterpret_cast<fftw_complex *>(in.data.ptr),
            reinterpret_cast<double *>(out.data.ptr));
        }
        return true;
      }

    private:
      fftw_plan plan;
    };

  }// namespace

  bool FFTCPU::Init(const OutputData &out, const InputData &in, const Settings &s)
  {
    if (IsInitialized()) { strategy.reset(); }
    SetSettings(s);
    auto tryToAdd = [this, &out, &in, &s](auto i) {
      bool canAdd = i->Init(out, in, s);
      if (canAdd) {
        spdlog::debug("Found valid strategy {}", i->GetName());
        strategy = std::move(i);
      }
      return canAdd;
    };

    return tryToAdd(std::make_unique<StrategyFloat>())
           || tryToAdd(std::make_unique<StrategyDouble>());
  }

  bool FFTCPU::Execute(const OutputData &out, const InputData &in)
  {
    if (!this->IsInitialized()) return false;
    return strategy->Execute(out, in, GetSettings());
  }
}// namespace fourier_transformation
}// namespace umpalumpa