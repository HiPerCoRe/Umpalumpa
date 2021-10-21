#include <libumpalumpa/algorithms/fourier_transformation/fft_cpu.hpp>
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
      auto &fd = in.payload.info;
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
        static_cast<int>(in.payload.info.GetPaddedSize().n),
        in.payload.ptr,
        nullptr,
        1,
        idist,
        out.payload.ptr,
        nullptr,
        1,
        odist,
        flags);
      return tmp;
    }

    struct StrategyFloat : public FFTCPU::Strategy
    {
      ~StrategyFloat() { fftwf_destroy_plan(plan); }

      static constexpr auto kStrategyName = "StrategyFloat";

      bool Init(const AFFT::OutputData &out,
        const AFFT::InputData &in,
        const Settings &settings) override final
      {
        bool canProcess = AFFT::IsFloat(out, in, settings.GetDirection());
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
        // FIXME Since FFTW is very picky about the data alignment etc.,
        // we currently internally reinitialize the algorithm each time Execute() is called
        // Based on the documentation, replanning for existing size should be fast, but
        // still this should be refactored in the future

        fftwf_destroy_plan(plan);// plan has to be valid because Init() should have been called
        this->Init(out, in, s);
        if (s.IsForward()) {
          fftwf_execute_dft_r2c(plan,
            reinterpret_cast<float *>(in.payload.ptr),
            reinterpret_cast<fftwf_complex *>(out.payload.ptr));
        } else {
          fftwf_execute_dft_c2r(plan,
            reinterpret_cast<fftwf_complex *>(in.payload.ptr),
            reinterpret_cast<float *>(out.payload.ptr));
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
        bool canProcess = AFFT::IsDouble(out, in, settings.GetDirection());
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
        // FIXME Since FFTW is very picky about the data alignment etc.,
        // we currently internally reinitialize the algorithm each time Execute() is called
        // Based on the documentation, replanning for existing size should be fast, but
        // still this should be refactored in the future

        fftw_destroy_plan(plan);// plan has to be valid because Init() should have been called
        this->Init(out, in, s);
        if (s.IsForward()) {
          fftw_execute_dft_r2c(plan,
            reinterpret_cast<double *>(in.payload.ptr),
            reinterpret_cast<fftw_complex *>(out.payload.ptr));
        } else {
          fftw_execute_dft_c2r(plan,
            reinterpret_cast<fftw_complex *>(in.payload.ptr),
            reinterpret_cast<double *>(out.payload.ptr));
        }
        return true;
      }

    private:
      fftw_plan plan;
    };

  }// namespace

  std::vector<std::unique_ptr<FFTCPU::Strategy>> FFTCPU::GetStrategies() const
  {
    std::vector<std::unique_ptr<FFTCPU::Strategy>> vec;
    vec.emplace_back(std::make_unique<StrategyFloat>());
    vec.emplace_back(std::make_unique<StrategyDouble>());
    return vec;
  }

}// namespace fourier_transformation
}// namespace umpalumpa