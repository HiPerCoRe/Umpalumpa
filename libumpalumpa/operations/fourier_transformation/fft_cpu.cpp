#include <libumpalumpa/operations/fourier_transformation/fft_cpu.hpp>
#include <fftw3.h>
#include <mutex>

namespace umpalumpa::fourier_transformation {

namespace {// to avoid poluting
  // FIXME for inverse transformation, we need to either copy data to aux array or
  // add some flag to settings stating that user is fine with data rewriting

  // Since planning is not thread safe, we need to have a single mutex for all instances
  static std::mutex mutex;

  struct FloatFFTWHelper
  {
    constexpr static auto kCondition = AFFT::IsFloat;
    constexpr static auto kPlanDestructor = fftwf_destroy_plan;
    constexpr static auto kForwardPlanConstructor = fftwf_plan_many_dft_r2c;
    constexpr static auto kInversePlanConstructor = fftwf_plan_many_dft_c2r;
    constexpr static auto kForwardPlanExecutor = fftwf_execute_dft_r2c;
    constexpr static auto kInversePlanExecutor = fftwf_execute_dft_c2r;
    constexpr static auto kSetThreads = fftwf_plan_with_nthreads;
    constexpr static auto kStrategyName = "StrategyFloat";
    typedef float kSimpleType;
    typedef fftwf_complex kComplexType;
    typedef fftwf_plan kPlanType;

  private:
    struct Init
    {
      Init() { fftwf_init_threads(); }
      // FIXME should we cleanup using these?
      // fftwf_cleanup_threads(void);
      // fftwf_cleanup
      // Problem is that calling them will damage existing plans
    };
    static Init init;
  };

  struct DoubleFFTWHelper
  {
    constexpr static auto kCondition = AFFT::IsDouble;
    constexpr static auto kPlanDestructor = fftw_destroy_plan;
    constexpr static auto kForwardPlanConstructor = fftw_plan_many_dft_r2c;
    constexpr static auto kInversePlanConstructor = fftw_plan_many_dft_c2r;
    constexpr static auto kForwardPlanExecutor = fftw_execute_dft_r2c;
    constexpr static auto kInversePlanExecutor = fftw_execute_dft_c2r;
    constexpr static auto kSetThreads = fftw_plan_with_nthreads;
    constexpr static auto kStrategyName = "StrategyDouble";
    typedef double kSimpleType;
    typedef fftw_complex kComplexType;
    typedef fftw_plan kPlanType;

  private:
    struct Init
    {
      Init() { fftw_init_threads(); }
      // FIXME should we cleanup using these?
      // fftw_cleanup_threads(void);
      // fftw_cleanup
      // Problem is that calling them will damage existing plans
    };
    static Init init;
  };

  template<typename THREAD_SET_FUNCTION, typename PLAN_FUNCTION>
  auto PlanHelper(const AFFT::OutputData &out,
    const AFFT::InputData &in,
    const Settings &settings,
    THREAD_SET_FUNCTION thrSetFunc,
    PLAN_FUNCTION planFunc)
  {
    auto &fd = in.GetData().info;
    auto n = std::array<int, 3>{ static_cast<int>(fd.GetPaddedSpatialSize().z),
      static_cast<int>(fd.GetPaddedSpatialSize().y),
      static_cast<int>(fd.GetPaddedSpatialSize().x) };

    int rank = ToInt(fd.GetPaddedSpatialSize().GetDim());
    size_t offset = 3 - static_cast<size_t>(rank);

    // void *in = nullptr;
    // void *out = settings.isInPlace() ? in : &m_mockOut;

    // no input-preserving operations are implemented for multi-dimensional c2r transforms
    // see http://www.fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags
    auto flags = FFTW_ESTIMATE | (settings.IsForward() ? FFTW_PRESERVE_INPUT : FFTW_DESTROY_INPUT);
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
    thrSetFunc(static_cast<int>(settings.GetThreads()));
    auto tmp = planFunc(rank,
      &n[offset],
      static_cast<int>(in.GetData().info.GetPaddedSize().n),
      in.GetData().GetPtr(),
      nullptr,
      1,
      idist,
      out.GetData().GetPtr(),
      nullptr,
      1,
      odist,
      flags);
    return tmp;
  }

  template<typename T> struct UniversalStrategy final : public FFTCPU::Strategy
  {
    using FFTCPU::Strategy::Strategy;

    ~UniversalStrategy()
    {
      if (op.Get().IsInitialized()) {
        // fftw does not like destroying uninitialized plan
        T::kPlanDestructor(plan);
      }
    }

    bool Init(const AFFT::OutputData &out, const AFFT::InputData &in, const Settings &settings)
    {
      bool canProcess = T::kCondition(out, in, settings.GetDirection());
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
          return T::kForwardPlanConstructor(rank,
            n,
            batch,
            reinterpret_cast<typename T::kSimpleType *>(ptrIn),
            nullptr,
            1,
            idist,
            reinterpret_cast<typename T::kComplexType *>(ptrOut),
            nullptr,
            1,
            odist,
            flags);
        } else {
          return T::kInversePlanConstructor(rank,
            n,
            batch,
            reinterpret_cast<typename T::kComplexType *>(ptrIn),
            nullptr,
            1,
            idist,
            reinterpret_cast<typename T::kSimpleType *>(ptrOut),
            nullptr,
            1,
            odist,
            flags);
        }
      };
      plan = PlanHelper(out, in, settings, T::kSetThreads, f);
      return true;
    }

    bool Init()
    {
      const auto &out = op.Get().GetOutputRef();
      const auto &in = op.Get().GetInputRef();
      const auto &settings = op.Get().GetSettings();
      return Init(out, in, settings);
    }

    std::string GetName() const override final { return T::kStrategyName; }

    bool Execute(const AFFT::OutputData &out, const AFFT::InputData &in) override final
    {
      // FIXME Since FFTW is very picky about the data alignment etc.,
      // we currently internally reinitialize the operation each time Execute() is called
      // Based on the documentation, replanning for existing size should be fast, but
      // still this should be refactored in the future

      T::kPlanDestructor(plan);// plan has to be valid because Init() should have been called
      const auto &s = op.Get().GetSettings();
      this->Init(out, in, s);
      if (s.IsForward()) {
        T::kForwardPlanExecutor(plan,
          reinterpret_cast<typename T::kSimpleType *>(in.GetData().GetPtr()),
          reinterpret_cast<typename T::kComplexType *>(out.GetData().GetPtr()));
      } else {
        T::kInversePlanExecutor(plan,
          reinterpret_cast<typename T::kComplexType *>(in.GetData().GetPtr()),
          reinterpret_cast<typename T::kSimpleType *>(out.GetData().GetPtr()));
      }
      return true;
    }

  private:
    typename T::kPlanType plan;
  };
}// namespace

std::vector<std::unique_ptr<FFTCPU::Strategy>> FFTCPU::GetStrategies() const
{
  std::vector<std::unique_ptr<FFTCPU::Strategy>> vec;
  vec.emplace_back(std::make_unique<UniversalStrategy<FloatFFTWHelper>>(*this));
  vec.emplace_back(std::make_unique<UniversalStrategy<DoubleFFTWHelper>>(*this));
  return vec;
}
}// namespace umpalumpa::fourier_transformation
