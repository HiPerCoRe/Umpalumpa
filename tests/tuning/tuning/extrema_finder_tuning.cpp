#include <tests/algorithms/extrema_finder/extrema_finder_common.hpp>
#include <libumpalumpa/algorithms/extrema_finder/single_extrema_finder_cuda.hpp>

using namespace umpalumpa::tuning;

class SingleExtremaFinderTuningTest : public ExtremaFinder_Tests
{
public:
  SingleExtremaFinderCUDA &GetAlg() override { return transformer; }

  using ExtremaFinder_Tests::SetUp;

  PhysicalDescriptor Create(size_t bytes, DataType type) override
  {
    void *ptr;
    CudaErrchk(cudaMallocManaged(&ptr, bytes));
    return PhysicalDescriptor(ptr, bytes, type, ManagedBy::CUDA, nullptr);
  }

  void Remove(const PhysicalDescriptor &pd) override { CudaErrchk(cudaFree(pd.GetPtr())); }

  void Register(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Unregister(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Acquire(const PhysicalDescriptor &pd) override
  {
    CudaErrchk(cudaMemPrefetchAsync(pd.GetPtr(), pd.GetBytes(), worker));
  }

  void Release(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

private:
  const int worker = 0;
  SingleExtremaFinderCUDA transformer = SingleExtremaFinderCUDA(worker);
};

// TESTS

TEST_F(SingleExtremaFinderTuningTest, subsequent_executions_use_best_found_configuration)
{
  auto size = Size(9, 1, 1, 5);

  auto refLocs = std::make_unique<float[]>(size.n * size.GetDimAsNumber());

  auto settings =
    Settings(ExtremaType::kMax, Location::kEntire, Result::kLocation, Precision::k3x3);

  SetUp(settings, size);

  {
    Acquire(pData->dataInfo);
    memset(pData->GetPtr(), 0, pData->dataInfo.GetBytes());
    auto *ptr = reinterpret_cast<float *>(pData->GetPtr());
    auto SetSignal = [&size, ptr](size_t n) { return ptr + n * size.single; };
    {
      // in the range
      size_t s = 0;
      SetSignal(s)[6] = 0.1;
      SetSignal(s)[7] = 1;
      SetSignal(s)[8] = 0.5;
      refLocs[s] = 7.25;
    }
    {
      // only one value
      size_t s = 1;
      SetSignal(s)[1] = 1;
      refLocs[s] = 1;
    }
    {
      // crop first values
      size_t s = 2;
      SetSignal(s)[0] = 1;
      SetSignal(s)[1] = 0.6;
      refLocs[s] = 0.375;
    }
    {
      // crop last values
      size_t s = 3;
      SetSignal(s)[7] = 0.6;
      SetSignal(s)[8] = 1;
      refLocs[s] = 7.625;
    }
    {
      // in the range, non-unit values
      size_t s = 4;
      SetSignal(s)[6] = 0.3;
      SetSignal(s)[7] = 3;
      SetSignal(s)[8] = 1.5;
      refLocs[s] = 7.25;
    }
    Release(pData->dataInfo);
  }

  auto out = AExtremaFinder::OutputData(*pValues, *pLocations);
  auto in = AExtremaFinder::InputData(*pData);

  auto &alg = GetAlg();

  // 1. execution with tuning

  ASSERT_TRUE(alg.Init(out, in, settings));
  // After Init there is a strategy that we can work with!
  // HERE WE SET THE TUNING
  dynamic_cast<TunableStrategy &>(alg.GetStrategy())
    .SetTuningApproach(TuningApproach::kEntireStrategy);

  // Now finish the rest of the test!
  ASSERT_TRUE(alg.Execute(out, in));
  // wait till the work is done
  alg.Synchronize();

  // check results
  Acquire(pLocations->dataInfo);
  auto *actualLocs = reinterpret_cast<float *>(pLocations->GetPtr());
  for (size_t n = 0; n < size.n; ++n) {
    ASSERT_FLOAT_EQ(refLocs[n], actualLocs[n]) << " for n=" << n;
  }
  Release(pLocations->dataInfo);

  // 2. execution without tuning

  ASSERT_TRUE(alg.Init(out, in, settings));
  // After Init there is a strategy that we can work with!
  // HERE WE SET THE TUNING
  dynamic_cast<TunableStrategy &>(alg.GetStrategy()).SetTuningApproach(TuningApproach::kNoTuning);

  // Now finish the rest of the test!
  ASSERT_TRUE(alg.Execute(out, in));
  // wait till the work is done
  alg.Synchronize();

  // check results
  Acquire(pLocations->dataInfo);
  actualLocs = reinterpret_cast<float *>(pLocations->GetPtr());
  for (size_t n = 0; n < size.n; ++n) {
    ASSERT_FLOAT_EQ(refLocs[n], actualLocs[n]) << " for n=" << n;
  }
  Release(pLocations->dataInfo);

  // 3. execution with tuning

  ASSERT_TRUE(alg.Init(out, in, settings));
  // After Init there is a strategy that we can work with!
  // HERE WE SET THE TUNING
  dynamic_cast<TunableStrategy &>(alg.GetStrategy())
    .SetTuningApproach(TuningApproach::kEntireStrategy);

  // Now finish the rest of the test!
  ASSERT_TRUE(alg.Execute(out, in));
  // wait till the work is done
  alg.Synchronize();

  // check results
  Acquire(pLocations->dataInfo);
  actualLocs = reinterpret_cast<float *>(pLocations->GetPtr());
  for (size_t n = 0; n < size.n; ++n) {
    ASSERT_FLOAT_EQ(refLocs[n], actualLocs[n]) << " for n=" << n;
  }
  Release(pLocations->dataInfo);

  // 4. execution without tuning

  ASSERT_TRUE(alg.Init(out, in, settings));
  // After Init there is a strategy that we can work with!
  // HERE WE SET THE TUNING
  dynamic_cast<TunableStrategy &>(alg.GetStrategy()).SetTuningApproach(TuningApproach::kNoTuning);

  // Now finish the rest of the test!
  ASSERT_TRUE(alg.Execute(out, in));
  // wait till the work is done
  alg.Synchronize();

  // check results
  Acquire(pLocations->dataInfo);
  actualLocs = reinterpret_cast<float *>(pLocations->GetPtr());
  for (size_t n = 0; n < size.n; ++n) {
    ASSERT_FLOAT_EQ(refLocs[n], actualLocs[n]) << " for n=" << n;
  }
  Release(pLocations->dataInfo);
}

TEST_F(SingleExtremaFinderTuningTest, only_FindMax_kernel_should_be_tuned)
{
  auto size = Size(9, 1, 1, 5);

  auto refLocs = std::make_unique<float[]>(size.n * size.GetDimAsNumber());

  auto settings =
    Settings(ExtremaType::kMax, Location::kEntire, Result::kLocation, Precision::k3x3);

  SetUp(settings, size);

  {
    Acquire(pData->dataInfo);
    memset(pData->GetPtr(), 0, pData->dataInfo.GetBytes());
    auto *ptr = reinterpret_cast<float *>(pData->GetPtr());
    auto SetSignal = [&size, ptr](size_t n) { return ptr + n * size.single; };
    {
      // in the range
      size_t s = 0;
      SetSignal(s)[6] = 0.1;
      SetSignal(s)[7] = 1;
      SetSignal(s)[8] = 0.5;
      refLocs[s] = 7.25;
    }
    {
      // only one value
      size_t s = 1;
      SetSignal(s)[1] = 1;
      refLocs[s] = 1;
    }
    {
      // crop first values
      size_t s = 2;
      SetSignal(s)[0] = 1;
      SetSignal(s)[1] = 0.6;
      refLocs[s] = 0.375;
    }
    {
      // crop last values
      size_t s = 3;
      SetSignal(s)[7] = 0.6;
      SetSignal(s)[8] = 1;
      refLocs[s] = 7.625;
    }
    {
      // in the range, non-unit values
      size_t s = 4;
      SetSignal(s)[6] = 0.3;
      SetSignal(s)[7] = 3;
      SetSignal(s)[8] = 1.5;
      refLocs[s] = 7.25;
    }
    Release(pData->dataInfo);
  }

  auto out = AExtremaFinder::OutputData(*pValues, *pLocations);
  auto in = AExtremaFinder::InputData(*pData);

  auto &alg = GetAlg();

  // 1. execution with tuning

  ASSERT_TRUE(alg.Init(out, in, settings));
  // After Init there is a strategy that we can work with!
  // HERE WE SET THE TUNING
  auto &strat = dynamic_cast<TunableStrategy &>(alg.GetStrategy());
  strat.SetTuningApproach(TuningApproach::kSelectedKernels);
  // TODO
  // strat.SetTuningFor()

  // Now finish the rest of the test!
  ASSERT_TRUE(alg.Execute(out, in));
  // wait till the work is done
  alg.Synchronize();

  // check results
  Acquire(pLocations->dataInfo);
  auto *actualLocs = reinterpret_cast<float *>(pLocations->GetPtr());
  for (size_t n = 0; n < size.n; ++n) {
    ASSERT_FLOAT_EQ(refLocs[n], actualLocs[n]) << " for n=" << n;
  }
  Release(pLocations->dataInfo);

  // 2. execution without tuning

  ASSERT_TRUE(alg.Init(out, in, settings));
  // After Init there is a strategy that we can work with!
  // HERE WE SET THE TUNING
  dynamic_cast<TunableStrategy &>(alg.GetStrategy()).SetTuningApproach(TuningApproach::kNoTuning);

  // Now finish the rest of the test!
  ASSERT_TRUE(alg.Execute(out, in));
  // wait till the work is done
  alg.Synchronize();

  // check results
  Acquire(pLocations->dataInfo);
  actualLocs = reinterpret_cast<float *>(pLocations->GetPtr());
  for (size_t n = 0; n < size.n; ++n) {
    ASSERT_FLOAT_EQ(refLocs[n], actualLocs[n]) << " for n=" << n;
  }
  Release(pLocations->dataInfo);

  // 3. execution with tuning

  ASSERT_TRUE(alg.Init(out, in, settings));
  // After Init there is a strategy that we can work with!
  // HERE WE SET THE TUNING
  dynamic_cast<TunableStrategy &>(alg.GetStrategy())
    .SetTuningApproach(TuningApproach::kEntireStrategy);

  // Now finish the rest of the test!
  ASSERT_TRUE(alg.Execute(out, in));
  // wait till the work is done
  alg.Synchronize();

  // check results
  Acquire(pLocations->dataInfo);
  actualLocs = reinterpret_cast<float *>(pLocations->GetPtr());
  for (size_t n = 0; n < size.n; ++n) {
    ASSERT_FLOAT_EQ(refLocs[n], actualLocs[n]) << " for n=" << n;
  }
  Release(pLocations->dataInfo);

  // 4. execution without tuning

  ASSERT_TRUE(alg.Init(out, in, settings));
  // After Init there is a strategy that we can work with!
  // HERE WE SET THE TUNING
  dynamic_cast<TunableStrategy &>(alg.GetStrategy()).SetTuningApproach(TuningApproach::kNoTuning);

  // Now finish the rest of the test!
  ASSERT_TRUE(alg.Execute(out, in));
  // wait till the work is done
  alg.Synchronize();

  // check results
  Acquire(pLocations->dataInfo);
  actualLocs = reinterpret_cast<float *>(pLocations->GetPtr());
  for (size_t n = 0; n < size.n; ++n) {
    ASSERT_FLOAT_EQ(refLocs[n], actualLocs[n]) << " for n=" << n;
  }
  Release(pLocations->dataInfo);
}
