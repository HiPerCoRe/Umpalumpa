#include <tests/algorithms/correlation/acorrelation_common.hpp>
#include <libumpalumpa/algorithms/correlation/correlation_cuda.hpp>
#include <algorithm>

using namespace umpalumpa::tuning;

class CorrelationTuningTest : public Correlation_Tests
{
public:
  Correlation_CUDA &GetAlg() override { return transformer; }

  using Correlation_Tests::SetUp;

  PhysicalDescriptor Create(size_t bytes, DataType type) override
  {
    void *ptr;
    CudaErrchk(cudaMallocManaged(&ptr, bytes));
    memset(ptr, 0, bytes);
    memory.emplace_back(ptr);
    return PhysicalDescriptor(ptr, bytes, type, ManagedBy::CUDA, nullptr);
  }

  PhysicalDescriptor Copy(const PhysicalDescriptor &pd) override
  {
    return pd.CopyWithPtr(pd.GetPtr());
  }

  void Remove(const PhysicalDescriptor &pd) override
  {
    if (auto it = std::find(memory.begin(), memory.end(), pd.GetPtr()); memory.end() != it) {
      CudaErrchk(cudaFree(pd.GetPtr()));
      memory.erase(it);
    }
  }

  void Register(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Unregister(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  void Acquire(const PhysicalDescriptor &pd) override
  {
    CudaErrchk(cudaMemPrefetchAsync(pd.GetPtr(), pd.GetBytes(), worker));
  }

  void Release(const PhysicalDescriptor &pd) override{ /* nothing to do */ };

  TestProcessingUnit GetTestProcessingUnit() const override { return TestProcessingUnit::kGPU; }

private:
  const int worker = 0;
  Correlation_CUDA transformer = Correlation_CUDA(worker);
  std::vector<void *> memory;
};

// TESTS

TEST_F(CorrelationTuningTest, subsequent_executions_use_best_found_configuration)
{
  Settings settings(CorrelationType::kMToN);

  Size inSizeTmp(20, 20, 1, 3);

  SetUp(settings, inSizeTmp, inSizeTmp.CopyFor(2), false);

  // Have to be here, because we NEED to run SetUp otherwise it is a Seg. fault
  if (GetTestProcessingUnit() != TestProcessingUnit::kGPU) { return; }

  auto in = ACorrelation::InputData(*pData1, *pData2);
  auto out = ACorrelation::OutputData(*pOut);

  auto mt = std::mt19937(42);
  auto dist = std::normal_distribution<float>((float)0, (float)1);
  auto ip = [&mt, &dist](std::complex<float> *arr, size_t size) {
    for (size_t i = 0; i < size; i++) { arr[i] = { dist(mt), dist(mt) }; }
  };

  auto *input1 = reinterpret_cast<std::complex<float> *>(in.GetData1().GetPtr());
  auto *input2 = reinterpret_cast<std::complex<float> *>(in.GetData2().GetPtr());
  auto inSize = in.GetData1().info.GetSize();
  auto inSize2 = in.GetData2().info.GetSize();

  auto &corr = GetAlg();
  float delta = 0.00001f;

  // 1. execution with tuning

  Acquire(in.GetData1().dataInfo);
  Acquire(in.GetData2().dataInfo);
  ip(input1, inSize.total);
  if (input1 != input2) { ip(input2, inSize2.total); }
  Release(in.GetData2().dataInfo);
  Release(in.GetData1().dataInfo);

  ASSERT_TRUE(corr.Init(out, in, settings));
  // After Init there is a strategy that we can work with!
  // HERE WE SET THE TUNING
  dynamic_cast<TunableStrategy &>(corr.GetStrategy())
    .SetTuningApproach(TuningApproach::kEntireStrategy);

  // Now finish the rest of the test!
  ASSERT_TRUE(corr.Execute(out, in));
  corr.Synchronize();

  // make sure that data are on this memory node
  Acquire(out.GetCorrelations().dataInfo);
  // check results
  check(reinterpret_cast<std::complex<float> *>(out.GetCorrelations().GetPtr()),
    settings,
    input1,
    inSize,
    input2,
    inSize2,
    delta);
  // we're done with those data
  Release(out.GetCorrelations().dataInfo);

  // 2. execution without tuning

  Acquire(in.GetData1().dataInfo);
  Acquire(in.GetData2().dataInfo);
  ip(input1, inSize.total);
  if (input1 != input2) { ip(input2, inSize2.total); }
  Release(in.GetData2().dataInfo);
  Release(in.GetData1().dataInfo);

  ASSERT_TRUE(corr.Init(out, in, settings));
  // After Init there is a strategy that we can work with!
  // HERE WE SET THE TUNING
  dynamic_cast<TunableStrategy &>(corr.GetStrategy()).SetTuningApproach(TuningApproach::kNoTuning);

  // Now finish the rest of the test!
  ASSERT_TRUE(corr.Execute(out, in));
  corr.Synchronize();

  // make sure that data are on this memory node
  Acquire(out.GetCorrelations().dataInfo);
  // check results
  check(reinterpret_cast<std::complex<float> *>(out.GetCorrelations().GetPtr()),
    settings,
    input1,
    inSize,
    input2,
    inSize2,
    delta);
  // we're done with those data
  Release(out.GetCorrelations().dataInfo);

  // 3. execution with tuning

  Acquire(in.GetData1().dataInfo);
  Acquire(in.GetData2().dataInfo);
  ip(input1, inSize.total);
  if (input1 != input2) { ip(input2, inSize2.total); }
  Release(in.GetData2().dataInfo);
  Release(in.GetData1().dataInfo);

  ASSERT_TRUE(corr.Init(out, in, settings));
  // After Init there is a strategy that we can work with!
  // HERE WE SET THE TUNING
  dynamic_cast<TunableStrategy &>(corr.GetStrategy())
    .SetTuningApproach(TuningApproach::kEntireStrategy);

  // Now finish the rest of the test!
  ASSERT_TRUE(corr.Execute(out, in));
  corr.Synchronize();

  // make sure that data are on this memory node
  Acquire(out.GetCorrelations().dataInfo);
  // check results
  check(reinterpret_cast<std::complex<float> *>(out.GetCorrelations().GetPtr()),
    settings,
    input1,
    inSize,
    input2,
    inSize2,
    delta);
  // we're done with those data
  Release(out.GetCorrelations().dataInfo);

  // 4. execution without tuning

  Acquire(in.GetData1().dataInfo);
  Acquire(in.GetData2().dataInfo);
  ip(input1, inSize.total);
  if (input1 != input2) { ip(input2, inSize2.total); }
  Release(in.GetData2().dataInfo);
  Release(in.GetData1().dataInfo);

  ASSERT_TRUE(corr.Init(out, in, settings));
  // After Init there is a strategy that we can work with!
  // HERE WE SET THE TUNING
  dynamic_cast<TunableStrategy &>(corr.GetStrategy()).SetTuningApproach(TuningApproach::kNoTuning);

  // Now finish the rest of the test!
  ASSERT_TRUE(corr.Execute(out, in));
  corr.Synchronize();

  // make sure that data are on this memory node
  Acquire(out.GetCorrelations().dataInfo);
  // check results
  check(reinterpret_cast<std::complex<float> *>(out.GetCorrelations().GetPtr()),
    settings,
    input1,
    inSize,
    input2,
    inSize2,
    delta);
  // we're done with those data
  Release(out.GetCorrelations().dataInfo);
}
