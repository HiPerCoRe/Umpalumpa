#pragma once

TEST_F(NAME, FloatToZero)
{
  auto size = Size(7, 11, 13, 17);
  SetUp(size);

  Acquire(pVal->dataInfo);
  Acquire(pInOut->dataInfo);
  FillIncreasing(reinterpret_cast<float *>(pInOut->GetPtr()), size.total + padding, 1.f);
  FillConstant(reinterpret_cast<float *>(pVal->GetPtr()), pVal->info.Elems(), 0.f);
  Release(pInOut->dataInfo);
  Release(pVal->dataInfo);

  auto out = Abstract::OutputData(*pInOut);
  auto in = Abstract::InputData(*pInOut, *pVal);

  auto &alg = GetAlg();
  ASSERT_TRUE(alg.Init(out, in, {}));
  ASSERT_TRUE(alg.Execute(out, in));
  // wait till the work is done
  alg.Synchronize();
  // make sure that data are on this memory node

  Acquire(pInOut->dataInfo);
  auto ref = 0.f;
  for (size_t i = 0; i < size.total; ++i) {
    auto &res = reinterpret_cast<float *>(pInOut->GetPtr())[i];
    ASSERT_EQ(res, ref) << " for i=" << i;
  }
  for (size_t i = size.total; i < size.total + padding; ++i) {
    auto &res = reinterpret_cast<float *>(pInOut->GetPtr())[i];
    auto ref = float(i + 1);// we filled array from 1
    ASSERT_EQ(res, ref) << " invalid read / write behind the end of the data at i=" << i;
  }
  Release(pInOut->dataInfo);
}