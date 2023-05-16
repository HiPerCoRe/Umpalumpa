#pragma once

TEST_F(NAME, PiecewiseSum)
{
  auto size = Size(7, 11, 13, 17);
  SetUp(size);

  Settings settings = {};
  settings.SetOperation(Settings::Operation::kPiecewiseSum);

  Acquire(pOut->dataInfo);
  Acquire(pIn->dataInfo);
  FillIncreasing(reinterpret_cast<float *>(pOut->GetPtr()), size.total + padding, 1.f);
  FillIncreasing(reinterpret_cast<float *>(pIn->GetPtr()), size.total + padding, 1.f);
  Release(pIn->dataInfo);
  Release(pOut->dataInfo);

  auto out = Abstract::OutputData(*pOut);
  auto in = Abstract::InputData(*pIn);

  auto &op = GetOp();
  ASSERT_TRUE(op.Init(out, in, settings));
  ASSERT_TRUE(op.Execute(out, in));
  // wait till the work is done
  op.Synchronize();
  // make sure that data are on this memory node

  Acquire(pOut->dataInfo);
  Acquire(pIn->dataInfo);
  for (size_t i = 0; i < size.total; ++i) {
    auto &res = reinterpret_cast<float *>(pOut->GetPtr())[i];
    auto &ref = reinterpret_cast<float *>(pIn->GetPtr())[i];
    ASSERT_EQ(res, ref * 2) << " for i=" << i;
  }
  for (size_t i = size.total; i < size.total + padding; ++i) {
    auto &res = reinterpret_cast<float *>(pOut->GetPtr())[i];
    auto ref = float(i + 1); // we filled array from 1
    ASSERT_EQ(res, ref) << " invalid read / write behind the end of the data at i=" << i;
  }
  Release(pIn->dataInfo);
  Release(pOut->dataInfo);
}