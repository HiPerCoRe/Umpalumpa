#pragma once

TEST_F(NAME, 1D_batch_noPadd_max_valOnly)
{
  auto size = Size(10, 1, 1, 3);
  auto settings = Settings(ExtremaType::kMax, Location::kEntire, Result::kValue);

  SetUp(settings, size);

  auto dataOrig = std::unique_ptr<float[]>(new float[pData->info.Elems()]);
  FillNormalDist(dataOrig.get(), pData->info.Elems());

  Acquire(pData->dataInfo);
  memcpy(dataOrig.get(), pData->GetPtr(), pData->dataInfo.GetBytes());
  // Print(*pData, pData->info.GetPaddedSize());
  Release(pData->dataInfo);

  auto out = AExtremaFinder::OutputData(*pValues, *pLocations);
  auto in = AExtremaFinder::InputData(*pData);

  auto &alg = GetAlg();
  ASSERT_TRUE(alg.Init(out, in, settings));
  ASSERT_TRUE(alg.Execute(out, in));
  // wait till the work is done
  alg.Synchronize();
  // make sure that data are on this memory node

  Acquire(pData->dataInfo);
  // check results
  // Print(inData, in.GetData().info.GetPaddedSize());
  // make sure that we didn't change data
  ASSERT_EQ(0, memcmp(pData->GetPtr(), dataOrig.get(), pData->dataInfo.GetBytes()));
  Release(pData->dataInfo);
  // test that we found good maximas
  CheckValues();
}

TEST_F(NAME, 3D_batch_noPadd_max_valOnly)
{
  auto size = Size(120, 173, 150, 103);
  std::cout << "This test will need at least " << size.total * sizeof(float) / 1048576 << " MB"
            << std::endl;
  auto settings = Settings(ExtremaType::kMax, Location::kEntire, Result::kValue);

  SetUp(settings, size);

  Acquire(pData->dataInfo);
  FillRandom(pData->GetPtr(), pData->dataInfo.GetBytes());
  Release(pData->dataInfo);

  auto out = AExtremaFinder::OutputData(*pValues, *pLocations);
  auto in = AExtremaFinder::InputData(*pData);

  auto &alg = GetAlg();
  ASSERT_TRUE(alg.Init(out, in, settings));
  ASSERT_TRUE(alg.Execute(out, in));
  // wait till the work is done
  alg.Synchronize();
  // check results
  CheckValues();
}

TEST_F(NAME, 2D_batch_noPadd_max_rectCenter_posOnly)
{
  auto size = Size(120, 100, 1, 3);

  auto settings = Settings(ExtremaType::kMax, Location::kRectCenter, Result::kLocation);

  SetUp(settings, size);

  Acquire(pData->dataInfo);
  FillIncreasing(reinterpret_cast<float *>(pData->GetPtr()), pData->info.GetSize().total, 0.f);
  Release(pData->dataInfo);

  auto out = AExtremaFinder::OutputData(*pValues, *pLocations);
  auto in = AExtremaFinder::InputData(*pData);

  auto &alg = GetAlg();
  ASSERT_TRUE(alg.Init(out, in, settings));
  ASSERT_TRUE(alg.Execute(out, in));
  // wait till the work is done
  alg.Synchronize();

  // check results
  Acquire(pLocations->dataInfo);
  // FIXME these values should be read from settings
  Size searchRect(28, 17, 1, 1);
  size_t searchRectOffsetX = (in.GetData().info.GetPaddedSize().x - searchRect.x) / 2;
  size_t searchRectOffsetY = (in.GetData().info.GetPaddedSize().y - searchRect.y) / 2;
  // test that we found good maximas
  for (int n = 0; n < size.n; ++n) {
    auto expectedResult = searchRectOffsetY * size.x + searchRectOffsetX;// beginning of searchRect
    expectedResult += size.x * (searchRect.y - 1) + (searchRect.x - 1);// last element of searchRect
    auto actualResult = reinterpret_cast<float *>(pLocations->GetPtr())[n];
    ASSERT_FLOAT_EQ(expectedResult, actualResult) << " for n=" << n;
  }
  Release(pLocations->dataInfo);
}
