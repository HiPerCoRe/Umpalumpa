#pragma once

TEST_F(NAME, 1D_batch_noPadd_max_valOnly)
{
  auto size = Size(10, 1, 1, 3);
  auto settings = Settings(ExtremaType::kMax, Location::kEntire, Result::kValue);

  SetUp(settings, size);

  auto dataOrig = std::unique_ptr<float[]>(new float[pData->info.Elems()]);
  FillNormalDist(dataOrig.get(), pData->info.Elems());

  Acquire(pData->dataInfo);
  memcpy(pData->GetPtr(), dataOrig.get(), pData->dataInfo.GetBytes());
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
  memset(pData->GetPtr(), 0, pData->dataInfo.GetBytes()); // not needed, but Valgrind doesnt like reading random bytes
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

TEST_F(NAME, 3D_max_entire_locOnly_singlePrecision)
{
  auto size = Size(120, 100, 3, 3);
  TestLocsMaxEntireSingle(size);
}

TEST_F(NAME, 2D_max_entire_locOnly_singlePrecision)
{
  auto size = Size(101, 103, 1, 5);
  TestLocsMaxEntireSingle(size);
}


TEST_F(NAME, 1D_max_entire_locOnly_singlePrecision)
{
  auto size = Size(107, 1, 1, 7);
  TestLocsMaxEntireSingle(size);
}


TEST_F(NAME, 1D_max_entire_locOnly_Precision3x3)
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
  ASSERT_TRUE(alg.Init(out, in, settings));
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
}

TEST_F(NAME, 2D_max_entire_locOnly_Precision3x3)
{
  auto size = Size(9, 9, 1, 5);

  auto refLocs = std::make_unique<float[]>(size.n * size.GetDimAsNumber());

  auto settings =
    Settings(ExtremaType::kMax, Location::kEntire, Result::kLocation, Precision::k3x3);

  SetUp(settings, size);

  {
    Acquire(pData->dataInfo);
    memset(pData->GetPtr(), 0, pData->dataInfo.GetBytes());
    auto *ptr = reinterpret_cast<float *>(pData->GetPtr());
    auto GetSignal = [&size, ptr](
                       size_t n) { return reinterpret_cast<float(&)[9][9]>(ptr[n * size.single]); };
    size_t s = 0;
    {
      // only one value
      auto *sig = GetSignal(s);
      sig[3][3] = 1.f;
      refLocs[2 * s] = 3;// X
      refLocs[2 * s + 1] = 3;// Y
      ++s;
    }
    {
      // in the range, cross + X
      auto *sig = GetSignal(s);
      sig[3][3] = 1.f;
      sig[2][3] = sig[3][2] = sig[4][3] = sig[3][4] = 0.6f;// cross
      sig[2][2] = sig[4][2] = sig[2][4] = sig[4][4] = 0.1f;// X
      refLocs[2 * s] = 3;// X
      refLocs[2 * s + 1] = 3;// Y
      ++s;
    }
    {
      // in the range, cross + X, non-unit values
      auto *sig = GetSignal(s);
      sig[3][3] = 3.f;
      sig[2][3] = sig[3][2] = sig[4][3] = sig[3][4] = 1.8f;// cross
      sig[2][2] = sig[4][2] = sig[2][4] = sig[4][4] = 0.3f;// X
      refLocs[2 * s] = 3;// X
      refLocs[2 * s + 1] = 3;// Y
      ++s;
    }
    {
      // crop top left corner
      auto *sig = GetSignal(s);
      sig[2][2] = 1.f;
      sig[2][3] = sig[3][2] = 0.5f;
      refLocs[2 * s] = 2.25;// X
      refLocs[2 * s + 1] = 2.25;// Y
      ++s;
    }
    {
      // crop bottom right corner
      auto *sig = GetSignal(s);
      sig[8][8] = 1.f;
      sig[7][8] = sig[8][7] = 0.5f;
      refLocs[2 * s] = 7.75;// X
      refLocs[2 * s + 1] = 7.75;// Y
      ++s;
    }
    ASSERT_EQ(s, size.n);
    Release(pData->dataInfo);
  }

  auto out = AExtremaFinder::OutputData(*pValues, *pLocations);
  auto in = AExtremaFinder::InputData(*pData);

  auto &alg = GetAlg();
  ASSERT_TRUE(alg.Init(out, in, settings));
  ASSERT_TRUE(alg.Execute(out, in));
  // wait till the work is done
  alg.Synchronize();

  // check results
  Acquire(pLocations->dataInfo);
  auto *actualLocs = reinterpret_cast<float *>(pLocations->GetPtr());
  for (size_t n = 0; n < size.n; ++n) {
    ASSERT_FLOAT_EQ(refLocs[2 * n], actualLocs[2 * n]) << " for X and n=" << n;
    ASSERT_FLOAT_EQ(refLocs[2 * n + 1], actualLocs[2 * n + 1]) << " for Y and n=" << n;
  }
  Release(pLocations->dataInfo);
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
