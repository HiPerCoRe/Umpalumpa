#pragma once

#include <libumpalumpa/algorithms/initialization/abstract.hpp>
#include <tests/algorithms/common.hpp>
#include <tests/utils.hpp>
#include <gmock/gmock.h>

using namespace umpalumpa::initialization;
using namespace umpalumpa::data;
using namespace umpalumpa::test;

class Initialization_Tests : public TestAlg<Abstract>
{
protected:
  const size_t padding = 10;

  auto CreatePayload(const Size &size, const std::string &name)
  {
    auto ld = LogicalDescriptor(size);
    auto type = DataType::Get<float>();
    // allocate extra space, used to ensure we don't access invalid memory
    auto bytes = (ld.Elems() + padding) * type.GetSize();
    auto pd = Create(bytes, type);
    return Payload(ld, std::move(pd), name);
  }

  void SetUp(const Size &sizeData)
  {
    pInOut = std::make_unique<Payload<LogicalDescriptor>>(CreatePayload(sizeData, "InOut"));
    Register(pInOut->dataInfo);

    pVal = std::make_unique<Payload<LogicalDescriptor>>(CreatePayload(Size(1, 1, 1, 1), "Value"));
    Register(pVal->dataInfo);
  }

  /**
   * Called at the end of each test fixture
   **/
  void TearDown() override
  {
    auto Clear = [this](auto &p) {
      Unregister(p->dataInfo);
      Remove(p->dataInfo);
    };

    Clear(pInOut);
    Clear(pVal);
  }

  std::unique_ptr<Payload<LogicalDescriptor>> pInOut;
  std::unique_ptr<Payload<LogicalDescriptor>> pVal;
};
