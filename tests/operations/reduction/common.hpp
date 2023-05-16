#pragma once

#include <libumpalumpa/operations/reduction/abstract.hpp>
#include <tests/operations/common.hpp>
#include <tests/utils.hpp>
#include <gmock/gmock.h>

using namespace umpalumpa::reduction;
using namespace umpalumpa::data;
using namespace umpalumpa::test;

class Reduction_Tests : public TestOp<Abstract>
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
    pIn = std::make_unique<Payload<LogicalDescriptor>>(CreatePayload(sizeData, "In"));
    Register(pIn->dataInfo);

    pOut = std::make_unique<Payload<LogicalDescriptor>>(CreatePayload(sizeData, "Out"));
    Register(pOut->dataInfo);
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

    Clear(pIn);
    Clear(pOut);
  }

  std::unique_ptr<Payload<LogicalDescriptor>> pIn;
  std::unique_ptr<Payload<LogicalDescriptor>> pOut;
};
