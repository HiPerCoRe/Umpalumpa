#include <libumpalumpa/algorithms/initialization/cpu.hpp>
#include <cstring>// memset
namespace umpalumpa::initialization {

namespace {// to avoid poluting
  struct BasicInit final : public CPU::Strategy
  {
    // Inherit constructor
    using CPU::Strategy::Strategy;

    bool Init() override final
    {
      return !alg.GetInputRef().GetData().info.IsPadded()
             && alg.GetInputRef().GetValue().dataInfo.GetType().Is<float>();
    }

    std::string GetName() const override { return "Strategy"; }

    bool Execute(const Abstract::OutputData &, const Abstract::InputData &in) override
    {
      auto IsFine = [](const auto &p) { return p.IsValid() && !p.IsEmpty(); };
      if (!IsFine(in.GetData()) || !IsFine(in.GetValue())) return false;

      if (0.f == reinterpret_cast<float *>(in.GetValue().GetPtr())[0]) {
        const auto &d = in.GetData();
        memset(d.GetPtr(), 0, d.GetRequiredBytes());
        return true;
      }
      return false;
    }
  };

}// namespace

std::vector<std::unique_ptr<CPU::Strategy>> CPU::GetStrategies() const
{
  std::vector<std::unique_ptr<CPU::Strategy>> vec;
  vec.emplace_back(std::make_unique<BasicInit>(*this));
  return vec;
}
}// namespace umpalumpa::initialization
