#include <libumpalumpa/operations/reduction/cpu.hpp>
#include <libumpalumpa/operations/reduction/cpu_kernels.hpp>

namespace umpalumpa::reduction {

namespace {// to avoid poluting
  struct PiecewiseSum final : public CPU::Strategy
  {
    // Inherit constructor
    using CPU::Strategy::Strategy;

    bool Init() override final
    {
      bool isValidOp = op.Get().GetSettings().GetOperation() == Settings::Operation::kPiecewiseSum;
      return isValidOp && !op.GetInputRef().GetData().info.IsPadded()
             && !op.GetOutputRef().GetData().info.IsPadded();
    }

    std::string GetName() const override { return "PiecewiseSum"; }

    bool Execute(const Abstract::OutputData &out, const Abstract::InputData &in) override
    {
      auto IsFine = [](const auto &p) { return p.IsValid() && !p.IsEmpty(); };
      if (!IsFine(in.GetData()) || !IsFine(out.GetData())) return false;

      if (in.GetData().dataInfo.GetType().Is<float>()) {
        return PiecewiseOp(reinterpret_cast<float *>(out.GetData().GetPtr()),
          reinterpret_cast<float *>(in.GetData().GetPtr()),
          in.GetData().info.GetSize(),
          [](auto l, auto r) { return l + r; });
      }
      return false;
    }
  };

}// namespace

std::vector<std::unique_ptr<CPU::Strategy>> CPU::GetStrategies() const
{
  std::vector<std::unique_ptr<CPU::Strategy>> vec;
  vec.emplace_back(std::make_unique<PiecewiseSum>(*this));
  return vec;
}
}// namespace umpalumpa::reduction
