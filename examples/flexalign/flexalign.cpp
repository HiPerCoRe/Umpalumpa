#include "flexalign.hpp"
#include <cassert>
#include <iostream>
#include <vector>

template<typename T> void FlexAlign<T>::Execute(const umpalumpa::data::Size &size)
{
  assert(size.x > 5);
  assert(size.y > 5);
  assert(size.z == 1);

  auto cropSize = Size(size.x / 2, size.y / 2, size.z, size.n);
  auto crossSize = Size(3, 3, 1, 1);

  auto images = std::vector<std::unique_ptr<Payload<LogicalDescriptor>>>();
  images.reserve(size.n);
  auto ffts = std::vector<std::unique_ptr<Payload<FourierDescriptor>>>();
  ffts.reserve(size.n);

  for (size_t j = 0; j < size.n; ++j) {
    images.emplace_back(Generate(j, size));
    auto &img = *images.at(j).get();
    GenerateCross(j, img, crossSize, 0, 0);
    ffts.emplace_back(ConvertToFFTAndCrop(j, img, cropSize));
    for (size_t i = 0; i < j; ++i) {
      auto correlation = Correlate(i, j, *ffts.at(i), *ffts.at(j));
      auto shift = FindMax(i, j, *correlation);
      std::cout << "Shift of img " << i << " and " << j << " is [" << shift.x << ", " << shift.y
                << "]\n";
    }
  }
}

template<typename T>
typename FlexAlign<T>::Shift
  FlexAlign<T>::FindMax(size_t i, size_t j, Payload<FourierDescriptor> &correlation)
{
  std::cout << "FindMax correlation " << i << " and " << j << "\n";
  return { static_cast<float>(i), static_cast<float>(j) };
};

template<typename T>
std::unique_ptr<Payload<FourierDescriptor>> FlexAlign<T>::Correlate(size_t i,
  size_t j,
  Payload<FourierDescriptor> &first,
  Payload<FourierDescriptor> &second)
{
  std::cout << "Correlate img " << i << " and " << j << "\n";
  auto ld = FourierDescriptor(first.info.GetSize());
  return std::make_unique<Payload<FourierDescriptor>>(ld, "");
}

template<typename T>
std::unique_ptr<Payload<LogicalDescriptor>> FlexAlign<T>::Generate(size_t index, const Size &size)
{
  std::cout << "Generating image " << index << "\n";
  auto ld = LogicalDescriptor(size);
  return std::make_unique<Payload<LogicalDescriptor>>(ld, "");
};

template<typename T>
void FlexAlign<T>::GenerateCross(size_t index,
  const Payload<LogicalDescriptor> &p,
  const Size &crossSize,
  size_t x,
  size_t y)
{
  //   assert(p.IsValid() && !p.IsEmpty());
  std::cout << "Generating cross " << index << "\n";
}

template class FlexAlign<float>;