#pragma once

#include <libumpalumpa/algorithms/fourier_transformation/afft.hpp>
#include <libumpalumpa/algorithms/fourier_processing/afp.hpp>
#include <libumpalumpa/algorithms/fourier_reconstruction/afr.hpp>

#include <random>

using umpalumpa::data::Size;
using umpalumpa::data::Payload;
using umpalumpa::data::FourierDescriptor;
using umpalumpa::data::LogicalDescriptor;
using umpalumpa::data::PhysicalDescriptor;
using umpalumpa::data::DataType;
using umpalumpa::fourier_transformation::AFFT;
using umpalumpa::fourier_processing::AFP;
using umpalumpa::fourier_reconstruction::AFR;

/**
 * This example simulates core functionality of the Fourier Reconstruction.
 * In a nutshell, this program takes several 2D images, and after
 * conversion to the Fourier Domain and crop inserts them
 * under a random orientation into a 3D cube.
 * See http://dx.doi.org/10.1177/1094342019832958 for more details.
 **/
template<typename T> class FourierReconstruction
{
public:
  FourierReconstruction() : generator(42)// fixed seed for reproducibility
  {}

  void Execute(const Size &imgSize, size_t noOfSymmetries, size_t batchSize);

  virtual ~FourierReconstruction() = default;

protected:
  /**
   * This method creates a Physical Payload.
   * If necessary, data should be registered in the respective Memory Manager.
   * If tmp is True, this data are not meant for long-term storage.
   **/
  virtual PhysicalDescriptor CreatePD(size_t bytes, DataType type, bool copyInRAM, bool pinned) = 0;

  /**
   * This method removes all data allocated by the Physical Descriptor
   **/
  virtual void RemovePD(const PhysicalDescriptor &pd, bool pinned) const = 0;

  /**
   * This method fetches data represented by the Physical Descriptor to main RAM.
   * Once not needed in RAM, data should be released.
   **/
  virtual void Acquire(const PhysicalDescriptor &pd) const = 0;

  /**
   * This method release data represented by the Physical Descriptor from main RAM.
   * It has to be called once data fetched by Acquire are no longer needed.
   **/
  virtual void Release(const PhysicalDescriptor &p) const = 0;

  virtual AFFT &GetFFTAlg() const = 0;

  virtual AFP &GetCropAlg() const = 0;

  virtual AFR &GetFRAlg() const = 0;

private:
  using Quaternion = std::array<float, 4>;
  using Matrix3x3 = std::array<float, 9>;

  /**
   * Generate test image content
   **/
  void GenerateData(size_t index, const Payload<LogicalDescriptor> &p);

  /**
   * Generate traverse spaces
   **/
  void GenerateTraverseSpaces(const umpalumpa::data::Size &imgBatchSize,
    const umpalumpa::data::Size &volumeSize,
    const Payload<LogicalDescriptor> &p,
    const std::vector<Matrix3x3> &symmetries,
    const umpalumpa::fourier_reconstruction::Settings &settings);

  /**
   * Generate uniform random rotation in form of the unit quaternion
   * See https://stackoverflow.com/a/56794499
   **/
  auto GenerateQuaternion()
  {
    float x, y, z, u, v, w, s;
    std::uniform_real_distribution<> dis(-1.f, 1.f);
    do {
      x = dis(generator);
      y = dis(generator);
      z = x * x + y * y;
    } while (z > 1);
    do {
      u = dis(generator);
      v = dis(generator);
      w = u * u + v * v;
    } while (w > 1);
    s = std::sqrt((1 - z) / w);
    return Quaternion{ x, y, s * u, s * v };
  }

  /**
   * Generate random rotation matrices representing different symmetries
   **/
  auto GenerateSymmetries(size_t count);

  auto GenerateMatrix() { return ToMatrix(GenerateQuaternion()); }

  /**
   * Multiply two 3D matrices and return the result
   **/
  auto Multiply(const Matrix3x3 &l, const Matrix3x3 &r)
  {
    Matrix3x3 res = {};
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 3; ++j)
        for (size_t k = 0; k < 3; ++k) res[i * 3 + j] += l[i * 3 + k] * r[k * 3 + j];
    return res;
  }

  /**
   * Convert unit quaternion to a 3x3 rotation matrix
   * See https://stackoverflow.com/a/1556470
   **/
  Matrix3x3 ToMatrix(const Quaternion &q)
  {
    auto qx = q[0];
    auto qy = q[1];
    auto qz = q[2];
    auto qw = q[3];
    return { 1.0f - 2.f * qy * qy - 2.f * qz * qz,
      2.f * qx * qy - 2.f * qz * qw,
      2.f * qx * qz + 2.f * qy * qw,// first row
      2.0f * qx * qy + 2.0f * qz * qw,
      1.0f - 2.0f * qx * qx - 2.0f * qz * qz,
      2.0f * qy * qz - 2.0f * qx * qw,// second row
      2.0f * qx * qz - 2.0f * qy * qw,
      2.0f * qy * qz + 2.0f * qx * qw,
      1.0f - 2.0f * qx * qx - 2.0f * qy * qy };
  }

  void Print(const Matrix3x3 &m)
  {
    printf("m=[[% 5.3f, % 5.3f, % 5.3f]\n", m[0], m[1], m[2]);
    printf("   [% 5.3f, % 5.3f, % 5.3f]\n", m[3], m[4], m[5]);
    printf("   [% 5.3f, % 5.3f, % 5.3f]]\n", m[6], m[7], m[8]);
  }


  void Print(const Quaternion &q)
  {
    printf("q=[% 7.7f, % 7.7f, % 7.7f, % 7.7f]\n", q[0], q[1], q[2], q[3]);
  }

  template<typename U>
  void Print(const Payload<U> &p, const std::string &name);

  std::mt19937 generator;

  //   /**
  //    * Move Shift values by specified offset and scale by the specified factor
  //    **/
  //   std::vector<Shift> Transform(const std::vector<Shift> &shift,
  //     float scaleX,
  //     float scaleY,
  //     float shiftX,
  //     float shiftY)
  //   {
  //     auto res = std::vector<Shift>();
  //     for (const auto &s : shift) {
  //       auto x = -(s.x - shiftX) * scaleX;
  //       auto y = -(s.y - shiftY) * scaleY;
  //       res.push_back({ x, y });
  //     }
  //     return res;
  //   }

  /**
   * Create Payload representing a filter applied to the data
   **/
  Payload<LogicalDescriptor> CreatePayloadFilter(const Size &size);

  /**
   * Generate Payload representing (multiple) image(s) of given size.
   **/
  Payload<LogicalDescriptor> CreatePayloadImage(const Size &size, const std::string &name);

  /**
   * Generate Payload representing (multiple) traverse space(s) of given size.
   **/
  Payload<LogicalDescriptor> CreatePayloadTraverseSpace(const Size &size, const std::string &name);

  /**
   * Generate Payload representing a 3D volume holding the inserted projections
   **/
  auto CreatePayloadVolume(const Size &size);

  /**
   * Generate Payload representing a 3D volume holding weigths for inserted projections
   **/
  auto CreatePayloadWeight(const Size &size);

  /**
   * Generate Payload representing a table with interpolation values
   **/
  auto CreatePayloadBlobTable(const umpalumpa::fourier_reconstruction::Settings &settings);

  Payload<FourierDescriptor> ConvertToFFT(const Payload<LogicalDescriptor> &img,
    const std::string &name);

  Payload<FourierDescriptor> Crop(const Payload<FourierDescriptor> &fft,
    Payload<LogicalDescriptor> &filter,
    const std::string &name);

  void InsertToVolume(Payload<FourierDescriptor> &fft,
    Payload<FourierDescriptor> &volume,
    Payload<LogicalDescriptor> &weight,
    Payload<LogicalDescriptor> &traverseSpace,
    Payload<LogicalDescriptor> &table,
    const umpalumpa::fourier_reconstruction::Settings &settings);

  //   Payload<FourierDescriptor> Correlate(Payload<FourierDescriptor> &first,
  //     Payload<FourierDescriptor> &second,
  //     const std::string &name);

  //   Payload<FourierDescriptor> ConvertFromFFT(Payload<FourierDescriptor> &correlation,
  //     const std::string &name);

  //   Payload<LogicalDescriptor> FindMax(Payload<FourierDescriptor> &outCorrelation,
  //     const std::string &name);

  //   std::vector<Shift> ExtractShift(const Payload<LogicalDescriptor> &shift);

  //   void LogResult(size_t i, size_t j, size_t batch, const std::vector<Shift> &shift);

  //   size_t NoOfCorrelations(size_t batch, bool isWithin)
  //   {
  //     // Note: we are wasting some performence by computing intra-buffer correlations
  //     // return isWithin ? ((batch * (batch - 1)) / 2) : (batch * batch);
  //     return batch * batch;
  //   }

  //   size_t NoOfBatches(const Size &s, size_t batch)
  //   {
  //     return ((s.n / batch + 1) * (s.n / batch)) / 2;
  //   }

  size_t GetAvailableCores() const;
};