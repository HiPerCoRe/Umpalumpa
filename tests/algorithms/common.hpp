#pragma once

#include <gtest/gtest.h>
#include <libumpalumpa/data/physical_desriptor.hpp>
#include <tests/algorithms/test_processing_unit.hpp>

namespace umpalumpa::test {

using data::PhysicalDescriptor;
using data::DataType;

/**
 * Class responsible for testing.
 * Specific implementation of the algorithms should inherit from it.
 **/
template<typename T> class TestAlg : public ::testing::Test
{
protected:
  virtual T &GetAlg() = 0;
  /**
   * Methods to create / delete Payload Descriptor.
   * Payload is expected to point to a valid data location.
   * Data location does not have to be initialized to specific value.
   * It's test responsibility to do so)
   **/
  virtual PhysicalDescriptor Create(size_t bytes, DataType type) = 0;
  virtual void Remove(const PhysicalDescriptor &pd) = 0;
  /**
   * Methods that will register some memory in the memory manager, if any.
   **/
  virtual void Register(const PhysicalDescriptor &pd) = 0;
  virtual void Unregister(const PhysicalDescriptor &pd) = 0;
  /**
   * Methods that will bring data to the current memory node.
   * These methods will be called before / after the results are checked.
   * If necessary, these method should be blocking.
   **/
  virtual void Acquire(const PhysicalDescriptor &pd) = 0;
  virtual void Release(const PhysicalDescriptor &pd) = 0;
  /**
   * Serves for distinguishing tests that can be run only at some specific processing unit.
   **/
  virtual TestProcessingUnit GetTestProcessingUnit() const { return TestProcessingUnit::kAny; }
};
}// namespace umpalumpa::test
