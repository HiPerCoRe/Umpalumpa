#pragma once

#include <libumpalumpa/data/size.hpp>
#include <limits>

namespace umpalumpa::reduction {

/**
 * Perform OP(out, in) on each item and store result to out
 **/
template<typename T, typename OP>
bool PiecewiseOp(T *__restrict__ out,
  T *__restrict__ in,
  const umpalumpa::data::Size &size,
  const OP &op)
{
  for (size_t i = 0; i < size.total; ++i) { out[i] = op(out[i], in[i]); }
  return true;
}
}// namespace umpalumpa::reduction
