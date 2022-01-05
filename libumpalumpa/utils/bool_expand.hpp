#pragma once

namespace umpalumpa::utils {

/**
 * This structure allows you to expand bool arguments into template parameters.
 * To do so, you need bool-templated structure, with (optionally templated) static void Execute()
 * method:
 * @code
 * template<bool b>
 * struct Example {
 *  template<typename T>
 *  static void Execute(T *t) {...}
 * };
 * @endcode
 *
 * To execute it, you simply:
 * @code
 * ExpandBools<Example>::Expand(GetBool(), GetT());
 * @endcode
 **/
template<template<bool...> class Function, bool... Bs> struct ExpandBools
{
  template<typename... Args> static void Expand(Args &&...args)
  {
    return Function<Bs...>::Execute(std::forward<Args>(args)...);
  }

  template<typename... Args> static void Expand(bool b, Args &&...args)
  {
    return b ? ExpandBools<Function, Bs..., true>::Expand(std::forward<Args>(args)...)
             : ExpandBools<Function, Bs..., false>::Expand(std::forward<Args>(args)...);
  }
};
}// namespace umpalumpa::utils