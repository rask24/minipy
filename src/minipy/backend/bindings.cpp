#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "array_ops.hpp"

namespace py = pybind11;

PYBIND11_MODULE(array_ops, m) {
  m.doc() = "MiniPy array operations";
  m.def("add", &minipy::ArrayOps::add_cpu, "Add two arrays");
}
