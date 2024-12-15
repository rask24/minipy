#include "array_ops.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(array_ops, m) {
  m.doc() = "MiniPy array operations";
  m.def("add_cpu", &minipy::ArrayOps::add_cpu, "Add two arrays on CPU",
        py::arg("a"), py::arg("b"));
  m.def("add_gpu", &minipy::ArrayOps::add_gpu, "Add two arrays on GPU",
        py::arg("a"), py::arg("b"));
}
