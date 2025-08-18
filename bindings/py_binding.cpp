#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h> // 必须在 pybind11 前面

#include "../include/data_mover.hpp"

namespace py = pybind11;

PYBIND11_MODULE(datamover, m) {
  m.doc() = "Python bindings for DataMover";

  // 先暴露 Mode 枚举
  py::enum_<Mode>(m, "Mode")
      .value("CPU", Mode::CPU)
      .value("GPU", Mode::GPU)
      .export_values();

  // 暴露 DataMover 类
  py::class_<DataMover>(m, "DataMover")
      .def(py::init<>())
      .def_static("init", &DataMover::init, py::arg("mode"))
      .def_static("register_buffer", &DataMover::register_buffer,
                  py::arg("size"), py::arg("name"))
      .def_static("load_file_to_buffer_sync",
                  &DataMover::load_file_to_buffer_sync, py::arg("name"),
                  py::arg("files"))
      .def_static("load_buffer_to_gpu_sync",
                  &DataMover::load_buffer_to_gpu_sync, py::arg("name"),
                  py::arg("tensor"))
      .def_static("print_shm_info", &DataMover::print_shm_info,
                  py::arg("name"));
}
