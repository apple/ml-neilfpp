//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2020 Apple Inc. All Rights Reserved.
//
#include "SampleRenderer.h"
#include <iostream>
#include <memory>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include "torch/extension.h"

using std::vector;
using std::cout;
using std::endl;
using std::unique_ptr;

namespace py = pybind11;

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


class PyRT {
public:
    PyRT(const py::array_t<float>& vertices, const py::array_t<uint32_t>& triangles, const int deviceID) {
        py::buffer_info vertices_info = vertices.request();
        float *p_vertices = reinterpret_cast<float*>(vertices_info.ptr);
        py::buffer_info triangles_info = triangles.request();
        uint32_t *p_triangles = reinterpret_cast<uint32_t*>(triangles_info.ptr);

        uint32_t nv = vertices_info.shape[0];
        uint32_t nt = triangles_info.shape[0];
        
        vector<TriangleMeshBuf> meshes;
        meshes.emplace_back(TriangleMeshBuf{p_vertices, nv, p_triangles, nt});
        renderer = unique_ptr<SampleRenderer>(new SampleRenderer(meshes, deviceID));
    };
    py::array_t<float> trace(const py::array_t<float>& origin, const py::array_t<float>& dir) {
        py::buffer_info origin_info = origin.request();
        float *p_origin = reinterpret_cast<float*>(origin_info.ptr);
        py::buffer_info dir_info = dir.request();
        float *p_dir = reinterpret_cast<float*>(dir_info.ptr);

        uint32_t nr = origin_info.shape[0];

        auto result = py::array_t<float>(nr*6);
        py::buffer_info result_info = result.request();
        float *p_result = reinterpret_cast<float*>(result_info.ptr);
        
        if (nr) {
            renderer->resize(vec2i(nr,1));
            renderer->setRaySBT(RayBuf{p_origin, p_dir, nr});
            renderer->render();
            renderer->downloadPixels(p_result);
        }

        return result.reshape({int(nr),6});
    };
    torch::Tensor trace_torch(const torch::Tensor& origin, const torch::Tensor& dir) {
        CHECK_INPUT(origin);
        CHECK_INPUT(dir);

        uint32_t nr = origin.size(0);
        auto output = torch::zeros({nr,6}, torch::TensorOptions().dtype(torch::kFloat32)).cuda();
        
        if (nr) {
            renderer->resize_torch(vec2i(nr,1), output.data_ptr<float>());
            renderer->setRaySBT_torch(origin.data_ptr<float>(), dir.data_ptr<float>());
            renderer->render();
        }

        return output;
    };

    unique_ptr<SampleRenderer> renderer;
};

PYBIND11_MODULE(pyrt, m) {
    py::class_<PyRT>(m, "PyRT")
        .def(py::init<const py::array_t<float>&, const py::array_t<uint32_t>&, const int>())
        .def("trace", &PyRT::trace)
        .def("trace_torch", &PyRT::trace_torch);
}
