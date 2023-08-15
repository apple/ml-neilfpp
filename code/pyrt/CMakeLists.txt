#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

# originally taken (and adapted) from https://github.com/ingowald/optix7course, 
# under following license

# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(PyRT VERSION 0.1)

cmake_minimum_required(VERSION 2.8)
if (NOT WIN32)
# visual studio doesn't like these (not need them):
# set (CMAKE_CXX_STANDARD 17)
# set (CUDA_PROPAGATE_HOST_FLAGS ON)
endif()

# ------------------------------------------------------------------
# first, include gdt project to do some general configuration stuff
# (build modes, glut, optix, etc)
# ------------------------------------------------------------------
set(gdt_dir ${PROJECT_SOURCE_DIR}/common/gdt/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")
include(${gdt_dir}/cmake/configure_build_type.cmake)
include(${gdt_dir}/cmake/configure_optix.cmake)

#set(glfw_dir ${PROJECT_SOURCE_DIR}/submodules/glfw/)
#include(${gdt_dir}/cmake/configure_glfw.cmake)

mark_as_advanced(CUDA_SDK_ROOT_DIR)

# ------------------------------------------------------------------
# import gdt submodule
# ------------------------------------------------------------------
include_directories(${gdt_dir})
add_subdirectory(${gdt_dir} EXCLUDE_FROM_ALL)

# # ------------------------------------------------------------------
# # build glfw
# # ------------------------------------------------------------------
# set(OpenGL_GL_PREFERENCE LEGACY)
# if (WIN32)
# #  set(glfw_dir ${PROJECT_SOURCE_DIR}/submodules/glfw/)
#   set(glfw_dir ${PROJECT_SOURCE_DIR}/common/3rdParty/glfw/)
#   include_directories(${glfw_dir}/include)
#   add_subdirectory(${glfw_dir} EXCLUDE_FROM_ALL)
# else()
#   find_package(glfw3 REQUIRED)
# endif()
include_directories(common)
# add_subdirectory(common/glfWindow EXCLUDE_FROM_ALL)


# ------------------------------------------------------------------
# and final build rules for the project
# ------------------------------------------------------------------

set(optix_LIBRARY "")

include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

# add_executable(optix
#   ${embedded_ptx_code}
#   optix7.h
#   CUDABuffer.h
#   LaunchParams.h
#   SampleRenderer.h
#   SampleRenderer.cpp
#   optix.cpp
#   )
# target_link_libraries(optix
#   gdt
#   ${optix_LIBRARY}
#   ${CUDA_LIBRARIES}
#   ${CUDA_CUDA_LIBRARY}
#   )

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# message(STATUS "${TORCH_LIBRARIES}")
pybind11_add_module(pyrt
  ${embedded_ptx_code}
  optix7.h
  CUDABuffer.h
  LaunchParams.h
  SampleRenderer.h
  SampleRenderer.cpp
  pyrt.cpp
  )
target_link_libraries(pyrt
  gdt
  ${optix_LIBRARY}
  ${TORCH_LIBRARIES}
  $ENV{Torch_DIR}/../../../lib/libtorch_python.so
  ${CUDA_LIBRARIES}
  # ${CUDA_CUDA_LIBRARY}
  # ${CUDA_cublas_LIBRARY}
  )
