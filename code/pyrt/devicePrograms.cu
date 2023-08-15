//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2020 Apple Inc. All Rights Reserved.
//

// originally taken (and adapted) from https://github.com/ingowald/optix7course, 
// under following license

// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>

#include "LaunchParams.h"
  
/*! launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
extern "C" __constant__ LaunchParams optixLaunchParams;

// for this simple example, we have a single ray type
enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };

static __forceinline__ __device__
void *unpackPointer( uint32_t i0, uint32_t i1 )
{
  const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
  void*           ptr = reinterpret_cast<void*>( uptr ); 
  return ptr;
}

static __forceinline__ __device__
void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
{
  const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
  i0 = uptr >> 32;
  i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T *getPRD()
{ 
  const uint32_t u0 = optixGetPayload_0();
  const uint32_t u1 = optixGetPayload_1();
  return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
}

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__radiance()
{
  const TriangleMeshSBTDataBuf &sbtData
    = *(const TriangleMeshSBTDataBuf*)optixGetSbtDataPointer();

  // compute normal:
  const int   primID = optixGetPrimitiveIndex();
  const vec3i index(sbtData.index[3*primID], sbtData.index[3*primID+1], sbtData.index[3*primID+2]);
  const vec3f A(sbtData.vertex[3*index.x], sbtData.vertex[3*index.x+1], sbtData.vertex[3*index.x+2]);
  const vec3f B(sbtData.vertex[3*index.y], sbtData.vertex[3*index.y+1], sbtData.vertex[3*index.y+2]);
  const vec3f C(sbtData.vertex[3*index.z], sbtData.vertex[3*index.z+1], sbtData.vertex[3*index.z+2]);
  const vec3f rayDir = optixGetWorldRayDirection();
  const float2 &uv = optixGetTriangleBarycentrics();

  float w = 1 - uv.x - uv.y;
  const vec3f pos = A*uv.x + B*uv.y + C*w;
  const vec3f Ng = normalize(cross(B-A,C-A));
  // const float cosDN  = 0.2f + .8f*fabsf(dot(rayDir,Ng));

  vec6f &prd = *(vec6f*)getPRD<vec6f>();
  // prd = cosDN * sbtData.color;
  // prd = gdt::randomColor(primID);
  prd[0] = pos.x;
  prd[1] = pos.y;
  prd[2] = pos.z;
  prd[3] = Ng.x;
  prd[4] = Ng.y;
  prd[5] = Ng.z;
}

extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */ }



//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance()
{
  vec6f &prd = *(vec6f*)getPRD<vec6f>();
  // set to constant white as background color
  prd = vec6f(9999.f);
}



//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame()
{
  // compute a test pattern based on pixel ID
  const int ix = optixGetLaunchIndex().x;
  const int iy = optixGetLaunchIndex().y;

  // const auto &camera = optixLaunchParams.camera;

  // our per-ray data for this example. what we initialize it to
  // won't matter, since this value will be overwritten by either
  // the miss or hit program, anyway
  vec6f pixelColorPRD(0.f);

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer( &pixelColorPRD, u0, u1 );

  // normalized screen plane position, in [0,1]^2
  // const vec2f screen(vec2f(ix+.5f,iy+.5f)
  //                     / vec2f(optixLaunchParams.frame.size));
  
  // generate ray direction
  // vec3f rayDir = normalize(camera.direction
  //                           + (screen.x - 0.5f) * camera.horizontal
  //                           + (screen.y - 0.5f) * camera.vertical);

  const RaySBTDataBuf &raySBTData = *(const RaySBTDataBuf*)optixGetSbtDataPointer();
  const size_t offset_idx = iy * optixLaunchParams.frame.size.x + ix;

  vec3f ray_origin(raySBTData.origin[3*offset_idx], 
                   raySBTData.origin[3*offset_idx+1], 
                   raySBTData.origin[3*offset_idx+2]);
  vec3f ray_dir(raySBTData.dir[3*offset_idx], 
                raySBTData.dir[3*offset_idx+1], 
                raySBTData.dir[3*offset_idx+2]);

  optixTrace(optixLaunchParams.traversable,
              ray_origin,
              ray_dir,
              1e-3f,    // tmin
              1e20f,  // tmax
              0.0f,   // rayTime
              OptixVisibilityMask( 255 ),
              OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
              SURFACE_RAY_TYPE,             // SBT offset
              RAY_TYPE_COUNT,               // SBT stride
              SURFACE_RAY_TYPE,             // missSBTIndex 
              u0, u1 );

  // const int r = int(255.99f*pixelColorPRD.x);
  // const int g = int(255.99f*pixelColorPRD.y);
  // const int b = int(255.99f*pixelColorPRD.z);

  // // convert to 32-bit rgba value (we explicitly set alpha to 0xff
  // // to make stb_image_write happy ...
  // const uint32_t rgba = 0xff000000
  //   | (r<<0) | (g<<8) | (b<<16);

  // and write to frame buffer ...
  // const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;
  optixLaunchParams.frame.colorBuffer[6*offset_idx] = pixelColorPRD[0];
  optixLaunchParams.frame.colorBuffer[6*offset_idx+1] = pixelColorPRD[1];
  optixLaunchParams.frame.colorBuffer[6*offset_idx+2] = pixelColorPRD[2];
  optixLaunchParams.frame.colorBuffer[6*offset_idx+3] = pixelColorPRD[3];
  optixLaunchParams.frame.colorBuffer[6*offset_idx+4] = pixelColorPRD[4];
  optixLaunchParams.frame.colorBuffer[6*offset_idx+5] = pixelColorPRD[5];
}
