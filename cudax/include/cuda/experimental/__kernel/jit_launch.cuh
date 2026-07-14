//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___KERNEL_JIT_LAUNCH
#define _CUDAX___KERNEL_JIT_LAUNCH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__driver/driver_api.h>
#  include <cuda/devices>
#  include <cuda/experimental/__kernel/jit_dispatch.cuh>
#  include <cuda/experimental/launch.cuh>
#  include <cuda/experimental/library.cuh>
#  include <nvrtc.h>

#  include <cstddef>
#  include <stdexcept>
#  include <string>
#  include <vector>

namespace cuda::experimental::prototype
{
namespace __detail
{
inline constexpr char __jit_entry_name[] = "cccl_jit_dispatch_entry";

inline void __check_nvrtc(nvrtcResult __status, const char* __message)
{
  if (__status != NVRTC_SUCCESS)
  {
    throw ::std::runtime_error(::std::string{__message} + ": " + nvrtcGetErrorString(__status));
  }
}

template <class _Dispatcher>
inline ::std::vector<char> __compile_with_nvrtc()
{
  ::std::string __source = "#include \"" + ::std::string{jit_kernel_source<_Dispatcher>::implementation_header} + "\"\n";
  __source += "extern \"C\" __global__ void ";
  __source += __jit_entry_name;
  __source += "(";
  __source += jit_kernel_source<_Dispatcher>::dispatcher_type_name;
  __source += " __dispatcher)\n{\n  ";
  __source += "(void) ";
  __source += jit_kernel_source<_Dispatcher>::dispatcher_type_name;
  __source += "::dispatch(__dispatcher);\n}\n";

  nvrtcProgram __program{};
  __check_nvrtc(nvrtcCreateProgram(&__program, __source.c_str(), "cccl_jit_dispatch.cu", 0, nullptr, nullptr),
                "nvrtcCreateProgram failed");

  const ::cuda::device_ref __device{0};
  const int __cc_major = ::cuda::device_attributes::compute_capability_major(__device);
  const int __cc_minor = ::cuda::device_attributes::compute_capability_minor(__device);

  ::std::vector<::std::string> __options_storage;
  __options_storage.push_back("-std=c++17");
  __options_storage.push_back("--gpu-architecture=sm_" + ::std::to_string(__cc_major) + ::std::to_string(__cc_minor));
  for (::std::size_t __i = 0; __i != jit_kernel_source<_Dispatcher>::include_path_count; ++__i)
  {
    __options_storage.push_back("--include-path=" + ::std::string{jit_kernel_source<_Dispatcher>::include_paths[__i]});
  }

  ::std::vector<const char*> __options;
  __options.reserve(__options_storage.size());
  for (const auto& __option : __options_storage)
  {
    __options.push_back(__option.c_str());
  }

  const nvrtcResult __compile_status =
    nvrtcCompileProgram(__program, static_cast<int>(__options.size()), __options.data());

  size_t __log_size = 0;
  __check_nvrtc(nvrtcGetProgramLogSize(__program, &__log_size), "nvrtcGetProgramLogSize failed");
  ::std::string __log;
  if (__log_size > 1)
  {
    __log.resize(__log_size);
    __check_nvrtc(nvrtcGetProgramLog(__program, __log.data()), "nvrtcGetProgramLog failed");
  }

  if (__compile_status != NVRTC_SUCCESS)
  {
    nvrtcDestroyProgram(&__program);
    throw ::std::runtime_error(
      ::std::string{"nvrtcCompileProgram failed: "} + nvrtcGetErrorString(__compile_status) + "\n" + __log);
  }

  size_t __cubin_size = 0;
  __check_nvrtc(nvrtcGetCUBINSize(__program, &__cubin_size), "nvrtcGetCUBINSize failed");

  ::std::vector<char> __cubin(__cubin_size);
  __check_nvrtc(nvrtcGetCUBIN(__program, __cubin.data()), "nvrtcGetCUBIN failed");
  __check_nvrtc(nvrtcDestroyProgram(&__program), "nvrtcDestroyProgram failed");
  return __cubin;
}
} // namespace __detail

template <class _Stream, class _LaunchConfig, class _Dispatcher>
void jit_launch(_Stream& __stream, _LaunchConfig __launch_config, _Dispatcher __dispatcher)
{
  const auto __cubin = __detail::__compile_with_nvrtc<_Dispatcher>();

  auto __library = ::cuda::experimental::library::from_native_handle(
    ::cuda::__driver::__libraryLoadData(__cubin.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  auto __kernel = __library.kernel<void(_Dispatcher)>(__detail::__jit_entry_name);

  ::cuda::experimental::launch(__stream, __launch_config, __kernel, __dispatcher);
}
} // namespace cuda::experimental::prototype

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDAX___KERNEL_JIT_LAUNCH
