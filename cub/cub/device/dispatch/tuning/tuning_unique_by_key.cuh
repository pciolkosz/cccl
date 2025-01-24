/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_unique_by_key.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{

namespace unique_by_key
{

enum class primitive_key
{
  no,
  yes
};
enum class primitive_val
{
  no,
  yes
};
enum class key_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};
enum class val_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};

template <class T>
constexpr primitive_key is_primitive_key()
{
  return Traits<T>::PRIMITIVE ? primitive_key::yes : primitive_key::no;
}

template <class T>
constexpr primitive_val is_primitive_val()
{
  return Traits<T>::PRIMITIVE ? primitive_val::yes : primitive_val::no;
}

template <class KeyT>
constexpr key_size classify_key_size()
{
  return sizeof(KeyT) == 1 ? key_size::_1
       : sizeof(KeyT) == 2 ? key_size::_2
       : sizeof(KeyT) == 4 ? key_size::_4
       : sizeof(KeyT) == 8 ? key_size::_8
       : sizeof(KeyT) == 16
         ? key_size::_16
         : key_size::unknown;
}

template <class ValueT>
constexpr val_size classify_val_size()
{
  return sizeof(ValueT) == 1 ? val_size::_1
       : sizeof(ValueT) == 2 ? val_size::_2
       : sizeof(ValueT) == 4 ? val_size::_4
       : sizeof(ValueT) == 8 ? val_size::_8
       : sizeof(ValueT) == 16
         ? val_size::_16
         : val_size::unknown;
}

template <class KeyT,
          class ValueT,
          primitive_key PrimitiveKey   = is_primitive_key<KeyT>(),
          primitive_val PrimitiveAccum = is_primitive_val<ValueT>(),
          key_size KeySize             = classify_key_size<KeyT>(),
          val_size AccumSize           = classify_val_size<ValueT>()>
struct sm80_tuning;

// 8-bit key
template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<835>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<765>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1155>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_8>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1065>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_1, val_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<248, 1200>;
};

// 16-bit key
template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_1>
{
  static constexpr int threads                       = 320;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1020>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_2>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 22;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<328, 1080>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<535>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_8>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1055>;
};

// 32-bit key
template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1120>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1185>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1115>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_8>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<320, 1115>;
};

// 64-bit key
template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<24, 555>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<324, 1105>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<740, 1105>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_8>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<764, 1155>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_8, val_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<992, 1135>;
};

template <class KeyT,
          class ValueT,
          primitive_key PrimitiveKey   = is_primitive_key<KeyT>(),
          primitive_val PrimitiveAccum = is_primitive_val<ValueT>(),
          key_size KeySize             = classify_key_size<KeyT>(),
          val_size AccumSize           = classify_val_size<ValueT>()>
struct sm90_tuning;

// 8-bit key
template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<550>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_2>
{
  static constexpr int threads                       = 448;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<725>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1130>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_8>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1100>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_1, val_size::_16>
{
  static constexpr int threads                       = 288;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<344, 1165>;
};

// 16-bit key
template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<640>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_2>
{
  static constexpr int threads                       = 288;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<404, 710>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_4>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<525>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_8>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 23;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1200>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_2, val_size::_16>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<424, 1055>;
};

// 32-bit key
template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_1>
{
  static constexpr int threads                       = 448;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<348, 580>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_2>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1060>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_4>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1045>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_8>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1120>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_4, val_size::_16>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1025>;
};

// 64-bit key
template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_1>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1060>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_2>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<964, 1125>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_4>
{
  static constexpr int threads                       = 640;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1070>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_8>
{
  static constexpr int threads                       = 448;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1190>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_8, val_size::_16>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1155>;
};

template <class KeyT, class ValueT>
struct policy_hub
{
  template <int Nominal4bItemsPerThread, int Threads>
  struct DefaultPolicy
  {
    static constexpr int items_per_thread = Nominal4BItemsToItems<KeyT>(Nominal4bItemsPerThread);
    using UniqueByKeyPolicyT =
      AgentUniqueByKeyPolicy<Threads,
                             items_per_thread,
                             BLOCK_LOAD_WARP_TRANSPOSE,
                             LOAD_LDG,
                             BLOCK_SCAN_WARP_SCANS,
                             detail::default_delay_constructor_t<int>>;
  };

  struct Policy350
      : DefaultPolicy<9, 128>
      , ChainedPolicy<350, Policy350, Policy350>
  {};

  // Use values from tuning if a specialization exists, otherwise pick the default
  template <typename Tuning>
  static auto select_agent_policy(int)
    -> AgentUniqueByKeyPolicy<Tuning::threads,
                              Tuning::items,
                              Tuning::load_algorithm,
                              Tuning::load_modifier,
                              BLOCK_SCAN_WARP_SCANS,
                              typename Tuning::delay_constructor>;
  template <typename Tuning>
  static auto select_agent_policy(long) -> typename DefaultPolicy<11, 64>::UniqueByKeyPolicyT;

  struct Policy520
      : DefaultPolicy<11, 64>
      , ChainedPolicy<520, Policy520, Policy350>
  {};

  struct Policy800 : ChainedPolicy<800, Policy800, Policy520>
  {
    using UniqueByKeyPolicyT = decltype(select_agent_policy<sm80_tuning<KeyT, ValueT>>(0));
  };

  struct Policy860
      : DefaultPolicy<11, 64>
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using UniqueByKeyPolicyT = decltype(select_agent_policy<sm90_tuning<KeyT, ValueT>>(0));
  };

  using MaxPolicy = Policy900;
};

} // namespace unique_by_key
} // namespace detail

template <typename KeyInputIteratorT, typename ValueInputIteratorT = unsigned long long int*>
using DeviceUniqueByKeyPolicy CCCL_DEPRECATED_BECAUSE("This class is considered an implementation detail and it will "
                                                      "be removed.") =
  detail::unique_by_key::policy_hub<detail::value_t<KeyInputIteratorT>, detail::value_t<ValueInputIteratorT>>;

CUB_NAMESPACE_END
