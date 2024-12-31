// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Access to architecture-specific intrinsics.

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
