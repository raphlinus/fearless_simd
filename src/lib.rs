// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A helper library to make SIMD more friendly.

// TODO: remove when target_feature is stabilized (hopefully soon)
#![feature(target_feature_11)]

#![allow(non_camel_case_types)]

mod base;
mod macros;

pub use base::*;

#[cfg(target_arch = "aarch64")]
mod neon;
