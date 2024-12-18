// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fallback implementations of SIMD methods.

pub mod f32_4;

// Discussion question: what's the natural width? Most other implementations
// implementations default to 1, but perhaps we want to express the idea
// of operating on larger chunks of data, and encouraging autovectorization.
pub use f32_4 as f32s;
