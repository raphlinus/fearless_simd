// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Access to intrinsics on x86_64.

mod avx;
mod avx2;
mod fma;
mod sse;
mod sse2;
mod sse3;
mod ssse3;
mod sse4_1;
mod sse4_2;

pub use avx::Avx;
pub use avx2::Avx2;
pub use fma::Fma;
pub use sse::Sse;
pub use sse2::Sse2;
pub use sse3::Sse3;
pub use ssse3::Ssse3;
pub use sse4_1::Sse4_1;
pub use sse4_2::Sse4_2;
