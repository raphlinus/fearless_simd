// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

pub mod f16_8;
pub mod f32_4;
pub mod mask16_8;
pub mod mask32_4;
pub mod u32_4;

pub use f16_8 as f16s;
pub use f32_4 as f32s;
pub use mask32_4 as mask32s;
pub use mask16_8 as mask16s;
pub use u32_4 as u32s;
