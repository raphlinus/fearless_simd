// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![feature(target_feature_11)]

use fearless_simd::f32x4;
use fearless_simd_macro::simd_dispatch;

// This expands to a module named `fast_inv_sqrt_f32x4` with a free
// function for each level, plus a dispatch function that checks
// CPU level at runtime and calls the best instance.

#[simd_dispatch(levels = "neon, fallback", module = true)]
#[inline]
fn fast_inv_sqrt_f32x4(x: f32x4) -> f32x4 {
    // Example of calling arch-specific intrinsics directly
    #[cfg(fearless_simd_level = "neon")]
    {
        use core::arch::aarch64::*;
        // This unsafe will not be needed when safe intrinsics lands, see
        // https://github.com/rust-lang/libs-team/issues/494
        unsafe {
            let a = x.into();
            let b = vrsqrteq_f32(a);
            let c = vmulq_f32(vrsqrtsq_f32(vmulq_f32(a, b), b), b);
            c.into()
        }
    }
    #[cfg(not(fearless_simd_level = "neon"))]
    {
        use simd::f32_4::{div, splat, sqrt};
        div(splat(1.0), sqrt(x))
    }
}

// Similar to above, but instances are hidden inside the function.
#[simd_dispatch(levels = "neon, fallback")]
pub fn foo(x: f32) -> f32 {
    // This lint is super annoying in the presence of cfg's.
    #[allow(unused)]
    use simd::f32s::{add, div, splat, sqrt};
    let a = splat(x);
    let b = add(a, a);

    // Call specialized instance directly, without dispatch.
    // Note the size is fixed now.
    #[cfg(fearless_simd_level = "neon")]
    let c = fast_inv_sqrt_f32x4::neon(b);

    #[cfg(not(fearless_simd_level = "neon"))]
    let c = div(splat(1.0), sqrt(b));

    c.to_array()[0]
}

fn main() {
    let a = foo(42.0);
    println!("a = {a}");
}
