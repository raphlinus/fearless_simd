// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Example of fast feature detection.

#![feature(target_feature_11)]

use fearless_simd_macro::simd_dispatch;

// Run this to see the generated asm (using cargo-show-asm):
//
// `cargo asm -p fearless_simd --example fast_detect foo 0`
#[simd_dispatch(levels = "avx2, fallback", detect = "fast")]
#[inline(never)]
fn foo(inp: [f32; 4]) -> [f32; 4] {
    use simd::f32_4::*;
    let x = inp.into();
    mul_add(x, x, splat(1.0)).into()
}

fn main() {
    // Safety: doing this before launching any threads that might use SIMD.
    unsafe {
        fearless_simd::init_simd_detect();
    }
    let inp = [0.0, 1.0, 2.0, 3.0];
    println!("{:?}", foo(inp));
}
