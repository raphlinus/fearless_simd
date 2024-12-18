// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![feature(target_feature_11)]

use std::arch::is_aarch64_feature_detected;

use fearless_simd::f32x4;

#[target_feature(enable = "neon")]
fn simd_inner() {
    use fearless_simd::neon as simd;
    use simd::f32_4;
    let a = f32_4::splat(42.0);
    let b = f32_4::add(a, a);
    println!("{:?}", b.to_array());
    let c = f32x4::from_array([0.0, 1.0, 42.0, 3.14]);
    let d = f32_4::simd_eq(a, c);
    println!("{:?}", d.to_array());
    println!("{:?}", f32_4::round_cast_u32(c).to_array());
}

#[target_feature(enable = "fp16")]
fn fp16_inner() {
    use fearless_simd::neon as simd;
    use simd::f16_8;
    let a = f16_8::splat_f32_const(42.0);
    let b = f16_8::add(a, a);
    let c = f16_8::cvt_f32(b);
    println!("{:?}", c.to_array());
}

fn main() {
    // Safety: only call the simd function when the feature
    // is detected. We'll want a macro for this.
    if is_aarch64_feature_detected!("neon") {
        unsafe {
            simd_inner();
        }
    }
    if is_aarch64_feature_detected!("fp16") {
        unsafe {
            fp16_inner();
        }
    }
    #[cfg(all())]
    println!("true is true");
}
