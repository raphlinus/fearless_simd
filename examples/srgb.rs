// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![feature(target_feature_11)]

use fearless_simd::f32x4;
use fearless_simd_macro::simd_dispatch;

#[simd_dispatch(levels = "neon, avx2, fallback")]
#[inline]
fn to_srgb(inp: [f32; 4]) -> [f32; 4] {
    use simd::f32_4::*;
    let v = f32x4::from_array(inp);
    let vabs = abs(v);
    let x = add(vabs, splat(-5.35862651e-04));
    let x2 = mul(x, x);
    let even1 = mul_add(x, splat(-9.12795913e-01), splat(-2.88143143e-02));
    let even2 = mul_add(x2, splat(-7.29192910e-01), even1);
    let odd1 = mul_add(x, splat(1.06133172e+00), splat(1.40194533e+00));
    let odd2 = mul_add(x2, splat(2.07758287e-01), odd1);
    let poly = mul_add(odd2, sqrt(x), even2);
    let lin = mul(vabs, splat(12.92));
    let z = select(simd_gt(vabs, splat(0.0031308)), poly, lin);
    let z_signed = copysign(z, v);
    let out = copy_lane::<3, 3>(z_signed, v);
    out.to_array()
}

fn main() {
    let rgba = [0.1, -0.2, 0.001, 0.4];
    println!("{:?}", to_srgb(rgba));
}
