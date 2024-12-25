// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![feature(target_feature_11)]

use fearless_simd::token::{f32x4, Aarch64, IntoSimdVec, Select, Simd, WithSimd};

#[allow(unused)]
struct Foo;

#[inline(never)]
fn sqrt<S: Simd>(x: f32x4<S>) -> f32x4<S> {
    if let Some(n) = x.simd.arch().as_neon() {
        //n.call(|x| n.vsqrtq_f32(x.into()).into_simd(x.simd), x)
        n.vsqrtq_f32(x.into()).into_simd(x.simd)
    } else {
        let result = x.val.map(|x| x.sqrt());
        result.into_simd(x.simd)
    }
}

impl WithSimd for Foo {
    type Output = f32;

    fn with_simd<S: fearless_simd::token::Simd>(self, simd: S) -> Self::Output {
        let a = simd.splat_f32x4(42.0);
        let b = sqrt(a);
        b.val[0]
    }
}

// Free functions polymorphic in Simd are ok.
#[allow(unused)]
#[inline(always)]
fn quux<S: Simd>(simd: S) -> f32 {
    let a = simd.splat_f32x4(42.0);
    let b = sqrt(a);
    b.val[0]
}

struct ToSrgb([f32; 4]);

impl WithSimd for ToSrgb {
    type Output = [f32; 4];

    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let v: f32x4<S> = self.0.into_simd(simd);
        let vabs = v.abs();
        // TODO: impl sub
        let x = vabs + -5.35862651e-04;
        let x2 = x * x;
        let even1 = x.mul_add(-9.12795913e-01, -2.88143143e-02);
        let even2 = x2.mul_add(-7.29192910e-01, even1);
        let odd1 = x.mul_add(1.06133172e+00, 1.40194533e+00);
        let odd2 = x2.mul_add(2.07758287e-01, odd1);
        let poly = odd2.mul_add(sqrt(x), even2);
        let lin = vabs * 12.92;
        let z = vabs.simd_gt(0.0031308).select(poly, lin);
        z.val
    }
}

#[inline(never)]
fn bar(arch: Aarch64, rgba: [f32; 4]) -> [f32; 4] {
    arch.dispatch(ToSrgb(rgba))
}

fn main() {
    let aarch64 = Aarch64::new();
    let x = bar(aarch64, [0.1, -0.2, 0.001, 0.4]);
    println!("x = {x:?}");
}
