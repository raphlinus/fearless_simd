// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::arch::aarch64::float32x4_t;

use fearless_simd::{f32x4, simd_dispatch, simd_dispatch_explicit, Level, Neon, Select, Simd, SimdInto};

// This block shows how to use safe wrappers for compile-time enforcement
// of using valid SIMD intrinsics.
#[cfg(feature = "safe_wrappers")]
#[inline(always)]
fn copy_alpha<S: Simd>(a: f32x4<S>, b: f32x4<S>) -> f32x4<S> {
    #[cfg(target_arch = "x86_64")]
    if let Some(avx2) = a.simd.level().as_avx2() {
        return avx2
            .sse4_1
            ._mm_blend_ps::<8>(a.into(), b.into())
            .simd_into(a.simd);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = a.simd.level().as_neon() {
        return neon
            .neon
            .vcopyq_laneq_f32::<3, 3>(a.into(), b.into())
            .simd_into(a.simd);
    }
    let mut result = a;
    result[3] = b[3];
    result
}

// This block lets the example compile without safe wrappers.
#[cfg(not(feature = "safe_wrappers"))]
#[inline(always)]
fn copy_alpha<S: Simd>(a: f32x4<S>, b: f32x4<S>) -> f32x4<S> {
    #[cfg(target_arch = "aarch64")]
    if let Some(_neon) = a.simd.level().as_neon() {
        unsafe {
            return core::arch::aarch64::vcopyq_laneq_f32::<3, 3>(a.into(), b.into())
                .simd_into(a.simd);
        }
    }
    let mut result = a;
    result[3] = b[3];
    result
}

#[target_feature(enable = "neon")]
fn copy_alpha_neon(_neon: Neon, a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let a = unsafe { core::mem::transmute(a) };
    let b = unsafe { core::mem::transmute(b) };
    let c = core::arch::aarch64::vcopyq_laneq_f32::<3, 3>(a, b);
    unsafe { core::mem::transmute(c) }
}

simd_dispatch_explicit!(copy_alpha_exp(a: [f32; 4], b: [f32; 4]) -> [f32; 4] =
    match level {
        neon => copy_alpha_neon,
    }
);

#[inline(always)]
fn to_srgb_impl<S: Simd>(simd: S, rgba: [f32; 4]) -> [f32; 4] {
    let v: f32x4<S> = rgba.simd_into(simd);
    let vabs = v.abs();
    let x = vabs - 5.35862651e-04;
    let x2 = x * x;
    let even1 = x * -9.12795913e-01 + -2.88143143e-02;
    let even2 = x2 * -7.29192910e-01 + even1;
    let odd1 = x * 1.06133172e+00 + 1.40194533e+00;
    let odd2 = x2 * 2.07758287e-01 + odd1;
    let poly = odd2 * x.sqrt() + even2;
    let lin = vabs * 12.92;
    let z = vabs.simd_gt(0.0031308).select(poly, lin);
    let z_signed = z.copysign(v);
    let result = copy_alpha_exp(simd.level(), z_signed.into(), v.into());
    result.into()
}

simd_dispatch!(to_srgb(level, rgba: [f32; 4]) -> [f32; 4] = to_srgb_impl);

fn main() {
    let level = Level::new();
    let rgba = [0.1, -0.2, 0.001, 0.4];
    println!("{:?}", to_srgb(level, rgba));
}
