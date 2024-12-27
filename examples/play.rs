// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::{simd_dispatch, Level, Simd, WithSimd};

// The WithSimd idea is adapted from pulp but is clunky; we
// will probably prefer the `simd_dispatch!` macro.
struct Foo;

impl WithSimd for Foo {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let a = simd.splat_f32x4(42.0);
        let b = a + a;
        b[0]
    }
}

#[inline(always)]
fn foo_inner<S: Simd>(simd: S, x: f32) -> f32 {
    simd.splat_f32x4(x).sqrt()[0]
}

simd_dispatch!(foo(level, x: f32) -> f32 = foo_inner);

fn do_something_on_neon(_level: Level) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = _level.as_neon() {
        return neon.vectorize(
            #[inline(always)]
            || {
                let v = neon.vdupq_n_f32(42.0);
                neon.vgetq_lane_f32::<0>(v)
            },
        );
    }
    0.0
}

fn main() {
    let level = Level::new();
    let x = level.dispatch(Foo);
    let y = foo(level, 42.0);
    let z = do_something_on_neon(level);

    println!("level = {level:?}, x = {x}, y = {y}, z = {z}");
}
