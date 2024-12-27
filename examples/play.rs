// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::{simd_dispatch, Level, Simd, WithSimd};

struct Foo;

impl WithSimd for Foo {
    type Output = f32;

    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let a = simd.splat_f32x4(42.0);
        let b = a + a;
        b[0]
    }
}

fn foo_inner<S: Simd>(simd: S, x: f32) -> f32 {
    simd.splat_f32x4(x).sqrt()[0]
}

simd_dispatch!(foo(level, x: f32) -> f32 = foo_inner);

fn main() {
    let level = Level::new();
    let x = level.dispatch(Foo);
    let y = foo(level, 42.0);

    println!("level = {level:?}, x = {x}, y = {y}");
}
