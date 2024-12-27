// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::Level;

use seal::Seal;

/// A trait for SIMD capability tokens.
///
/// The tokens are zero-sized and represent CPU features, which can be
/// detected at runtime. Methods on this trait implement a common (but
/// not exhaustive) set of SIMD operations, which use the efficient SIMD
/// intrinsics.
pub trait Simd: Seal + Sized + Clone + Copy + Send + Sync + 'static {
    fn level(self) -> Level;

    /// Call function with CPU features enabled.
    ///
    /// For performance, the provided function should be `#[inline(always)]`.
    fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R;

    fn splat_f32x4(self, val: f32) -> f32x4<Self>;
    fn add_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> f32x4<Self>;
    fn sub_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> f32x4<Self>;
    fn mul_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> f32x4<Self>;
    fn div_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> f32x4<Self>;
    fn mul_add_f32x4(self, a: f32x4<Self>, b: f32x4<Self>, c: f32x4<Self>) -> f32x4<Self>;
    fn abs_f32x4(self, a: f32x4<Self>) -> f32x4<Self>;
    fn sqrt_f32x4(self, a: f32x4<Self>) -> f32x4<Self>;
    fn copysign_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> f32x4<Self>;
    fn simd_gt_f32x4(self, a: f32x4<Self>, b: f32x4<Self>) -> mask32x4<Self>;
    fn select_f32x4(self, a: mask32x4<Self>, b: f32x4<Self>, c: f32x4<Self>) -> f32x4<Self>;
}

pub trait Select<T> {
    fn select(self, if_true: T, if_false: T) -> T;
}

// Same as pulp
pub trait WithSimd {
    type Output;

    fn with_simd<S: Simd>(self, simd: S) -> Self::Output;
}

impl<R, F: FnOnce(Level) -> R> WithSimd for F {
    type Output = R;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        self(simd.level())
    }
}

pub(crate) mod seal {
    pub trait Seal {}
}

/// Value conversion, adding a SIMD blessing.
///
/// Analogous to [`From`], but takes a SIMD token, which is used to bless
/// the new value. Most such conversions are safe transmutes, but this
/// trait also supports splats, and implementations can use the SIMD token
/// to use an efficient splat intrinsic.
///
/// The [`SimdInto`] trait is also provided for convenience.
pub trait SimdFrom<T, S: Simd> {
    fn simd_from(value: T, simd: S) -> Self;
}

/// Value conversion, adding a SIMD blessing.
///
/// This trait is syntactic sugar for [`SimdFrom`] and exists only to allow
/// `impl SimdInto` syntax in signatures, which would otherwise require
/// cumbersome `where` clauses in terms of `SimdFrom`.
///
/// Avoid implementing this trait directly, prefer implementing [`SimdFrom`].
pub trait SimdInto<T, S> {
    fn simd_into(self, simd: S) -> T;
}

impl<F, T: SimdFrom<F, S>, S: Simd> SimdInto<T, S> for F {
    fn simd_into(self, simd: S) -> T {
        SimdFrom::simd_from(self, simd)
    }
}

impl<T, S: Simd> SimdFrom<T, S> for T {
    fn simd_from(value: T, _simd: S) -> Self {
        value
    }
}

macro_rules! impl_simd_type {
    ($name:ident, $scalar:ty, $n:literal, $align:literal) => {
        #[derive(Clone, Copy)]
        #[repr(C, align($align))]
        pub struct $name<S: Simd> {
            pub val: [$scalar; $n],
            pub simd: S,
        }

        impl<S: Simd> SimdFrom<[$scalar; $n], S> for $name<S> {
            fn simd_from(val: [$scalar; $n], simd: S) -> Self {
                $name { val, simd }
            }
        }

        impl<S: Simd> From<$name<S>> for [$scalar; $n] {
            #[inline(always)]
            fn from(value: $name<S>) -> Self {
                value.val
            }
        }

        impl<S: Simd> std::ops::Deref for $name<S> {
            type Target = [$scalar; $n];
            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                &self.val
            }
        }

        impl<S: Simd> std::ops::DerefMut for $name<S> {
            #[inline(always)]
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.val
            }
        }
    };
}

impl_simd_type!(f32x4, f32, 4, 16);

impl<S: Simd> SimdFrom<f32, S> for f32x4<S> {
    fn simd_from(value: f32, simd: S) -> Self {
        simd.splat_f32x4(value)
    }
}

// TODO: macros to reduce boilerplate
impl<S: Simd> f32x4<S> {
    #[inline(always)]
    pub fn abs(self) -> f32x4<S> {
        self.simd.abs_f32x4(self)
    }

    #[inline(always)]
    pub fn sqrt(self) -> f32x4<S> {
        self.simd.sqrt_f32x4(self)
    }

    #[inline(always)]
    pub fn copysign(self, rhs: impl SimdInto<Self, S>) -> f32x4<S> {
        self.simd.copysign_f32x4(self, rhs.simd_into(self.simd))
    }

    #[inline(always)]
    pub fn mul_add(self, b: impl SimdInto<Self, S>, c: impl SimdInto<Self, S>) -> Self {
        self.simd
            .mul_add_f32x4(self, b.simd_into(self.simd), c.simd_into(self.simd))
    }

    #[inline(always)]
    pub fn simd_gt(self, rhs: impl SimdInto<Self, S>) -> mask32x4<S> {
        self.simd.simd_gt_f32x4(self, rhs.simd_into(self.simd))
    }
}

impl_simd_type!(mask32x4, i32, 4, 16);

impl<S: Simd> Select<f32x4<S>> for mask32x4<S> {
    fn select(self, if_true: f32x4<S>, if_false: f32x4<S>) -> f32x4<S> {
        self.simd.select_f32x4(self, if_true, if_false)
    }
}
