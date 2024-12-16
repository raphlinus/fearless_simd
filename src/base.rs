// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

pub trait Simd {
    const LEN: usize;

    type Element;

    type Mask: Mask;

    fn to_mask(self) -> Self::Mask;

    fn from_mask(value: Self::Mask) -> Self;
}

pub trait Mask {
    const LEN: usize;

    type Element;
}

macro_rules! impl_basetype {
    ($simd:ident, $element:ty, $n:expr) => {
        // TODO: consider Debug, but our f16 doesn't
        #[derive(Clone, Copy, Default)]
        #[repr(transparent)]
        pub struct $simd(pub [$element; $n]);

        impl $simd {
            #[inline]
            pub const fn len(&self) -> usize {
                $n
            }

            #[inline]
            pub const fn as_array(&self) -> &[$element; $n] {
                &self.0
            }

            #[inline]
            pub fn as_mut_array(&mut self) -> &mut [$element; $n] {
                &mut self.0
            }

            #[inline]
            pub const unsafe fn load(ptr: *const [$element; $n]) -> Self {
                // This is adapted from std::simd but all this is almost certainly
                // not needed, because the alignment matches.
                unsafe {
                    let mut tmp = core::mem::MaybeUninit::<Self>::uninit();
                    core::ptr::copy_nonoverlapping(ptr, tmp.as_mut_ptr().cast(), 1);
                    tmp.assume_init()
                }
            }

            #[inline]
            pub unsafe fn store(self, ptr: *mut [$element; $n]) {
                // Same comment as for `load`
                unsafe {
                    let tmp = self;
                    core::ptr::copy_nonoverlapping(tmp.as_array(), ptr, 1);
                }
            }

            #[inline]
            pub const fn from_array(array: [$element; $n]) -> Self {
                Self(array)
            }

            #[inline]
            pub fn to_array(self) -> [$element; $n] {
                self.0
            }

            #[must_use]
            #[inline]
            pub fn from_slice(slice: &[$element]) -> Self {
                assert!(
                    slice.len() >= $n,
                    "slice length must be at least the number of elements"
                );
                unsafe { Self::load(slice.as_ptr().cast()) }
            }
        }

        impl From<[$element; $n]> for $simd {
            #[inline]
            fn from(array: [$element; $n]) -> Self {
                Self(array)
            }
        }

        impl From<$simd> for [$element; $n] {
            #[inline]
            fn from(vector: $simd) -> Self {
                vector.0
            }
        }

        // TODO: TryFrom impls, following std::simd
    };
}

macro_rules! impl_simd {
    ($simd:ident, $element:ty, $n:expr, $mask:ty) => {
        impl_basetype!($simd, $element, $n);
        impl $crate::Simd for $simd {
            const LEN: usize = $n;

            type Element = $element;

            type Mask = $mask;

            /// A bitcast into the mask type, which must be the same size.
            fn to_mask(self) -> <Self as $crate::Simd>::Mask {
                unsafe { core::mem::transmute(self) }
            }

            /// A bitcast from the mask type, which must be the same size.
            fn from_mask(mask: <Self as $crate::Simd>::Mask) -> Self {
                unsafe { core::mem::transmute(mask) }
            }
        }
    };
}

macro_rules! impl_mask {
    ($simd:ident, $element:ty, $n:expr) => {
        impl_basetype!($simd, $element, $n);
        impl $crate::Mask for $simd {
            const LEN: usize = $n;

            type Element = $element;
        }
    };
}

impl_simd!(f32x4, f32, 4, mask32x4);
impl_simd!(u32x4, u32, 4, mask32x4);
impl_simd!(u16x4, u16, 4, mask16x4);
impl_simd!(u16x8, u16, 8, mask16x8);
impl_mask!(mask16x4, i16, 4);
impl_mask!(mask16x8, i16, 8);
impl_mask!(mask32x4, i32, 4);

#[cfg(target_arch = "aarch64")]
mod f16;
pub use f16::*;
