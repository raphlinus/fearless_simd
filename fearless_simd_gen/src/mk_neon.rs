// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Literal, Span, TokenStream};
use quote::quote;
use syn::Ident;

use crate::arch::neon::split_intrinsic;
use crate::ops::{reinterpret_ty, valid_reinterpret};
use crate::types::ScalarType;
use crate::{
    arch::Arch,
    arch::neon::{Neon, cvt_intrinsic, simple_intrinsic},
    generic::{generic_combine, generic_op, generic_split},
    ops::{OpSig, TyFlavor, ops_for_type},
    types::{SIMD_TYPES, VecType, type_imports},
};

#[derive(Clone, Copy)]
pub enum Level {
    Neon,
    // TODO: Fp16,
}

impl Level {
    fn name(self) -> &'static str {
        match self {
            Level::Neon => "Neon",
        }
    }

    fn token(self) -> TokenStream {
        let ident = Ident::new(self.name(), Span::call_site());
        quote! { #ident }
    }
}

pub fn mk_neon_impl(level: Level) -> TokenStream {
    let imports = type_imports();
    let simd_impl = mk_simd_impl(level);
    let ty_impl = mk_type_impl();

    quote! {
        use core::arch::aarch64::*;

        use crate::{seal::Seal, Level, Simd, SimdFrom, SimdInto};

        #imports

        /// The SIMD token for the "neon" level.
        #[derive(Clone, Copy, Debug)]
        pub struct Neon {
            pub neon: crate::core_arch::aarch64::Neon,
        }

        impl Neon {
            #[inline]
            pub unsafe fn new_unchecked() -> Self {
                Neon {
                    neon: unsafe { crate::core_arch::aarch64::Neon::new_unchecked() },
                }
            }
        }

        impl Seal for Neon {}

        #simd_impl

        #ty_impl
    }
}

fn mk_simd_impl(level: Level) -> TokenStream {
    let level_tok = level.token();
    let mut methods = vec![];
    for vec_ty in SIMD_TYPES {
        let scalar_bits = vec_ty.scalar_bits;
        let ty_name = vec_ty.rust_name();
        let ty = vec_ty.rust();
        for (method, sig) in ops_for_type(vec_ty, true) {
            if (vec_ty.n_bits() > 128 && !matches!(method, "split" | "narrow"))
                || vec_ty.n_bits() > 256
            {
                methods.push(generic_op(method, sig, vec_ty));
                continue;
            }
            let method_name = format!("{method}_{ty_name}");
            let method_ident = Ident::new(&method_name, Span::call_site());
            let ret_ty = sig.ret_ty(vec_ty, TyFlavor::SimdTrait);
            let method = match sig {
                OpSig::Splat => {
                    let scalar = vec_ty.scalar.rust(scalar_bits);
                    let expr = Neon.expr(method, vec_ty, &[quote! { val }]);
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, val: #scalar) -> #ret_ty {
                            unsafe {
                                #expr.simd_into(self)
                            }
                        }
                    }
                }
                OpSig::Shift => {
                    let dup_type = VecType::new(ScalarType::Int, vec_ty.scalar_bits, vec_ty.len);
                    let scalar = dup_type.scalar.rust(scalar_bits);
                    let dup_intrinsic = split_intrinsic("vdup", "n", &dup_type);
                    let shift = if method == "shr" {
                        quote! { -(shift as #scalar) }
                    } else {
                        quote! { shift as #scalar }
                    };
                    let expr = Neon.expr(
                        method,
                        vec_ty,
                        &[quote! { val.into() }, quote! { #dup_intrinsic ( #shift ) }],
                    );
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, val: #ty<Self>, shift: u32) -> #ret_ty {
                            unsafe {
                                #expr.simd_into(self)
                            }
                        }
                    }
                }
                OpSig::Unary => {
                    let args = [quote! { a.into() }];
                    let expr = Neon.expr(method, vec_ty, &args);
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                            unsafe {
                                #expr.simd_into(self)
                            }
                        }
                    }
                }
                OpSig::WidenNarrow(target_ty) => {
                    let ret_ty = sig.ret_ty(&target_ty, TyFlavor::SimdTrait);
                    let vec_scalar_ty = vec_ty.scalar.rust(vec_ty.scalar_bits);
                    let target_scalar_ty = target_ty.scalar.rust(target_ty.scalar_bits);

                    if method == "narrow" {
                        let arch = Neon.arch_ty(vec_ty);

                        let id1 =
                            Ident::new(&format!("vmovn_{}", vec_scalar_ty), Span::call_site());
                        let id2 = Ident::new(
                            &format!("vcombine_{}", target_scalar_ty),
                            Span::call_site(),
                        );

                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                                unsafe {
                                    let converted: #arch = a.into();
                                    let low = #id1(converted.0);
                                    let high = #id1(converted.1);

                                    #id2(low, high).simd_into(self)
                                }
                            }
                        }
                    } else {
                        let arch = Neon.arch_ty(&target_ty);
                        let id1 =
                            Ident::new(&format!("vmovl_{}", vec_scalar_ty), Span::call_site());
                        let id2 =
                            Ident::new(&format!("vget_low_{}", vec_scalar_ty), Span::call_site());
                        let id3 =
                            Ident::new(&format!("vget_high_{}", vec_scalar_ty), Span::call_site());

                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                                unsafe {
                                    let low = #id1(#id2(a.into()));
                                    let high = #id1(#id3(a.into()));

                                    #arch(low, high).simd_into(self)
                                }
                            }
                        }
                    }
                }
                OpSig::Binary => {
                    let args = [quote! { a.into() }, quote! { b.into() }];
                    if method == "copysign" {
                        let shift_amt = Literal::usize_unsuffixed(vec_ty.scalar_bits - 1);
                        let unsigned_ty = VecType::new(
                            crate::types::ScalarType::Unsigned,
                            vec_ty.scalar_bits,
                            vec_ty.len,
                        );
                        let sign_mask =
                            Neon.expr("splat", &unsigned_ty, &[quote! { 1 << #shift_amt }]);
                        let vbsl = simple_intrinsic("vbsl", vec_ty);

                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                                unsafe {
                                    let sign_mask = #sign_mask;
                                    #vbsl(sign_mask, b.into(), a.into()).simd_into(self)
                                }
                            }
                        }
                    } else {
                        let expr = Neon.expr(method, vec_ty, &args);
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                                unsafe {
                                    #expr.simd_into(self)
                                }
                            }
                        }
                    }
                }
                OpSig::Ternary => {
                    let args = [
                        quote! { a.into() },
                        quote! { b.into() },
                        quote! { c.into() },
                    ];

                    let expr = Neon.expr(method, vec_ty, &args);
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>, c: #ty<Self>) -> #ret_ty {
                            unsafe {
                                #expr.simd_into(self)
                            }
                        }
                    }
                }
                OpSig::Compare => {
                    let args = [quote! { a.into() }, quote! { b.into() }];
                    let expr = Neon.expr(method, vec_ty, &args);
                    let opt_q = crate::arch::neon::opt_q(vec_ty);
                    let reinterpret_str =
                        format!("vreinterpret{opt_q}_s{scalar_bits}_u{scalar_bits}");
                    let reinterpret = Ident::new(&reinterpret_str, Span::call_site());
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                            unsafe {
                                #reinterpret(#expr).simd_into(self)
                            }
                        }
                    }
                }
                OpSig::Select => {
                    let opt_q = crate::arch::neon::opt_q(vec_ty);
                    let mask_ty = vec_ty.mask_ty().rust();
                    let reinterpret_str =
                        format!("vreinterpret{opt_q}_u{scalar_bits}_s{scalar_bits}");
                    let reinterpret = Ident::new(&reinterpret_str, Span::call_site());
                    let vbsl = simple_intrinsic("vbsl", vec_ty);
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #mask_ty<Self>, b: #ty<Self>, c: #ty<Self>) -> #ret_ty {
                            unsafe {
                                #vbsl(#reinterpret(a.into()), b.into(), c.into()).simd_into(self)
                            }
                        }
                    }
                }
                OpSig::Combine => generic_combine(vec_ty),
                OpSig::Split => generic_split(vec_ty),
                OpSig::Zip(zip1) => {
                    let neon = if zip1 { "vzip1" } else { "vzip2" };
                    let zip = simple_intrinsic(neon, vec_ty);
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                            let x = a.into();
                            let y = b.into();
                            unsafe {
                                #zip(x, y).simd_into(self)
                            }
                        }
                    }
                }
                OpSig::Cvt(scalar, scalar_bits) => {
                    let to_ty = &VecType::new(scalar, scalar_bits, vec_ty.len);
                    let neon = cvt_intrinsic("vcvtn", to_ty, vec_ty);
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                            unsafe {
                                #neon(a.into()).simd_into(self)
                            }
                        }
                    }
                }
                OpSig::Reinterpret(scalar, scalar_bits) => {
                    if valid_reinterpret(vec_ty, scalar, scalar_bits) {
                        let to_ty = reinterpret_ty(vec_ty, scalar, scalar_bits);
                        let neon = cvt_intrinsic("vreinterpret", &to_ty, vec_ty);

                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                                unsafe {
                                    #neon(a.into()).simd_into(self)
                                }
                            }
                        }
                    } else {
                        quote! {}
                    }
                }
            };
            methods.push(method);
        }
    }
    // Note: the `vectorize` implementation is pretty boilerplate and should probably
    // be factored out for DRY.
    quote! {
        impl Simd for #level_tok {
            type f32s = f32x4<Self>;
            type u8s = u8x16<Self>;
            type i8s = i8x16<Self>;
            type u16s = u16x8<Self>;
            type i16s = i16x8<Self>;
            type u32s = u32x4<Self>;
            type i32s = i32x4<Self>;
            type mask8s = mask8x16<Self>;
            type mask16s = mask16x8<Self>;
            type mask32s = mask32x4<Self>;
            #[inline(always)]
            fn level(self) -> Level {
                Level::#level_tok(self)
            }

            #[inline]
            fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R {
                #[target_feature(enable = "neon")]
                #[inline]
                // unsafe not needed here with tf11, but can be justified
                unsafe fn vectorize_neon<F: FnOnce() -> R, R>(f: F) -> R {
                    f()
                }
                unsafe { vectorize_neon(f) }
            }

            #( #methods )*
        }
    }
}

fn mk_type_impl() -> TokenStream {
    let mut result = vec![];
    for ty in SIMD_TYPES {
        let n_bits = ty.n_bits();
        if !(n_bits == 64 || n_bits == 128 || n_bits == 256 || n_bits == 512) {
            continue;
        }
        let simd = ty.rust();
        let arch = Neon.arch_ty(ty);
        result.push(quote! {
            impl<S: Simd> SimdFrom<#arch, S> for #simd<S> {
                #[inline(always)]
                fn simd_from(arch: #arch, simd: S) -> Self {
                    Self {
                        val: unsafe { core::mem::transmute(arch) },
                        simd
                    }
                }
            }
            impl<S: Simd> From<#simd<S>> for #arch {
                #[inline(always)]
                fn from(value: #simd<S>) -> Self {
                    unsafe { core::mem::transmute(value.val) }
                }
            }
        })
    }
    quote! {
        #( #result )*
    }
}
