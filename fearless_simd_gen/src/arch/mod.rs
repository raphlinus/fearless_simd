// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

pub(crate) mod fallback;
pub(crate) mod neon;
pub(crate) mod wasm;

use proc_macro2::TokenStream;

use crate::types::VecType;

pub(crate) trait Arch {
    fn arch_ty(&self, ty: &VecType) -> TokenStream;
    fn expr(&self, op: &str, ty: &VecType, args: &[TokenStream]) -> TokenStream;
}
