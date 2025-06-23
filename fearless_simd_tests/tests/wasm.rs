// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(target_arch = "wasm32")]
use fearless_simd::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

/// `test_wasm_simd_parity` enforces that the fallback level and +simd128 levels output the same
/// results.
macro_rules! test_wasm_simd_parity {
    (
        fn $test_name:ident() {
            |$s:ident| -> $ret_type:ty $body:block
        }
    ) => {
        #[cfg(target_arch = "wasm32")]
        #[wasm_bindgen_test]
        fn $test_name() {
            fn test_impl<S: Simd>($s: S) -> $ret_type $body

            simd_dispatch!(dispatch(level) -> $ret_type = test_impl);

            let fallback_result = dispatch(Level::Fallback(Fallback::new()));
            let wasm_result = dispatch(Level::WasmSimd128(wasm32::WasmSimd128::new_unchecked()));

            assert_eq!(fallback_result, wasm_result,
                concat!(stringify!($test_name), ": Expected fallback and WASM SIMD results to match."));
        }
    };
}

test_wasm_simd_parity! {
    fn add_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[1.0, 2.0, 3.0, 4.0]);
            let b = f32x4::from_slice(s, &[5.0, 4.0, 3.0, 2.0]);
            (a + b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn sub_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[1.0, 2.0, 3.0, 4.0]);
            let b = f32x4::from_slice(s, &[5.0, 4.0, 3.0, 2.0]);
            (a - b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn mul_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[1.0, 2.0, 3.0, 4.0]);
            let b = f32x4::from_slice(s, &[5.0, 4.0, 3.0, 2.0]);
            (a - b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn mul_u8x16() {
        |s| -> [u8; 16] {
            let a = u8x16::from_slice(s, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            let b = u8x16::from_slice(s, &[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
            (a * b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn mul_i8x16() {
        |s| -> [i8; 16] {
            let a = i8x16::from_slice(s, &[0, -0, 3, -3, 0, -0, 3, -3, 0, -0, 3, -3, 0, -0, 3, -3]);
            let b = i8x16::from_slice(s, &[0, 0, 0, 0, -0, -0, -0, -0, 3, 3, 3, 3, -3, -3, -3, -3]);
            (a * b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn splat_f32x4() {
        |s| -> [f32; 4] {
            f32x4::splat(s, 1.0).into()
        }
    }
}

test_wasm_simd_parity! {
    fn abs_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[-1.0, 0., 1.0, 2.3]);
            f32x4::abs(a).into()
        }
    }
}

test_wasm_simd_parity! {
    fn neg_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[-1.0, 0.0, 1.0, 2.3]);
            f32x4::neg(a).into()
        }
    }
}

test_wasm_simd_parity! {
    fn sqrt_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[4.0, 0.0, 1.0, 2.0]);
            f32x4::sqrt(a).into()
        }
    }
}

test_wasm_simd_parity! {
    fn div_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[4.0, 2.0, 1.0, 0.0]);
            let b = f32x4::from_slice(s, &[4., 1.0, 3.0, 0.1]);
            (a / b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn simd_eq_f32x4() {
        |s| -> [i32; 4] {
            let a = f32x4::from_slice(s, &[4.0, 2.0, 1.0, 0.0]);
            let b = f32x4::from_slice(s, &[4.0, 3.1, 1.0, 0.0]);
            a.simd_eq(b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn simd_lt_f32x4() {
        |s| -> [i32; 4] {
            let a = f32x4::from_slice(s, &[4.0, 3.0, 2.0, 1.0]);
            let b = f32x4::from_slice(s, &[1.0, 2.0, 2.0, 4.0]);
            a.simd_lt(b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn simd_le_f32x4() {
        |s| -> [i32; 4] {
            let a = f32x4::from_slice(s, &[4.0, 3.0, 2.0, 1.0]);
            let b = f32x4::from_slice(s, &[1.0, 2.0, 2.0, 4.0]);
            a.simd_le(b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn simd_ge_f32x4() {
        |s| -> [i32; 4] {
            let a = f32x4::from_slice(s, &[4.0, 3.0, 2.0, 1.0]);
            let b = f32x4::from_slice(s, &[1.0, 2.0, 2.0, 4.0]);
            a.simd_ge(b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn simd_gt_f32x4() {
        |s| -> [i32; 4] {
            let a = f32x4::from_slice(s, &[4.0, 3.0, 2.0, 1.0]);
            let b = f32x4::from_slice(s, &[1.0, 2.0, 2.0, 4.0]);
            a.simd_gt(b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn madd_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[2.0, -3.0, 0.0, 0.5]);
            let b = f32x4::from_slice(s, &[5.0, 4.0, 100.0, 8.0]);
            let c = f32x4::from_slice(s, &[1.0, -2.0, 7.0, 3.0]);
            a.madd(b, c).into()
        }
    }
}

test_wasm_simd_parity! {
    fn max_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[2.0, -3.0, 0.0, 0.5]);
            let b = f32x4::from_slice(s, &[1.0, -2.0, 7.0, 3.0]);
            a.max(b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn min_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[2.0, -3.0, 0.0, 0.5]);
            let b = f32x4::from_slice(s, &[1.0, -2.0, 7.0, 3.0]);
            a.min(b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn max_precise_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[2.0, -3.0, 0.0, 0.5]);
            let b = f32x4::from_slice(s, &[1.0, -2.0, 7.0, 3.0]);
            a.max_precise(b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn min_precise_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[2.0, -3.0, 0.0, 0.5]);
            let b = f32x4::from_slice(s, &[1.0, -2.0, 7.0, 3.0]);
            a.min_precise(b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn floor_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[2.0, -3.2, 0.0, 0.5]);
            a.floor().into()
        }
    }
}

test_wasm_simd_parity! {
    fn combine_f32x4() {
        |s| -> [f32; 8] {
            let a = f32x4::from_slice(s, &[1.0, 2.0, 3.0, 4.0]);
            let b = f32x4::from_slice(s, &[5.0, 6.0, 7.0, 8.0]);
            a.combine(b).into()
        }
    }
}
