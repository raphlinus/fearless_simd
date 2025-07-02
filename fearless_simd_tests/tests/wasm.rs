// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use fearless_simd::*;
use wasm_bindgen_test::*;

/// `test_wasm_simd_parity` enforces that the fallback level and +simd128 levels output the same
/// results.
macro_rules! test_wasm_simd_parity {
    (
        fn $test_name:ident() {
            |$s:ident| -> $ret_type:ty $body:block
        }
    ) => {
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

#[wasm_bindgen_test]
fn max_precise_f32x4_with_nan() {
    fn test_impl<S: Simd>(s: S) -> [f32; 4] {
        let a = f32x4::from_slice(s, &[f32::NAN, -3.0, f32::INFINITY, 0.5]);
        let b = f32x4::from_slice(s, &[1.0, f32::NAN, 7.0, f32::NEG_INFINITY]);
        a.max_precise(b).into()
    }

    simd_dispatch!(test(level) -> [f32; 4] = test_impl);
    let wasm_result = test(Level::WasmSimd128(wasm32::WasmSimd128::new_unchecked()));

    // Note: f32::NAN != f32::NAN hence we transmute to compare the bit pattern. In this case NaN
    // bit pattern is preserved.
    unsafe {
        assert_eq!(
            std::mem::transmute::<[f32; 4], [u32; 4]>(wasm_result),
            std::mem::transmute::<[f32; 4], [u32; 4]>([1., f32::NAN, f32::INFINITY, 0.5]),
            "Wasm did not match expected result."
        );
    }
}

#[wasm_bindgen_test]
fn min_precise_f32x4_with_nan() {
    fn test_impl<S: Simd>(s: S) -> [f32; 4] {
        let a = f32x4::from_slice(s, &[f32::NAN, -3.0, f32::INFINITY, 0.5]);
        let b = f32x4::from_slice(s, &[1.0, f32::NAN, 7.0, f32::NEG_INFINITY]);
        a.min_precise(b).into()
    }

    simd_dispatch!(test(level) -> [f32; 4] = test_impl);
    let wasm_result = test(Level::WasmSimd128(wasm32::WasmSimd128::new_unchecked()));

    // Note: f32::NAN != f32::NAN hence we transmute to compare the bit pattern. In this case NaN
    // bit pattern is preserved.
    unsafe {
        assert_eq!(
            std::mem::transmute::<[f32; 4], [u32; 4]>(wasm_result),
            std::mem::transmute::<[f32; 4], [u32; 4]>([1., f32::NAN, 7., f32::NEG_INFINITY]),
            "Wasm did not match expected result."
        );
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

test_wasm_simd_parity! {
    fn and_i8x16() {
        |s| -> [i8; 16] {
            let a = i8x16::from_slice(s, &[-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]);
            let b = i8x16::from_slice(s, &[85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85]);
            (a & b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn or_i8x16() {
        |s| -> [i8; 16] {
            let a = i8x16::from_slice(s, &[0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8]);
            let b = i8x16::from_slice(s, &[1, 1, 1, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0]);
            (a | b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn xor_i8x16() {
        |s| -> [i8; 16] {
            let a = i8x16::from_slice(s, &[0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, 0, 0, 0, 0]);
            let b = i8x16::from_slice(s, &[-1, -1, 0, 0, 5, 4, 7, 6, -1, 0, -1, 0, -1, 0, -1, 0]);
            (a ^ b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn not_i8x16() {
        |s| -> [i8; 16] {
            let a = i8x16::from_slice(s, &[0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8]);
            i8x16::not(a).into()
        }
    }
}

test_wasm_simd_parity! {
    fn and_u8x16() {
        |s| -> [u8; 16] {
            let a = u8x16::from_slice(s, &[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
            let b = u8x16::from_slice(s, &[85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85]);
            (a & b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn or_u8x16() {
        |s| -> [u8; 16] {
            let a = u8x16::from_slice(s, &[0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8]);
            let b = u8x16::from_slice(s, &[1, 1, 1, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0]);
            (a | b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn xor_u8x16() {
        |s| -> [u8; 16] {
            let a = u8x16::from_slice(s, &[0, 1, 2, 3, 4, 5, 6, 7, 1, 1, 1, 1, 0, 0, 0, 0]);
            let b = u8x16::from_slice(s, &[1, 1, 0, 0, 5, 4, 7, 6, 1, 0, 1, 0, 1, 0, 1, 0]);
            (a ^ b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn not_u8x16() {
        |s| -> [u8; 16] {
            let a = u8x16::from_slice(s, &[0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8]);
            u8x16::not(a).into()
        }
    }
}

test_wasm_simd_parity! {
    fn and_mask8x16() {
        |s| -> [i8; 16] {
            let a = mask8x16::from_slice(s, &[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
            let b = mask8x16::from_slice(s, &[85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85]);
            (a & b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn or_mask8x16() {
        |s| -> [i8; 16] {
            let a = mask8x16::from_slice(s, &[0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8]);
            let b = mask8x16::from_slice(s, &[1, 1, 1, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0]);
            (a | b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn xor_mask8x16() {
        |s| -> [i8; 16] {
            let a = mask8x16::from_slice(s, &[0, 1, 2, 3, 4, 5, 6, 7, 1, 1, 1, 1, 0, 0, 0, 0]);
            let b = mask8x16::from_slice(s, &[1, 1, 0, 0, 5, 4, 7, 6, 1, 0, 1, 0, 1, 0, 1, 0]);
            (a ^ b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn not_mask8x16() {
        |s| -> [i8; 16] {
            let a = mask8x16::from_slice(s, &[0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8]);
            mask8x16::not(a).into()
        }
    }
}

// Interleave tests

test_wasm_simd_parity! {
    fn load_interleaved_128_u32x16() {
        |s| -> [u32; 16] {
            let data: [u32; 16] = [
                1, 2, 3, 4,
                10, 20, 30, 40,
                100, 200, 300, 400,
                1000, 2000, 3000, 4000,
            ];

            s.load_interleaved_128_u32x16(&data).into()
        }
    }
}

test_wasm_simd_parity! {
    fn load_interleaved_128_u16x32() {
        |s| -> [u16; 32] {
            let data: [u16; 32] = [
                1,  2, 3, 4, 5, 6, 7, 8,
                10, 20, 30, 40, 50, 60, 70, 80,
                100, 200, 300, 400, 500, 600, 700, 800,
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,
            ];

            s.load_interleaved_128_u16x32(&data).into()
        }
    }
}

test_wasm_simd_parity! {
    fn load_interleaved_128_u8x64() {
        |s| -> [u8; 64] {
            let data: [u8; 64] = [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            ];

            s.load_interleaved_128_u8x64(&data).into()
        }
    }
}
