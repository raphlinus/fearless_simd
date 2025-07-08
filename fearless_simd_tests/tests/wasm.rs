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
    fn copysign_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[1.0, -2.0, -3.0, 4.0]);
            let b = f32x4::from_slice(s, &[-1.0, 1.0, -1.0, 1.0]);
            a.copysign(b).into()
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
    fn fract_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[1.7, -2.3, 3.9, -4.1]);
            s.fract_f32x4(a).into()
        }
    }
}

test_wasm_simd_parity! {
    fn trunc_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[2.9, -3.2, 0.0, 0.5]);
            a.trunc().into()
        }
    }
}

#[wasm_bindgen_test]
fn trunc_f32x4_special_values() {
    fn test_impl<S: Simd>(s: S) -> [f32; 4] {
        let a = f32x4::from_slice(s, &[f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -f32::NAN]);
        a.trunc().into()
    }

    simd_dispatch!(test(level) -> [f32; 4] = test_impl);
    let wasm_result = test(Level::WasmSimd128(wasm32::WasmSimd128::new_unchecked()));

    // Note: f32::NAN != f32::NAN hence we transmute to compare the bit pattern. In this case NaN
    // bit pattern is preserved.
    unsafe {
        assert_eq!(
            std::mem::transmute::<[f32; 4], [u32; 4]>(wasm_result),
            std::mem::transmute::<[f32; 4], [u32; 4]>([
                f32::NAN,
                f32::NEG_INFINITY,
                f32::INFINITY,
                -f32::NAN
            ]),
            "Wasm did not match expected result."
        );
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
    fn cvt_u32_f32x4() {
        |s| -> [u32; 4] {
            let a = f32x4::from_slice(s, &[
                -1.0,
                42.7,
                5e9,
                f32::NAN,
            ]);
            a.cvt_u32().into()
        }
    }
}

test_wasm_simd_parity! {
    fn cvt_f32_u32x4() {
        |s| -> [f32; 4] {
            let a = u32x4::from_slice(s, &[
                0,
                42,
                1000000,
                u32::MAX,
            ]);
            a.cvt_f32().into()
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

// Zip Load and High tests

test_wasm_simd_parity! {
    fn zip_low_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[0.0, 1.0, 2.0, 3.0]);
            let b = f32x4::from_slice(s, &[4.0, 5.0, 6.0, 7.0]);
            s.zip_low_f32x4(a, b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn zip_high_f32x4() {
        |s| -> [f32; 4] {
            let a = f32x4::from_slice(s, &[0.0, 1.0, 2.0, 3.0]);
            let b = f32x4::from_slice(s, &[4.0, 5.0, 6.0, 7.0]);
            s.zip_high_f32x4(a, b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn zip_low_i8x16() {
        |s| -> [i8; 16] {
            let a = i8x16::from_slice(s, &[1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16]);
            let b = i8x16::from_slice(s, &[17, -18, 19, -20, 21, -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32]);
            s.zip_low_i8x16(a, b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn zip_high_i8x16() {
        |s| -> [i8; 16] {
            let a = i8x16::from_slice(s, &[1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16]);
            let b = i8x16::from_slice(s, &[17, -18, 19, -20, 21, -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32]);
            s.zip_high_i8x16(a, b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn zip_low_u8x16() {
        |s| -> [u8; 16] {
            let a = u8x16::from_slice(s, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            let b = u8x16::from_slice(s, &[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]);
            s.zip_low_u8x16(a, b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn zip_high_u8x16() {
        |s| -> [u8; 16] {
            let a = u8x16::from_slice(s, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            let b = u8x16::from_slice(s, &[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]);
            s.zip_high_u8x16(a, b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn zip_low_i16x8() {
        |s| -> [i16; 8] {
            let a = i16x8::from_slice(s, &[1, -2, 3, -4, 5, -6, 7, -8]);
            let b = i16x8::from_slice(s, &[9, -10, 11, -12, 13, -14, 15, -16]);
            s.zip_low_i16x8(a, b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn zip_high_i16x8() {
        |s| -> [i16; 8] {
            let a = i16x8::from_slice(s, &[1, -2, 3, -4, 5, -6, 7, -8]);
            let b = i16x8::from_slice(s, &[9, -10, 11, -12, 13, -14, 15, -16]);
            s.zip_high_i16x8(a, b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn zip_low_u16x8() {
        |s| -> [u16; 8] {
            let a = u16x8::from_slice(s, &[0, 1, 2, 3, 4, 5, 6, 7]);
            let b = u16x8::from_slice(s, &[8, 9, 10, 11, 12, 13, 14, 15]);
            s.zip_low_u16x8(a, b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn zip_high_u16x8() {
        |s| -> [u16; 8] {
            let a = u16x8::from_slice(s, &[0, 1, 2, 3, 4, 5, 6, 7]);
            let b = u16x8::from_slice(s, &[8, 9, 10, 11, 12, 13, 14, 15]);
            s.zip_high_u16x8(a, b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn zip_low_i32x4() {
        |s| -> [i32; 4] {
            let a = i32x4::from_slice(s, &[1, -2, 3, -4]);
            let b = i32x4::from_slice(s, &[5, -6, 7, -8]);
            s.zip_low_i32x4(a, b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn zip_high_i32x4() {
        |s| -> [i32; 4] {
            let a = i32x4::from_slice(s, &[1, -2, 3, -4]);
            let b = i32x4::from_slice(s, &[5, -6, 7, -8]);
            s.zip_high_i32x4(a, b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn zip_low_u32x4() {
        |s| -> [u32; 4] {
            let a = u32x4::from_slice(s, &[0, 1, 2, 3]);
            let b = u32x4::from_slice(s, &[4, 5, 6, 7]);
            s.zip_low_u32x4(a, b).into()
        }
    }
}

test_wasm_simd_parity! {
    fn zip_high_u32x4() {
        |s| -> [u32; 4] {
            let a = u32x4::from_slice(s, &[0, 1, 2, 3]);
            let b = u32x4::from_slice(s, &[4, 5, 6, 7]);
            s.zip_high_u32x4(a, b).into()
        }
    }
}

// Right Shift

test_wasm_simd_parity! {
    fn shr_i8x16() {
        |s| -> [i8; 16] {
            let a = i8x16::from_slice(s, &[
                -128, -64, -32, -16, -8, -4, -2, -1,
                127, 64, 32, 16, 8, 4, 2, 1
            ]);
            a.shr(2).into()
        }
    }
}

test_wasm_simd_parity! {
    fn shr_u8x16() {
        |s| -> [u8; 16] {
            let a = u8x16::from_slice(s, &[
                255, 128, 64, 32, 16, 8, 4, 2,
                254, 127, 63, 31, 15, 7, 3, 1
            ]);
            a.shr(2).into()
        }
    }
}

test_wasm_simd_parity! {
    fn shr_i16x8() {
        |s| -> [i16; 8] {
            let a = i16x8::from_slice(s, &[
                -32768, -16384, -1024, -1,
                32767, 16384, 1024, 1
            ]);
            a.shr(4).into()
        }
    }
}

test_wasm_simd_parity! {
    fn shr_u16x8() {
        |s| -> [u16; 8] {
            let a = u16x8::from_slice(s, &[
                65535, 32768, 16384, 8192,
                4096, 2048, 1024, 512
            ]);
            a.shr(4).into()
        }
    }
}

test_wasm_simd_parity! {
    fn shr_i32x4() {
        |s| -> [i32; 4] {
            let a = i32x4::from_slice(s, &[
                i32::MIN,
                -65536,
                65536,
                i32::MAX
            ]);
            a.shr(8).into()
        }
    }
}

test_wasm_simd_parity! {
    fn shr_u32x4() {
        |s| -> [u32; 4] {
            let a = u32x4::from_slice(s, &[
                u32::MAX,
                2147483648,
                65536,
                256
            ]);
            a.shr(8).into()
        }
    }
}

// Select

test_wasm_simd_parity! {
    fn select_f32x4() {
        |s| -> [f32; 4] {
            let mask = mask32x4::from_slice(s, &[-1, 0, -1, 0]);
            let b = f32x4::from_slice(s, &[1.0, 2.0, 3.0, 4.0]);
            let c = f32x4::from_slice(s, &[5.0, 6.0, 7.0, 8.0]);
            mask.select(b, c).into()
        }
    }
}

test_wasm_simd_parity! {
    fn select_i8x16() {
        |s| -> [i8; 16] {
            let mask = mask8x16::from_slice(s, &[-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]);
            let b = i8x16::from_slice(s, &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, -10, -20, -30, -40]);
            let c = i8x16::from_slice(s, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -2, -3, -4]);
            mask.select(b, c).into()
        }
    }
}

test_wasm_simd_parity! {
    fn select_u8x16() {
        |s| -> [u8; 16] {
            let mask = mask8x16::from_slice(s, &[0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1]);
            let b = u8x16::from_slice(s, &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]);
            let c = u8x16::from_slice(s, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
            mask.select(b, c).into()
        }
    }
}

test_wasm_simd_parity! {
    fn select_mask8x16() {
        |s| -> [i8; 16] {
            let mask = mask8x16::from_slice(s, &[-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0]);
            let b = mask8x16::from_slice(s, &[-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]);
            let c = mask8x16::from_slice(s, &[0, -1, 0, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1]);
            let result: mask8x16<_> = mask.select(b, c);
            result.into()
        }
    }
}

test_wasm_simd_parity! {
    fn select_i16x8() {
        |s| -> [i16; 8] {
            let mask = mask16x8::from_slice(s, &[-1, 0, -1, 0, -1, 0, -1, 0]);
            let b = i16x8::from_slice(s, &[100, 200, 300, 400, -100, -200, -300, -400]);
            let c = i16x8::from_slice(s, &[10, 20, 30, 40, -10, -20, -30, -40]);
            mask.select(b, c).into()
        }
    }
}

test_wasm_simd_parity! {
    fn select_u16x8() {
        |s| -> [u16; 8] {
            let mask = mask16x8::from_slice(s, &[0, -1, 0, -1, 0, -1, 0, -1]);
            let b = u16x8::from_slice(s, &[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]);
            let c = u16x8::from_slice(s, &[100, 200, 300, 400, 500, 600, 700, 800]);
            mask.select(b, c).into()
        }
    }
}

test_wasm_simd_parity! {
    fn select_mask16x8() {
        |s| -> [i16; 8] {
            let mask = mask16x8::from_slice(s, &[-1, -1, 0, 0, -1, -1, 0, 0]);
            let b = mask16x8::from_slice(s, &[-1, 0, -1, 0, -1, 0, -1, 0]);
            let c = mask16x8::from_slice(s, &[0, -1, 0, -1, 0, -1, 0, -1]);
            let result: mask16x8<_> = mask.select(b, c);
            result.into()
        }
    }
}

test_wasm_simd_parity! {
    fn select_i32x4() {
        |s| -> [i32; 4] {
            let mask = mask32x4::from_slice(s, &[-1, 0, 0, -1]);
            let b = i32x4::from_slice(s, &[10000, 20000, -30000, -40000]);
            let c = i32x4::from_slice(s, &[100, 200, -300, -400]);
            mask.select(b, c).into()
        }
    }
}

test_wasm_simd_parity! {
    fn select_u32x4() {
        |s| -> [u32; 4] {
            let mask = mask32x4::from_slice(s, &[0, -1, -1, 0]);
            let b = u32x4::from_slice(s, &[100000, 200000, 300000, 400000]);
            let c = u32x4::from_slice(s, &[1000, 2000, 3000, 4000]);
            mask.select(b, c).into()
        }
    }
}

test_wasm_simd_parity! {
    fn select_mask32x4() {
        |s| -> [i32; 4] {
            let mask = mask32x4::from_slice(s, &[-1, 0, -1, 0]);
            let b = mask32x4::from_slice(s, &[-1, -1, 0, 0]);
            let c = mask32x4::from_slice(s, &[0, 0, -1, -1]);
            let result: mask32x4<_> = mask.select(b, c);
            result.into()
        }
    }
}

// Widen

test_wasm_simd_parity! {
    fn widen_u8x16() {
        |s| -> [u16; 16] {
            let a = u8x16::from_slice(s, &[
                0, 1, 2, 3, 4, 5, 6, 7,
                8, 9, 10, 11, 12, 13, 14, 15
            ]);
            s.widen_u8x16(a).into()
        }
    }
}
