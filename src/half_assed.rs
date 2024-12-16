// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Much of the code below is copy-pasted from half-rs by Kathryn Long, version 2.4.1.

#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct f16(u16);

impl f16 {
    /// Constructs a 16-bit floating point value from the raw bits.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u16) -> f16 {
        f16(bits)
    }

    /// Constructs a 16-bit floating point value from a 32-bit floating point value.
    ///
    /// This function is identical to [`from_f32`][Self::from_f32] except it never uses hardware
    /// intrinsics, which allows it to be `const`. [`from_f32`][Self::from_f32] should be preferred
    /// in any non-`const` context.
    ///
    /// This operation is lossy. If the 32-bit value is to large to fit in 16-bits, ±∞ will result.
    /// NaN values are preserved. 32-bit subnormal values are too tiny to be represented in 16-bits
    /// and result in ±0. Exponents that underflow the minimum 16-bit exponent will result in 16-bit
    /// subnormals or ±0. All other values are truncated and rounded to the nearest representable
    /// 16-bit value.
    #[inline]
    #[must_use]
    pub const fn from_f32_const(value: f32) -> f16 {
        f16(f32_to_f16_fallback(value))
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon", enable = "fp16")]
    #[inline]
    #[must_use]
    pub fn from_f32(value: f32) -> f16 {
        unsafe {
            let result: u16;
            core::arch::asm!(
                "fcvt {0:h}, {1:s}",
                out(vreg) result,
                in(vreg) value,
                options(pure, nomem, nostack, preserves_flags)
            );
            f16::from_bits(result)
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[inline]
    #[must_use]
    pub fn from_f32(value: f32) -> f16 {
        from_f32_const(value)
    }

    /// Converts a [`f16`] into the underlying bit representation.
    #[inline]
    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }
}

/////////////// Fallbacks ////////////////

// In the below functions, round to nearest, with ties to even.
// Let us call the most significant bit that will be shifted out the round_bit.
//
// Round up if either
//  a) Removed part > tie.
//     (mantissa & round_bit) != 0 && (mantissa & (round_bit - 1)) != 0
//  b) Removed part == tie, and retained part is odd.
//     (mantissa & round_bit) != 0 && (mantissa & (2 * round_bit)) != 0
// (If removed part == tie and retained part is even, do not round up.)
// These two conditions can be combined into one:
//     (mantissa & round_bit) != 0 && (mantissa & ((round_bit - 1) | (2 * round_bit))) != 0
// which can be simplified into
//     (mantissa & round_bit) != 0 && (mantissa & (3 * round_bit - 1)) != 0

#[inline]
pub(crate) const fn f32_to_f16_fallback(value: f32) -> u16 {
    let x: u32 = value.to_bits();

    // Extract IEEE754 components
    let sign = x & 0x8000_0000u32;
    let exp = x & 0x7F80_0000u32;
    let man = x & 0x007F_FFFFu32;

    // Check for all exponent bits being set, which is Infinity or NaN
    if exp == 0x7F80_0000u32 {
        // Set mantissa MSB for NaN (and also keep shifted mantissa bits)
        let nan_bit = if man == 0 { 0 } else { 0x0200u32 };
        return ((sign >> 16) | 0x7C00u32 | nan_bit | (man >> 13)) as u16;
    }

    // The number is normalized, start assembling half precision version
    let half_sign = sign >> 16;
    // Unbias the exponent, then bias for half precision
    let unbiased_exp = ((exp >> 23) as i32) - 127;
    let half_exp = unbiased_exp + 15;

    // Check for exponent overflow, return +infinity
    if half_exp >= 0x1F {
        return (half_sign | 0x7C00u32) as u16;
    }

    // Check for underflow
    if half_exp <= 0 {
        // Check mantissa for what we can do
        if 14 - half_exp > 24 {
            // No rounding possibility, so this is a full underflow, return signed zero
            return half_sign as u16;
        }
        // Don't forget about hidden leading mantissa bit when assembling mantissa
        let man = man | 0x0080_0000u32;
        let mut half_man = man >> (14 - half_exp);
        // Check for rounding (see comment above functions)
        let round_bit = 1 << (13 - half_exp);
        if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
            half_man += 1;
        }
        // No exponent for subnormals
        return (half_sign | half_man) as u16;
    }

    // Rebias the exponent
    let half_exp = (half_exp as u32) << 10;
    let half_man = man >> 13;
    // Check for rounding (see comment above functions)
    let round_bit = 0x0000_1000u32;
    if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
        // Round it
        ((half_sign | half_exp | half_man) + 1) as u16
    } else {
        (half_sign | half_exp | half_man) as u16
    }
}
