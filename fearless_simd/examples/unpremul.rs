// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::{Level, Simd};

#[cfg(target_arch = "aarch64")]
use fearless_simd::{aarch64::Fp16, f16};

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn unpremul_f1p6(fp16: Fp16, rgba: &mut [u8]) {
    let neon = fp16.neon;
    let scale = fp16.splat_f16x8(f16::from_f32_const(255.0)).into();
    let ones = fp16.splat_f16x8(f16::from_f32_const(1.0)).into();
    let mut iter = rgba.chunks_exact_mut(32);
    for chunk in &mut iter {
        let mut inp = unsafe { neon.vld4_u8(chunk.as_ptr()) };
        let alpha_u16 = neon.vmovl_u8(inp.3);
        let alphas = fp16.fp16.vcvtq_f16_u16(alpha_u16);
        let recip_raw = fp16.fp16.vdivq_f16(scale, alphas);
        let mask = fp16.fp16.vceqzq_f16(alphas);
        let recip = neon.vbslq_u16(mask, ones, recip_raw);
        let red = fp16.fp16.vcvtq_f16_u16(neon.vmovl_u8(inp.0));
        let red2 = fp16.fp16.vmulq_f16(red, recip);
        let red3 = fp16.fp16.vcvtnq_u16_f16(red2);
        inp.0 = neon.vqmovn_u16(red3);
        let green = fp16.fp16.vcvtq_f16_u16(neon.vmovl_u8(inp.1));
        let green2 = fp16.fp16.vmulq_f16(green, recip);
        let green3 = fp16.fp16.vcvtnq_u16_f16(green2);
        inp.1 = neon.vqmovn_u16(green3);
        let blue = fp16.fp16.vcvtq_f16_u16(neon.vmovl_u8(inp.2));
        let blue2 = fp16.fp16.vmulq_f16(blue, recip);
        let blue3 = fp16.fp16.vcvtnq_u16_f16(blue2);
        inp.2 = neon.vqmovn_u16(blue3);
        unsafe {
            neon.vst4_u8(chunk.as_mut_ptr(), inp);
        }
    }
    for chunk in iter.into_remainder().chunks_exact_mut(4) {
        let alpha = chunk[3];
        if alpha != 0 && alpha != 255 {
            let recip = 0xff00 / alpha as u32;
            for x in &mut chunk[..3] {
                let y = *x as u32 * recip;
                *x = ((128 + y) >> 8).min(255) as u8;
            }
        }
    }
}

#[inline(never)]
fn unpremultiply(_level: Level, rgba: &mut [u8]) {
    #[cfg(target_arch = "aarch64")]
    if let Some(fp16) = _level.as_fp16() {
        fp16.vectorize(
            #[inline(always)]
            || unpremul_f1p6(fp16, rgba),
        );
        return;
    }
    for chunk in rgba.chunks_exact_mut(4) {
        let alpha = chunk[3];
        if alpha != 0 && alpha != 255 {
            let recip = 0xff00 / alpha as u32;
            for x in &mut chunk[..3] {
                let y = *x as u32 * recip;
                *x = ((128 + y) >> 8).min(255) as u8;
            }
        }
    }
}

fn main() {
    let level = Level::new();
    let mut buf = vec![128u8; 1 << 20];
    let start = std::time::Instant::now();
    unpremultiply(level, &mut buf);
    println!("{:?}, elapsed {:?}", &buf[0..4], start.elapsed())
}
