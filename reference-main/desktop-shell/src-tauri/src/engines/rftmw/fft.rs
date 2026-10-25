// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
//
// fft.rs — Radix-2 DIT FFT (port of rftmw_spectral_kernels.c FFT section)

use std::f64::consts::PI;

/// In-place radix-2 decimation-in-time FFT.
/// `re` and `im` are real/imaginary arrays of length `n` (must be power of 2).
pub fn fft_f32(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    assert_eq!(n, im.len());
    assert!(n.is_power_of_two() && n >= 2);
    let log2n = n.trailing_zeros() as usize;

    // Bit-reverse permutation
    for i in 0..n {
        let j = bit_reverse(i, log2n);
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly
    let mut half_m = 1;
    for _stage in 0..log2n {
        let m = half_m << 1;
        let angle = -2.0 * PI / m as f64;
        let w_re_step = angle.cos() as f32;
        let w_im_step = angle.sin() as f32;

        let mut k = 0;
        while k < n {
            let mut w_re = 1.0_f32;
            let mut w_im = 0.0_f32;
            for j in 0..half_m {
                let p = k + j;
                let q = p + half_m;

                let t_re = w_re * re[q] - w_im * im[q];
                let t_im = w_re * im[q] + w_im * re[q];

                re[q] = re[p] - t_re;
                im[q] = im[p] - t_im;
                re[p] += t_re;
                im[p] += t_im;

                let new_w_re = w_re * w_re_step - w_im * w_im_step;
                let new_w_im = w_re * w_im_step + w_im * w_re_step;
                w_re = new_w_re;
                w_im = new_w_im;
            }
            k += m;
        }
        half_m = m;
    }
}

/// Inverse FFT: conjugate → FFT → conjugate → scale by 1/N.
pub fn ifft_f32(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    // Conjugate
    for v in im.iter_mut() {
        *v = -*v;
    }
    fft_f32(re, im);
    // Conjugate and scale
    let inv_n = 1.0 / n as f32;
    for v in re.iter_mut() {
        *v *= inv_n;
    }
    for v in im.iter_mut() {
        *v = -*v * inv_n;
    }
}

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_roundtrip() {
        let n = 256;
        let mut re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut im = vec![0.0_f32; n];
        let original = re.clone();

        fft_f32(&mut re, &mut im);
        ifft_f32(&mut re, &mut im);

        for (a, b) in re.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-4, "FFT roundtrip error: {} vs {}", a, b);
        }
    }
}
