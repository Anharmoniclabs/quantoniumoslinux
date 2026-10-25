; SPDX-License-Identifier: LicenseRef-SPX-Proprietary
; Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
;
; rftmw_spectral_dot.asm
; ----------------------
; AVX2/FMA sparse spectral dot product for RFTMW-space matmul.
;
; Computes: Re(C^H * x_spec)
;   = sum_i ( coeff_re[i] * x_spec_re[indices[i]]
;           + coeff_im[i] * x_spec_im[indices[i]] )
;
; This is the innermost loop of spectral-domain inference.
; Uses VGATHERDPS for indexed access into the transformed input.
;
; System V AMD64 ABI:
;   rdi = indices    (const uint16_t*)
;   rsi = coeff_re   (const float*)
;   rdx = coeff_im   (const float*)
;   rcx = x_spec_re  (const float*)
;   r8  = x_spec_im  (const float*)
;   r9d = n_nonzero  (int)
;   returns float result in xmm0

default rel
section .text

global rftmw_spectral_sparse_dot_asm

align 32
rftmw_spectral_sparse_dot_asm:
    ; Save callee-saved registers we'll use
    push rbx

    test r9d, r9d
    jle .zero_result

    ; Compute vectorized loop bound: n_nonzero & ~7
    mov r10d, r9d
    and r10d, -8          ; r10d = n_aligned (multiple of 8)

    vxorps ymm0, ymm0, ymm0   ; accumulator

    xor eax, eax          ; i = 0

    ; Check if we have at least 8 elements
    cmp r10d, 8
    jl .tail_start

; ---------------------------------------------------------------
; Vectorized loop: process 8 sparse entries per iteration
; Uses VGATHERDPS for indexed lookup into x_spec arrays
; ---------------------------------------------------------------
align 16
.vec_loop:
    cmp eax, r10d
    jge .tail_start

    ; 1. Load 8 uint16 indices and zero-extend to 8 int32
    ;    indices are at [rdi + eax*2]
    lea rbx, [rdi + rax*2]
    vpmovzxwd ymm4, [rbx]     ; 8 x uint16 -> 8 x int32

    ; 2. Gather x_spec_re[idx[i]] for 8 indices
    ;    VGATHERDPS: ymm_dst = gather(base, ymm_idx, scale=4)
    ;    Mask must be all-ones (set to all 1s in top bit)
    vpcmpeqd ymm5, ymm5, ymm5  ; all-ones mask for gather
    vxorps ymm1, ymm1, ymm1    ; clear destination
    vgatherdps ymm1, [rcx + ymm4*4], ymm5
    ; ymm1 = x_spec_re[indices[0..7]]

    ; 3. Gather x_spec_im[idx[i]] for 8 indices
    vpcmpeqd ymm5, ymm5, ymm5  ; reset mask (gather clears it)
    vxorps ymm2, ymm2, ymm2
    vgatherdps ymm2, [r8 + ymm4*4], ymm5
    ; ymm2 = x_spec_im[indices[0..7]]

    ; 4. Load coeff_re[i..i+7] and coeff_im[i..i+7]
    vmovups ymm6, [rsi + rax*4]   ; coeff_re
    vmovups ymm7, [rdx + rax*4]   ; coeff_im

    ; 5. acc += coeff_re * gathered_re
    vfmadd231ps ymm0, ymm6, ymm1

    ; 6. acc += coeff_im * gathered_im
    vfmadd231ps ymm0, ymm7, ymm2

    add eax, 8
    jmp .vec_loop

; ---------------------------------------------------------------
; Horizontal reduction of ymm0 -> single float
; ---------------------------------------------------------------
.tail_start:
    ; Reduce ymm0: 8 floats -> 1 float
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1     ; 4 floats
    vhaddps xmm0, xmm0, xmm0    ; 2 floats
    vhaddps xmm0, xmm0, xmm0    ; 1 float

    ; xmm0 now has the vectorized partial sum

; ---------------------------------------------------------------
; Scalar tail loop for remaining elements (n_nonzero % 8)
; ---------------------------------------------------------------
.tail_loop:
    cmp eax, r9d
    jge .done

    ; Load index: uint16 at [rdi + eax*2]
    movzx ebx, word [rdi + rax*2]

    ; Load x_spec_re[idx] and x_spec_im[idx]
    vmovss xmm1, [rcx + rbx*4]    ; x_spec_re[idx]
    vmovss xmm2, [r8 + rbx*4]     ; x_spec_im[idx]

    ; Load coeff_re[i] and coeff_im[i]
    vmovss xmm3, [rsi + rax*4]    ; coeff_re[i]
    vmovss xmm4, [rdx + rax*4]    ; coeff_im[i]

    ; acc += coeff_re * x_re + coeff_im * x_im
    vfmadd231ss xmm0, xmm3, xmm1
    vfmadd231ss xmm0, xmm4, xmm2

    inc eax
    jmp .tail_loop

.done:
    vzeroupper
    pop rbx
    ret

.zero_result:
    vxorps xmm0, xmm0, xmm0
    pop rbx
    ret
