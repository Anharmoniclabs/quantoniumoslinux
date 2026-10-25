; SPDX-License-Identifier: LicenseRef-SPX-Proprietary
; Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
;
; rftmw_sgemv_f32.asm
; -------------------
; Single-row AVX2/FMA float32 dot product used by the dense GEMV wrapper.
;
; Body is written against the Windows x64 ABI register assignment:
;   rcx = weights_row  (cols)
;   rdx = x            (cols)
;   r8  = out          (1 float)
;   r9d = cols
; On System V targets (elf64 / macho64) the entry shim remaps
; rdi, rsi, rdx, ecx into those registers. Only volatile registers
; are touched on either ABI.

default rel
section .text

global rftmw_dot_f32_asm

align 32
rftmw_dot_f32_asm:
%ifnidn __OUTPUT_FORMAT__, win64
    mov r9d, ecx
    mov r8, rdx
    mov rdx, rsi
    mov rcx, rdi
%endif
    test r9d, r9d
    jle .zero

    mov r10d, r9d
    and r10d, -8

    vxorps ymm0, ymm0, ymm0
    xor eax, eax

.vec_loop:
    cmp eax, r10d
    jae .reduce

    vmovups ymm1, [rcx + rax * 4]
    vmovups ymm2, [rdx + rax * 4]
    vfmadd231ps ymm0, ymm1, ymm2

    add eax, 8
    jmp .vec_loop

.reduce:
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

    mov eax, r10d

.tail_loop:
    cmp eax, r9d
    jae .store

    vmovss xmm1, [rcx + rax * 4]
    vmovss xmm2, [rdx + rax * 4]
    vfmadd231ss xmm0, xmm1, xmm2

    inc eax
    jmp .tail_loop

.store:
    vmovss [r8], xmm0
    vzeroupper
    ret

.zero:
    vxorps xmm0, xmm0, xmm0
    vmovss [r8], xmm0
    ret
