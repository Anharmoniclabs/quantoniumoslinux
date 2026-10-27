# SPDX-License-Identifier: LicenseRef-SPX-Proprietary
# Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
#
# QuantoniumOS Linux middleware: rftmwd (daemon) + rftmwctl (client).
# Built for distribution: no -march=native. The NASM kernels are AVX2/FMA and
# rftmwd only calls them after a runtime CPUID check.

cmake_minimum_required(VERSION 3.18)
project(rftmwd VERSION 0.1.0 LANGUAGES C CXX ASM_NASM)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

find_path(FFTW3_INCLUDE_DIR fftw3.h REQUIRED)
find_library(FFTW3_LIBRARY NAMES fftw3 REQUIRED)

set(RFTMW_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/src/rftmw_native"
    CACHE PATH "Path to runtime/src/rftmw_native")
if(NOT EXISTS "${RFTMW_NATIVE_DIR}/rftmw_core.hpp")
    message(FATAL_ERROR "rftmw_native sources not found at ${RFTMW_NATIVE_DIR}")
endif()

set(CMAKE_ASM_NASM_OBJECT_FORMAT elf64)
add_library(rftmw_os_kernels STATIC
    ${RFTMW_NATIVE_DIR}/rftmw_sgemv_f32.asm
    ${RFTMW_NATIVE_DIR}/rftmw_spectral_dot.asm
)
set_target_properties(rftmw_os_kernels PROPERTIES LINKER_LANGUAGE C)

add_executable(rftmwd src/rftmwd.cpp)
target_include_directories(rftmwd PRIVATE ${RFTMW_NATIVE_DIR} ${FFTW3_INCLUDE_DIR})
target_compile_definitions(rftmwd PRIVATE RFTMW_ENABLE_ASM=0 RFTMW_HAVE_FFTW3=1)
target_compile_options(rftmwd PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-unknown-pragmas)
target_link_libraries(rftmwd PRIVATE rftmw_os_kernels ${FFTW3_LIBRARY})
target_link_options(rftmwd PRIVATE -Wl,-z,noexecstack)

add_executable(test_fast_canonical src/test_fast_canonical.cpp)
target_include_directories(test_fast_canonical PRIVATE ${RFTMW_NATIVE_DIR} ${FFTW3_INCLUDE_DIR})
target_compile_definitions(test_fast_canonical PRIVATE RFTMW_ENABLE_ASM=0 RFTMW_HAVE_FFTW3=1)
target_compile_options(test_fast_canonical PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
target_link_libraries(test_fast_canonical PRIVATE rftmw_os_kernels ${FFTW3_LIBRARY})
enable_testing()
add_test(NAME fast_canonical_rft COMMAND test_fast_canonical)

add_executable(rftmwctl src/rftmwctl.c)
target_compile_options(rftmwctl PRIVATE -Wall -Wextra)

install(TARGETS rftmwd RUNTIME DESTINATION lib/quantoniumos)
install(TARGETS rftmwctl RUNTIME DESTINATION bin)
