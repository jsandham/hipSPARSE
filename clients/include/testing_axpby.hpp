/* ************************************************************************
 * Copyright (C) 2020 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef TESTING_AXPBY_HPP
#define TESTING_AXPBY_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <typeinfo>

using namespace hipsparse_test;

void testing_axpby_bad_arg(void)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)

    int64_t size = 100;
    int64_t nnz  = 100;

    float alpha = 3.7f;
    float beta  = 1.2f;

    hipsparseIndexType_t idxType  = HIPSPARSE_INDEX_32I;
    hipsparseIndexBase_t idxBase  = HIPSPARSE_INDEX_BASE_ZERO;
    hipDataType          dataType = HIP_R_32F;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    auto dx_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto dx_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dy_managed     = hipsparse_unique_ptr{device_malloc(sizeof(float) * size), device_free};

    float* dx_val = (float*)dx_val_managed.get();
    int*   dx_ind = (int*)dx_ind_managed.get();
    float* dy     = (float*)dy_managed.get();

    // Structures
    hipsparseSpVecDescr_t x;
    hipsparseDnVecDescr_t y;

    verify_hipsparse_status_success(
        hipsparseCreateSpVec(&x, size, nnz, dx_ind, dx_val, idxType, idxBase, dataType), "Success");
    verify_hipsparse_status_success(hipsparseCreateDnVec(&y, size, dy, dataType), "Success");

    // Axpby
    verify_hipsparse_status_invalid_handle(hipsparseAxpby(nullptr, &alpha, x, &beta, y));
    verify_hipsparse_status_invalid_pointer(hipsparseAxpby(handle, nullptr, x, &beta, y),
                                            "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseAxpby(handle, &alpha, nullptr, &beta, y),
                                            "Error: x is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseAxpby(handle, &alpha, x, nullptr, y),
                                            "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseAxpby(handle, &alpha, x, &beta, nullptr),
                                            "Error: y is nullptr");

    // Destruct
    verify_hipsparse_status_success(hipsparseDestroySpVec(x), "Success");
    verify_hipsparse_status_success(hipsparseDestroyDnVec(y), "Success");
#endif
}

template <typename I, typename T>
hipsparseStatus_t testing_axpby(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)

    std::cout << "argus.M: " << argus.M << std::endl;
    std::cout << "argus.N: " << argus.N << std::endl;
    std::cout << "argus.K: " << argus.K << std::endl;
    std::cout << "argus.nnz: " << argus.nnz << std::endl;
    std::cout << "argus.block_dim: " << argus.block_dim << std::endl;
    std::cout << "argus.row_block_dimA: " << argus.row_block_dimA << std::endl;
    std::cout << "argus.col_block_dimA: " << argus.col_block_dimA << std::endl;
    std::cout << "argus.row_block_dimB: " << argus.row_block_dimB << std::endl;
    std::cout << "argus.col_block_dimB: " << argus.col_block_dimB << std::endl;
    std::cout << "argus.lda: " << argus.lda << std::endl;
    std::cout << "argus.ldb: " << argus.ldb << std::endl;
    std::cout << "argus.ldc: " << argus.ldc << std::endl;
    std::cout << "argus.batch_count: " << argus.batch_count << std::endl;
    std::cout << "argus.index_type_I: " << argus.index_type_I << std::endl;
    std::cout << "argus.index_type_J: " << argus.index_type_J << std::endl;
    std::cout << "argus.compute_type: " << argus.compute_type << std::endl;
    std::cout << "argus.alpha: " << argus.alpha << std::endl;
    std::cout << "argus.alphai: " << argus.alphai << std::endl;
    std::cout << "argus.beta: " << argus.beta << std::endl;
    std::cout << "argus.betai: " << argus.betai << std::endl;
    std::cout << "argus.threshold: " << argus.threshold << std::endl;
    std::cout << "argus.percentage: " << argus.percentage << std::endl;
    std::cout << "argus.transA: " << argus.transA << std::endl;
    std::cout << "argus.transB: " << argus.transB << std::endl;
    std::cout << "argus.baseA: " << argus.baseA << std::endl;
    std::cout << "argus.baseB: " << argus.baseB << std::endl;
    std::cout << "argus.baseC: " << argus.baseC << std::endl;
    std::cout << "argus.baseD: " << argus.baseD << std::endl;
    std::cout << "argus.action: " << argus.action << std::endl;
    std::cout << "argus.part: " << argus.part << std::endl;
    std::cout << "argus.diag_type: " << argus.diag_type << std::endl;
    std::cout << "argus.fill_mode: " << argus.fill_mode << std::endl;
    std::cout << "argus.solve_policy: " << argus.solve_policy << std::endl;
    std::cout << "argus.dirA: " << argus.dirA << std::endl;
    std::cout << "argus.orderA: " << argus.orderA << std::endl;
    std::cout << "argus.orderB: " << argus.orderB << std::endl;
    std::cout << "argus.orderC: " << argus.orderC << std::endl;
    std::cout << "argus.formatA: " << argus.formatA << std::endl;
    std::cout << "argus.formatB: " << argus.formatB << std::endl;
    std::cout << "argus.csr2csc_alg: " << argus.csr2csc_alg << std::endl;
    std::cout << "argus.dense2sparse_alg: " << argus.dense2sparse_alg << std::endl;
    std::cout << "argus.sparse2dense_alg: " << argus.sparse2dense_alg << std::endl;
    std::cout << "argus.sddmm_alg: " << argus.sddmm_alg << std::endl;
    std::cout << "argus.spgemm_alg: " << argus.spgemm_alg << std::endl;
    std::cout << "argus.spmm_alg: " << argus.spmm_alg << std::endl;
    std::cout << "argus.spmv_alg: " << argus.spmv_alg << std::endl;
    std::cout << "argus.spsm_alg: " << argus.spsm_alg << std::endl;
    std::cout << "argus.spsv_alg: " << argus.spsv_alg << std::endl;
    std::cout << "argus.numericboost: " << argus.numericboost << std::endl;
    std::cout << "argus.boosttol: " << argus.boosttol << std::endl;
    std::cout << "argus.boostval: " << argus.boostval << std::endl;
    std::cout << "argus.boostvali: " << argus.boostvali << std::endl;
    std::cout << "argus.ell_width: " << argus.ell_width << std::endl;
    std::cout << "argus.permute: " << argus.permute << std::endl;
    std::cout << "argus.gtsv_alg: " << argus.gtsv_alg << std::endl;
    std::cout << "argus.gpsv_alg: " << argus.gpsv_alg << std::endl;
    std::cout << "argus.unit_check: " << argus.unit_check << std::endl;
    std::cout << "argus.timing: " << argus.timing << std::endl;
    std::cout << "argus.iters: " << argus.iters << std::endl;



    std::cout << "AAAA" << std::endl;

    I size = argus.N;
    I nnz  = argus.nnz;

    T alpha = make_DataType<T>(argus.alpha);
    T beta  = make_DataType<T>(argus.beta);

    hipsparseIndexBase_t idxBase = argus.baseA;

    // Index and data type
    hipsparseIndexType_t idxType  = getIndexType<I>();
    hipDataType          dataType = getDataType<T>();

    std::cout << "BBBB" << std::endl;

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    // Host structures
    std::vector<I> hx_ind(nnz);
    std::vector<T> hx_val(nnz);
    std::vector<T> hy(size);
    std::vector<T> hy_gold(size);

    // Initial Data on CPU
    srand(12345ULL);
    hipsparseInitIndex(hx_ind.data(), nnz, 1, size);
    hipsparseInit<T>(hx_val, 1, nnz);
    hipsparseInit<T>(hy, 1, size);

    std::cout << "CCCC" << std::endl;
    hy_gold = hy;

    // Allocate memory on device
    auto dx_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(I) * nnz), device_free};
    auto dx_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dy_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * size), device_free};

    I* dx_ind = (I*)dx_ind_managed.get();
    T* dx_val = (T*)dx_val_managed.get();
    T* dy     = (T*)dy_managed.get();

    std::cout << "DDDD" << std::endl;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind.data(), sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val, hx_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * size, hipMemcpyHostToDevice));

    // Create structures
    hipsparseSpVecDescr_t x;
    hipsparseDnVecDescr_t y;

    CHECK_HIPSPARSE_ERROR(
        hipsparseCreateSpVec(&x, size, nnz, dx_ind, dx_val, idxType, idxBase, dataType));
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnVec(&y, size, dy, dataType));

    std::cout << "EEEE" << std::endl;
    if(argus.unit_check)
    {
        // Axpby
        CHECK_HIPSPARSE_ERROR(hipsparseAxpby(handle, &alpha, x, &beta, y));

        // Copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T) * size, hipMemcpyDeviceToHost));

        // CPU
        for(int64_t i = 0; i < size; ++i)
        {
            hy_gold[i] = testing_mult(beta, hy_gold[i]);
        }

        for(int64_t i = 0; i < nnz; ++i)
        {
            hy_gold[hx_ind[i] - idxBase]
                = testing_fma(alpha, hx_val[i], hy_gold[hx_ind[i] - idxBase]);
        }

        // Verify results against host
        unit_check_general(1, size, 1, hy_gold.data(), hy.data());
    }

    std::cout << "FFFF" << std::endl;

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseAxpby(handle, &alpha, x, &beta, y));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseAxpby(handle, &alpha, x, &beta, y));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = axpby_gflop_count(nnz);
        double gbyte_count = axpby_gbyte_count<T>(nnz);

        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);
        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);

        display_timing_info(display_key_t::size,
                            size,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::alpha,
                            alpha,
                            display_key_t::beta,
                            beta,
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    std::cout << "GGGG" << std::endl;
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpVec(x));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnVec(y));

#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_AXPBY_HPP
