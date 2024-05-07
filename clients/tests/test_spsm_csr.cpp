/* ************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_spsm_csr.hpp"

#include <hipsparse.h>

template <typename T>
Arguments setup_spsm_csr_arguments()
{
    Arguments arg;
    arg.alpha     = make_DataType<T>(2.3);
    arg.alphai    = make_DataType<T>(0.0);
    arg.transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    arg.transB    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    arg.idx_baseA = HIPSPARSE_INDEX_BASE_ZERO;
    arg.diag_type = HIPSPARSE_DIAG_TYPE_NON_UNIT;
    arg.fill_mode = HIPSPARSE_FILL_MODE_LOWER;
    arg.orderA    = HIPSPARSE_ORDER_COL;
    arg.filename  = "nos3.bin";
    arg.timing    = 0;
    return arg;
}

// Only run tests for CUDA 11.1 or greater
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11010)
TEST(spsm_csr_bad_arg, spsm_csr_float)
{
    testing_spsm_csr_bad_arg();
}

TEST(spsm_csr, spsm_csr_i32_i32_float)
{
    hipsparseStatus_t status
        = testing_spsm_csr<int32_t, int32_t, float>(setup_spsm_csr_arguments<float>());
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

#if(!defined(CUDART_VERSION))
TEST(spsm_csr, spsm_csr_i64_i32_double)
{
    hipsparseStatus_t status
        = testing_spsm_csr<int64_t, int32_t, double>(setup_spsm_csr_arguments<double>());
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}
#endif

TEST(spsm_csr, spsm_csr_i64_i64_hipComplex)
{
    hipsparseStatus_t status
        = testing_spsm_csr<int64_t, int64_t, hipComplex>(setup_spsm_csr_arguments<float>());
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}
#endif
