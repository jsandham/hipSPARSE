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

#include "testing_sddmm_csc.hpp"

#include <hipsparse.h>

template <typename T>
Arguments setup_sddmm_csc_arguments()
{
    Arguments arg;
    arg.alpha     = make_DataType<T>(2.0);
    arg.alphai    = make_DataType<T>(0.0);
    arg.beta      = make_DataType<T>(1.0);
    arg.betai     = make_DataType<T>(0.0);
    arg.transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    arg.transB    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    arg.orderA    = HIPSPARSE_ORDER_COL;
    arg.idx_baseA = HIPSPARSE_INDEX_BASE_ZERO;
    arg.filename  = "nos3.bin";
    arg.timing    = 0;
    return arg;
}

// CSC format not supported in cusparse
#if(!defined(CUDART_VERSION))
TEST(sddmm_csc_bad_arg, sddmm_csc_float)
{
    testing_sddmm_csc_bad_arg();
}

TEST(sddmm_csc, sddmm_csc_i32_i32_float)
{
    hipsparseStatus_t status
        = testing_sddmm_csc<int32_t, int32_t, float>(setup_sddmm_csc_arguments<float>());
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST(sddmm_csc, sddmm_csc_i32_i32_double)
{
    hipsparseStatus_t status
        = testing_sddmm_csc<int32_t, int32_t, double>(setup_sddmm_csc_arguments<double>());
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST(sddmm_csc, sddmm_csc_i32_i32_hipComplex)
{
    hipsparseStatus_t status
        = testing_sddmm_csc<int32_t, int32_t, hipComplex>(setup_sddmm_csc_arguments<float>());
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}
#endif
