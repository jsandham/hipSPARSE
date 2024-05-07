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

#include "testing_gather.hpp"

#include <hipsparse.h>

Arguments setup_gather_arguments()
{
    Arguments arg;
    arg.N        = 15332;
    arg.nnz      = 500;
    arg.idx_baseA= HIPSPARSE_INDEX_BASE_ZERO;
    arg.timing   = 0;
    return arg;
}

// Only run tests for CUDA 11.1 or greater
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11010)
TEST(gather_bad_arg, gather_float)
{
    testing_gather_bad_arg();
}

TEST(gather, gather_i32_float)
{
    hipsparseStatus_t status = testing_gather<int32_t, float>(setup_gather_arguments());
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST(gather, gather_i64_double)
{
    hipsparseStatus_t status = testing_gather<int64_t, double>(setup_gather_arguments());
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}
#endif
