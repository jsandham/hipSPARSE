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

#include "testing_axpby.hpp"
#include <hipsparse.h>

#include "hipsparse_data.hpp"

// typedef hipsparseIndexBase_t               base;
// typedef std::tuple<int, int, double, base> axpby_tuple;

// int axpby_N_range[]   = {22031};
// int axpby_nnz_range[] = {0, 5, 1000, 10000};

// std::vector<double> axpby_alpha_range = {1.0, 0.0};

// base axpby_idx_base_range[] = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};

// class parameterized_axpby : public testing::TestWithParam<axpby_tuple>
// {
// protected:
//     parameterized_axpby() {}
//     virtual ~parameterized_axpby() {}
//     virtual void SetUp() {}
//     virtual void TearDown() {}
// };

// Arguments setup_axpby_arguments(axpby_tuple tup)
// {
//     Arguments arg;
//     arg.N      = std::get<0>(tup);
//     arg.nnz    = std::get<1>(tup);
//     arg.alpha  = std::get<2>(tup);
//     arg.baseA  = std::get<3>(tup);
//     arg.timing = 0;
//     return arg;
// }

// #if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
// TEST(axpby_bad_arg, axpby_float)
// {
//     testing_axpby_bad_arg();
// }

// TEST_P(parameterized_axpby, axpby_i32_float)
// {
//     Arguments arg = setup_axpby_arguments(GetParam());

//     hipsparseStatus_t status = testing_axpby<int32_t, float>(arg);
//     EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
// }

// TEST_P(parameterized_axpby, axpby_i64_double)
// {
//     Arguments arg = setup_axpby_arguments(GetParam());

//     hipsparseStatus_t status = testing_axpby<int64_t, double>(arg);
//     EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
// }

// TEST_P(parameterized_axpby, axpby_i32_float_complex)
// {
//     Arguments arg = setup_axpby_arguments(GetParam());

//     hipsparseStatus_t status = testing_axpby<int32_t, hipComplex>(arg);
//     EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
// }

// TEST_P(parameterized_axpby, axpby_i64_double_complex)
// {
//     Arguments arg = setup_axpby_arguments(GetParam());

//     hipsparseStatus_t status = testing_axpby<int64_t, hipDoubleComplex>(arg);
//     EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
// }

// INSTANTIATE_TEST_SUITE_P(axpby,
//                          parameterized_axpby,
//                          testing::Combine(testing::ValuesIn(axpby_N_range),
//                                           testing::ValuesIn(axpby_nnz_range),
//                                           testing::ValuesIn(axpby_alpha_range),
//                                           testing::ValuesIn(axpby_idx_base_range)));

// #endif







class axpby : public testing::TestWithParam<Arguments>
{
protected:
    axpby() {}
    virtual ~axpby() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};


TEST_P(axpby, level1)
{
    hipsparseStatus_t status = testing_axpby<int32_t, float>(GetParam());
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

















// template <typename T>
// inline void rocsparse_test_name_suffix_generator_print(std::ostream& s, T item)
// {
//     s << item;
// }

// template <typename T>
// inline void rocsparse_test_name_suffix_generator_remain(std::ostream& s, T item)
// {
//     rocsparse_test_name_suffix_generator_print(s << "_", item);
// }

// inline void rocsparse_test_name_suffix_generator_remain(std::ostream& s) {}
// template <typename T, typename... R>
// inline void rocsparse_test_name_suffix_generator_remain(std::ostream& s, T item, R... remains)
// {
//     rocsparse_test_name_suffix_generator_print(s << "_", item);
//     rocsparse_test_name_suffix_generator_remain(s, remains...);
// }


// template <typename T, typename... R>
// inline void rocsparse_test_name_suffix_generator(std::ostream& s, T item, R... remains)
// {
//     rocsparse_test_name_suffix_generator_print(s, item);
//     rocsparse_test_name_suffix_generator_remain(s, remains...);
// }

// static std::string name_suffix(const Arguments& arg)
// {
//     std::ostringstream s;
//     rocsparse_test_name_suffix_generator(s, __VA_ARGS__);
//     return s.str();
// }



class HipSPARSE_Test : public testing::TestWithParam<Arguments>
{
public:
    // Wrapper functor class which calls name_suffix()
    struct PrintToStringParamName
    {
        std::string operator()(const testing::TestParamInfo<Arguments>& info) const
        {
            static int i = 0;
            return "My_test_name_goes_here_" + std::to_string(i++);//name_suffix(info.param);
        }
    };
};




// INSTANTIATE_TEST_SUITE_P(
//     quick,
//     axpby,
//     testing::ValuesIn(HipSPARSE_TestData::begin([](const Arguments& arg) {return true/*!strcmp(arg.category, "quick")*/;}),
//                       HipSPARSE_TestData::end()),
//                       testclass::PrintToStringParamName());
INSTANTIATE_TEST_SUITE_P(
    quick,
    axpby,
    testing::ValuesIn(HipSPARSE_TestData::begin(), HipSPARSE_TestData::end()), HipSPARSE_Test::PrintToStringParamName());




