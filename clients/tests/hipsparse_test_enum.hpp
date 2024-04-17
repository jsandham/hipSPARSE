/*! \file */
/* ************************************************************************
* Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

template <std::size_t N, typename T>
static constexpr std::size_t countof2(T (&)[N])
{
    return N;
}

// clang-format off
#define HIPSPARSE_FOREACH_TEST_ENUM		\
    TRANSFORM_HIPSPARSE_TEST_ENUM(axpby)
// clang-format on

struct hipsparse_test_enum
{
private:
public:
    /////
#define TRANSFORM_HIPSPARSE_TEST_ENUM(x_) x_,
    typedef enum _ : int32_t
    {
        HIPSPARSE_FOREACH_TEST_ENUM
    } value_type;
    static constexpr value_type all_test_enum[] = {HIPSPARSE_FOREACH_TEST_ENUM};
#undef TRANSFORM_HIPSPARSE_TEST_ENUM
    /////
    static constexpr std::size_t num_test_enum = countof2(all_test_enum);
    value_type                   value{};

private:
    /////
#define TRANSFORM_HIPSPARSE_TEST_ENUM(x_) #x_,
    static constexpr const char* s_test_enum_names[num_test_enum]{HIPSPARSE_FOREACH_TEST_ENUM};
#undef TRANSFORM_HIPSPARSE_TEST_ENUM
    /////

public:
    static inline const char* to_string(hipsparse_test_enum::value_type value)
    {
        return s_test_enum_names[value];
    }
};
