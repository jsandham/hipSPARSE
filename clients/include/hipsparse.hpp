/* ************************************************************************
* Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
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
#ifndef _HIPSPARSE_HPP_
#define _HIPSPARSE_HPP_

#include <hipsparse.h>

namespace hipsparse
{

    template <typename T>
    hipsparseStatus_t hipsparseXaxpyi(hipsparseHandle_t    handle,
                                      int                  nnz,
                                      const T*             alpha,
                                      const T*             x_val,
                                      const int*           x_ind,
                                      T*                   y,
                                      hipsparseIndexBase_t idx_base);

    template <typename T>
    hipsparseStatus_t hipsparseXdoti(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     const T*             x_val,
                                     const int*           x_ind,
                                     const T*             y,
                                     T*                   result,
                                     hipsparseIndexBase_t idx_base);

    template <typename T>
    hipsparseStatus_t hipsparseXdotci(hipsparseHandle_t    handle,
                                      int                  nnz,
                                      const T*             x_val,
                                      const int*           x_ind,
                                      const T*             y,
                                      T*                   result,
                                      hipsparseIndexBase_t idx_base);

    template <typename T>
    hipsparseStatus_t hipsparseXgthr(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     const T*             y,
                                     T*                   x_val,
                                     const int*           x_ind,
                                     hipsparseIndexBase_t idx_base);

    template <typename T>
    hipsparseStatus_t hipsparseXgthrz(hipsparseHandle_t    handle,
                                      int                  nnz,
                                      T*                   y,
                                      T*                   x_val,
                                      const int*           x_ind,
                                      hipsparseIndexBase_t idx_base);

    template <typename T>
    hipsparseStatus_t hipsparseXroti(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     T*                   x_val,
                                     const int*           x_ind,
                                     T*                   y,
                                     const T*             c,
                                     const T*             s,
                                     hipsparseIndexBase_t idx_base);

    template <typename T>
    hipsparseStatus_t hipsparseXsctr(hipsparseHandle_t    handle,
                                     int                  nnz,
                                     const T*             x_val,
                                     const int*           x_ind,
                                     T*                   y,
                                     hipsparseIndexBase_t idx_base);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrmv(hipsparseHandle_t         handle,
                                      hipsparseOperation_t      trans,
                                      int                       m,
                                      int                       n,
                                      int                       nnz,
                                      const T*                  alpha,
                                      const hipsparseMatDescr_t descr,
                                      const T*                  csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      const T*                  x,
                                      const T*                  beta,
                                      T*                        y);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                                  hipsparseOperation_t      transA,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  T*                        csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csrsv2Info_t              info,
                                                  int*                      pBufferSizeInBytes);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                     hipsparseOperation_t      transA,
                                                     int                       m,
                                                     int                       nnz,
                                                     const hipsparseMatDescr_t descrA,
                                                     T*                        csrSortedValA,
                                                     const int*                csrSortedRowPtrA,
                                                     const int*                csrSortedColIndA,
                                                     csrsv2Info_t              info,
                                                     size_t*                   pBufferSize);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrsv2_analysis(hipsparseHandle_t         handle,
                                                hipsparseOperation_t      transA,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                const T*                  csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrsv2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrsv2_solve(hipsparseHandle_t         handle,
                                             hipsparseOperation_t      transA,
                                             int                       m,
                                             int                       nnz,
                                             const T*                  alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const T*                  csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csrsv2Info_t              info,
                                             const T*                  f,
                                             T*                        x,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXhybmv(hipsparseHandle_t         handle,
                                      hipsparseOperation_t      trans,
                                      const T*                  alpha,
                                      const hipsparseMatDescr_t descr,
                                      const hipsparseHybMat_t   hyb,
                                      const T*                  x,
                                      const T*                  beta,
                                      T*                        y);

    template <typename T>
    hipsparseStatus_t hipsparseXbsrmv(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      hipsparseOperation_t      transA,
                                      int                       mb,
                                      int                       nb,
                                      int                       nnzb,
                                      const T*                  alpha,
                                      const hipsparseMatDescr_t descrA,
                                      const T*                  bsrSortedValA,
                                      const int*                bsrSortedRowPtrA,
                                      const int*                bsrSortedColIndA,
                                      int                       blockDim,
                                      const T*                  x,
                                      const T*                  beta,
                                      T*                        y);

    template <typename T>
    hipsparseStatus_t hipsparseXbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                                  hipsparseDirection_t      dir,
                                                  hipsparseOperation_t      transA,
                                                  int                       mb,
                                                  int                       nnzb,
                                                  const hipsparseMatDescr_t descrA,
                                                  T*                        bsrSortedValA,
                                                  const int*                bsrSortedRowPtrA,
                                                  const int*                bsrSortedColIndA,
                                                  int                       blockDim,
                                                  bsrsv2Info_t              info,
                                                  int*                      pBufferSizeInBytes);

    template <typename T>
    hipsparseStatus_t hipsparseXbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                     hipsparseDirection_t      dir,
                                                     hipsparseOperation_t      transA,
                                                     int                       mb,
                                                     int                       nnzb,
                                                     const hipsparseMatDescr_t descrA,
                                                     T*                        bsrSortedValA,
                                                     const int*                bsrSortedRowPtrA,
                                                     const int*                bsrSortedColIndA,
                                                     int                       blockDim,
                                                     bsrsv2Info_t              info,
                                                     size_t*                   pBufferSize);

    template <typename T>
    hipsparseStatus_t hipsparseXbsrsv2_analysis(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dir,
                                                hipsparseOperation_t      transA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                const T*                  bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrsv2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXbsrsv2_solve(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dir,
                                             hipsparseOperation_t      transA,
                                             int                       mb,
                                             int                       nnzb,
                                             const T*                  alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const T*                  bsrSortedValA,
                                             const int*                bsrSortedRowPtrA,
                                             const int*                bsrSortedColIndA,
                                             int                       blockDim,
                                             bsrsv2Info_t              info,
                                             const T*                  f,
                                             T*                        x,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXbsrmm(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      hipsparseOperation_t      transA,
                                      hipsparseOperation_t      transB,
                                      int                       mb,
                                      int                       n,
                                      int                       kb,
                                      int                       nnzb,
                                      const T*                  alpha,
                                      const hipsparseMatDescr_t descr,
                                      const T*                  bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       blockDim,
                                      const T*                  B,
                                      int                       ldb,
                                      const T*                  beta,
                                      T*                        C,
                                      int                       ldc);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrmm2(hipsparseHandle_t         handle,
                                       hipsparseOperation_t      transA,
                                       hipsparseOperation_t      transB,
                                       int                       m,
                                       int                       n,
                                       int                       k,
                                       int                       nnz,
                                       const T*                  alpha,
                                       const hipsparseMatDescr_t descr,
                                       const T*                  csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       const T*                  B,
                                       int                       ldb,
                                       const T*                  beta,
                                       T*                        C,
                                       int                       ldc);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                     int                       algo,
                                                     hipsparseOperation_t      transA,
                                                     hipsparseOperation_t      transB,
                                                     int                       m,
                                                     int                       nrhs,
                                                     int                       nnz,
                                                     const T*                  alpha,
                                                     const hipsparseMatDescr_t descrA,
                                                     const T*                  csrSortedValA,
                                                     const int*                csrSortedRowPtrA,
                                                     const int*                csrSortedColIndA,
                                                     const T*                  B,
                                                     int                       ldb,
                                                     csrsm2Info_t              info,
                                                     hipsparseSolvePolicy_t    policy,
                                                     size_t*                   pBufferSize);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrsm2_analysis(hipsparseHandle_t         handle,
                                                int                       algo,
                                                hipsparseOperation_t      transA,
                                                hipsparseOperation_t      transB,
                                                int                       m,
                                                int                       nrhs,
                                                int                       nnz,
                                                const T*                  alpha,
                                                const hipsparseMatDescr_t descrA,
                                                const T*                  csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                const T*                  B,
                                                int                       ldb,
                                                csrsm2Info_t              info,
                                                hipsparseSolvePolicy_t    policy,
                                                void*                     pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrsm2_solve(hipsparseHandle_t         handle,
                                             int                       algo,
                                             hipsparseOperation_t      transA,
                                             hipsparseOperation_t      transB,
                                             int                       m,
                                             int                       nrhs,
                                             int                       nnz,
                                             const T*                  alpha,
                                             const hipsparseMatDescr_t descrA,
                                             const T*                  csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             T*                        B,
                                             int                       ldb,
                                             csrsm2Info_t              info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXgemmi(hipsparseHandle_t handle,
                                      int               m,
                                      int               n,
                                      int               k,
                                      int               nnz,
                                      const T*          alpha,
                                      const T*          A,
                                      int               lda,
                                      const T*          cscValB,
                                      const int*        cscColPtrB,
                                      const int*        cscRowIndB,
                                      const T*          beta,
                                      T*                C,
                                      int               ldc);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrgeam(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const T*                  alpha,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const T*                  csrValA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        const T*                  beta,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const T*                  csrValB,
                                        const int*                csrRowPtrB,
                                        const int*                csrColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        T*                        csrValC,
                                        int*                      csrRowPtrC,
                                        int*                      csrColIndC);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const T*                  alpha,
                                                       const hipsparseMatDescr_t descrA,
                                                       int                       nnzA,
                                                       const T*                  csrValA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const T*                  beta,
                                                       const hipsparseMatDescr_t descrB,
                                                       int                       nnzB,
                                                       const T*                  csrValB,
                                                       const int*                csrRowPtrB,
                                                       const int*                csrColIndB,
                                                       const hipsparseMatDescr_t descrC,
                                                       const T*                  csrValC,
                                                       const int*                csrRowPtrC,
                                                       const int*                csrColIndC,
                                                       size_t* pBufferSizeInBytes);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrgeam2(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         const T*                  alpha,
                                         const hipsparseMatDescr_t descrA,
                                         int                       nnzA,
                                         const T*                  csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const T*                  beta,
                                         const hipsparseMatDescr_t descrB,
                                         int                       nnzB,
                                         const T*                  csrValB,
                                         const int*                csrRowPtrB,
                                         const int*                csrColIndB,
                                         const hipsparseMatDescr_t descrC,
                                         T*                        csrValC,
                                         int*                      csrRowPtrC,
                                         int*                      csrColIndC,
                                         void*                     pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrgemm(hipsparseHandle_t         handle,
                                        hipsparseOperation_t      transA,
                                        hipsparseOperation_t      transB,
                                        int                       m,
                                        int                       n,
                                        int                       k,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const T*                  csrValA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const T*                  csrValB,
                                        const int*                csrRowPtrB,
                                        const int*                csrColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        T*                        csrValC,
                                        const int*                csrRowPtrC,
                                        int*                      csrColIndC);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       int                       k,
                                                       const T*                  alpha,
                                                       const hipsparseMatDescr_t descrA,
                                                       int                       nnzA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const hipsparseMatDescr_t descrB,
                                                       int                       nnzB,
                                                       const int*                csrRowPtrB,
                                                       const int*                csrColIndB,
                                                       const T*                  beta,
                                                       const hipsparseMatDescr_t descrD,
                                                       int                       nnzD,
                                                       const int*                csrRowPtrD,
                                                       const int*                csrColIndD,
                                                       csrgemm2Info_t            info,
                                                       size_t* pBufferSizeInBytes);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrgemm2(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       k,
                                         const T*                  alpha,
                                         const hipsparseMatDescr_t descrA,
                                         int                       nnzA,
                                         const T*                  csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const hipsparseMatDescr_t descrB,
                                         int                       nnzB,
                                         const T*                  csrValB,
                                         const int*                csrRowPtrB,
                                         const int*                csrColIndB,
                                         const T*                  beta,
                                         const hipsparseMatDescr_t descrD,
                                         int                       nnzD,
                                         const T*                  csrValD,
                                         const int*                csrRowPtrD,
                                         const int*                csrColIndD,
                                         const hipsparseMatDescr_t descrC,
                                         T*                        csrValC,
                                         const int*                csrRowPtrC,
                                         int*                      csrColIndC,
                                         const csrgemm2Info_t      info,
                                         void*                     pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                    int                       m,
                                                    int                       nnz,
                                                    const hipsparseMatDescr_t descrA,
                                                    T*                        csrSortedValA,
                                                    const int*                csrSortedRowPtrA,
                                                    const int*                csrSortedColIndA,
                                                    csrilu02Info_t            info,
                                                    int*                      pBufferSizeInBytes);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       nnz,
                                                       const hipsparseMatDescr_t descrA,
                                                       T*                        csrSortedValA,
                                                       const int*                csrSortedRowPtrA,
                                                       const int*                csrSortedColIndA,
                                                       csrilu02Info_t            info,
                                                       size_t*                   pBufferSize);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrilu02_analysis(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  const T*                  csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csrilu02Info_t            info,
                                                  hipsparseSolvePolicy_t    policy,
                                                  void*                     pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXcsrilu02(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       nnz,
                                         const hipsparseMatDescr_t descrA,
                                         T*                        csrSortedValA_valM,
                                         /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                         const int*             csrSortedRowPtrA,
                                         const int*             csrSortedColIndA,
                                         csrilu02Info_t         info,
                                         hipsparseSolvePolicy_t policy,
                                         void*                  pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXbsric02_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   T*                        bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       block_dim,
                                                   bsric02Info_t             info,
                                                   int*                      pBufferSizeInBytes);

    template <typename T>
    hipsparseStatus_t hipsparseXbsric02_analysis(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 const T*                  bsrValA,
                                                 const int*                bsrRowPtrA,
                                                 const int*                bsrColIndA,
                                                 int                       block_dim,
                                                 bsric02Info_t             info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 void*                     pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXbsric02(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        T*                        bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       block_dim,
                                        bsric02Info_t             info,
                                        hipsparseSolvePolicy_t    policy,
                                        void*                     pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXcsric02_bufferSize(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   T*                        csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csric02Info_t             info,
                                                   int*                      pBufferSizeInBytes);

    template <typename T>
    hipsparseStatus_t hipsparseXcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       nnz,
                                                      const hipsparseMatDescr_t descrA,
                                                      T*                        csrSortedValA,
                                                      const int*                csrSortedRowPtrA,
                                                      const int*                csrSortedColIndA,
                                                      csric02Info_t             info,
                                                      size_t*                   pBufferSize);

    template <typename T>
    hipsparseStatus_t hipsparseXcsric02_analysis(hipsparseHandle_t         handle,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 const T*                  csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csric02Info_t             info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 void*                     pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXcsric02(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       nnz,
                                        const hipsparseMatDescr_t descrA,
                                        T*                        csrSortedValA_valM,
                                        /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                        const int*             csrSortedRowPtrA,
                                        const int*             csrSortedColIndA,
                                        csric02Info_t          info,
                                        hipsparseSolvePolicy_t policy,
                                        void*                  pBuffer);

    template <typename T>
    hipsparseStatus_t hipsparseXnnz(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const T*                  A,
                                    int                       lda,
                                    int*                      nnzPerRowColumn,
                                    int*                      nnzTotalDevHostPtr);

    template <typename T>
    hipsparseStatus_t hipsparseXnnz_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             const hipsparseMatDescr_t descrA,
                                             const T*                  csrValA,
                                             const int*                csrRowPtrA,
                                             int*                      nnzPerRow,
                                             int*                      nnzC,
                                             T                         tol);

    template <typename T>
    hipsparseStatus_t hipsparseXdense2csr(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const T*                  A,
                                          int                       ld,
                                          const int*                nnzPerRow,
                                          T*                        csrVal,
                                          int*                      csrRowPtr,
                                          int*                      csrColInd);

    template <typename T>
    hipsparseStatus_t hipsparseXpruneDense2csr_bufferSize(hipsparseHandle_t         handle,
                                                          int                       m,
                                                          int                       n,
                                                          const T*                  A,
                                                          int                       lda,
                                                          const T*                  threshold,
                                                          const hipsparseMatDescr_t descr,
                                                          const T*                  csrVal,
                                                          const int*                csrRowPtr,
                                                          const int*                csrColInd,
                                                          size_t*                   bufferSize);

    template <typename T>
    hipsparseStatus_t hipsparseXpruneDense2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                             int                       m,
                                                             int                       n,
                                                             const T*                  A,
                                                             int                       lda,
                                                             const T*                  threshold,
                                                             const hipsparseMatDescr_t descr,
                                                             const T*                  csrVal,
                                                             const int*                csrRowPtr,
                                                             const int*                csrColInd,
                                                             size_t*                   bufferSize);

    template <typename T>
    hipsparseStatus_t hipsparseXpruneDense2csrNnz(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       n,
                                                  const T*                  A,
                                                  int                       lda,
                                                  const T*                  threshold,
                                                  const hipsparseMatDescr_t descr,
                                                  int*                      csrRowPtr,
                                                  int*                      nnzTotalDevHostPtr,
                                                  void*                     buffer);

    template <typename T>
    hipsparseStatus_t hipsparseXpruneDense2csr(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       n,
                                               const T*                  A,
                                               int                       lda,
                                               const T*                  threshold,
                                               const hipsparseMatDescr_t descr,
                                               T*                        csrVal,
                                               const int*                csrRowPtr,
                                               int*                      csrColInd,
                                               void*                     buffer);

    template <typename T>
    hipsparseStatus_t hipsparseXpruneDense2csrByPercentage_bufferSize(hipsparseHandle_t         handle,
                                                                      int                       m,
                                                                      int                       n,
                                                                      const T*                  A,
                                                                      int                       lda,
                                                                      T                         percentage,
                                                                      const hipsparseMatDescr_t descr,
                                                                      const T*                  csrVal,
                                                                      const int*                csrRowPtr,
                                                                      const int*                csrColInd,
                                                                      pruneInfo_t               info,
                                                                      size_t*                   bufferSize);

    template <typename T>
    hipsparseStatus_t hipsparseXpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                                         int                       m,
                                                                         int                       n,
                                                                         const T*                  A,
                                                                         int                       lda,
                                                                         T                         percentage,
                                                                         const hipsparseMatDescr_t descr,
                                                                         const T*                  csrVal,
                                                                         const int*                csrRowPtr,
                                                                         const int*                csrColInd,
                                                                         pruneInfo_t               info,
                                                                         size_t*                   bufferSize);

    template <typename T>
    hipsparseStatus_t hipsparseXpruneDense2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                              int                       m,
                                                              int                       n,
                                                              const T*                  A,
                                                              int                       lda,
                                                              T                         percentage,
                                                              const hipsparseMatDescr_t descr,
                                                              int*                      csrRowPtr,
                                                              int*                      nnzTotalDevHostPtr,
                                                              pruneInfo_t               info,
                                                              void*                     buffer);

    template <typename T>
    hipsparseStatus_t hipsparseXpruneDense2csrByPercentage(hipsparseHandle_t         handle,
                                                           int                       m,
                                                           int                       n,
                                                           const T*                  A,
                                                           int                       lda,
                                                           T                         percentage,
                                                           const hipsparseMatDescr_t descr,
                                                           T*                        csrVal,
                                                           const int*                csrRowPtr,
                                                           int*                      csrColInd,
                                                           pruneInfo_t               info,
                                                           void*                     buffer);

    template <typename T>
    hipsparseStatus_t hipsparseXdense2csc(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const T*                  A,
                                          int                       ld,
                                          const int*                nnzPerColumn,
                                          T*                        cscVal,
                                          int*                      cscRowInd,
                                          int*                      cscColPtr);

    template <typename T>
    hipsparseStatus_t hipsparseXcsr2dense(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const T*                  csrVal,
                                          const int*                csrRowPtr,
                                          const int*                csrColInd,
                                          T*                        A,
                                          int                       ld);

    template <typename T>
    hipsparseStatus_t hipsparseXcsc2dense(hipsparseHandle_t         handle,
                                          int                       m,
                                          int                       n,
                                          const hipsparseMatDescr_t descr,
                                          const T*                  cscVal,
                                          const int*                cscRowInd,
                                          const int*                cscColPtr,
                                          T*                        A,
                                          int                       ld);

    template <typename T>
    hipsparseStatus_t hipsparseXcsr2csc(hipsparseHandle_t    handle,
                                        int                  m,
                                        int                  n,
                                        int                  nnz,
                                        const T*             csr_val,
                                        const int*           csr_row_ptr,
                                        const int*           csr_col_ind,
                                        T*                   csc_val,
                                        int*                 csc_row_ind,
                                        int*                 csc_col_ptr,
                                        hipsparseAction_t    copy_values,
                                        hipsparseIndexBase_t idx_base);

    template <typename T>
    hipsparseStatus_t hipsparseXcsr2hyb(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descr,
                                        const T*                  csr_val,
                                        const int*                csr_row_ptr,
                                        const int*                csr_col_ind,
                                        hipsparseHybMat_t         hyb,
                                        int                       user_ell_width,
                                        hipsparseHybPartition_t   partition_type);

    template <typename T>
    hipsparseStatus_t hipsparseXcsr2bsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descrA,
                                        const T*                  csrValA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        int                       blockDim,
                                        const hipsparseMatDescr_t descrC,
                                        T*                        bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC);

    template <typename T>
    hipsparseStatus_t hipsparseXbsr2csr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        const hipsparseMatDescr_t descrA,
                                        const T*                  bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       blockDim,
                                        const hipsparseMatDescr_t descrC,
                                        T*                        csrValC,
                                        int*                      csrRowPtrC,
                                        int*                      csrColIndC);

    template <typename T>
    hipsparseStatus_t hipsparseXhyb2csr(hipsparseHandle_t         handle,
                                        const hipsparseMatDescr_t descrA,
                                        const hipsparseHybMat_t   hybA,
                                        T*                        csr_val,
                                        int*                      csr_row_ptr,
                                        int*                      csr_col_ind);

    template <typename T>
    hipsparseStatus_t hipsparseXcsr2csr_compress(hipsparseHandle_t         handle,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t descrA,
                                                 const T*                  csrValA,
                                                 const int*                csrColIndA,
                                                 const int*                csrRowPtrA,
                                                 int                       nnzA,
                                                 const int*                nnzPerRow,
                                                 T*                        csrValC,
                                                 int*                      csrColIndC,
                                                 int*                      csrRowPtrC,
                                                 T                         tol);

} // namespace hipsparse

#endif // _HIPSPARSE_HPP_
