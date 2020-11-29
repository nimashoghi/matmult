/***************************************************************************
Copyright (c) 2016, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

***************************************************************************/
#include "matmult.h"

// --------------------------------------------------------
// function to be accelerated in HW
template <typename T> void mmult_hw(T a[N][N], T b[N][N], T out[N][N]) {
//a is n x k, b is k x m, out is n x m (all squares of size N)
  for (int n = 0; n < N/tileHeight; n++) {
    for (int m = 0; m < N/tileLength; m++) {
      #pragma HLS pipeline
      T acc[tileLength][tileHeight];
      for (int i = 0; i < tileHeight; i++) {
        for (int j = 0; j < tileLength; j++) {
          #pragma HLS unroll
          acc[i][j] = 0;
        }
      }
      for (int k = 0; k < N; ++k){
        T a_buffer[tileHeight];
        for (int p = 0; p < tileHeight; p++) {
          #pragma HLS pipeline
          a_buffer[p] = a[n * tileHeight + p][k];
        }

        for (int t = 0; t < tileLength; t++) {
          #pragma HLS pipeline
          for (int p = 0; p < tileHeight; p++) {
            #pragma HLS unroll
            acc[t][p] += a_buffer[p] * b[k][m * tileLength + t];
          }
        }
      }
      for (int i = 0; i <tileHeight; i++){
        #pragma HLS unroll
        for (int j = 0; j < tileLength; j++) {
          out[n * tileHeight + i][m * tileLength + j] = acc[j][i];
        }
      }
    }
  }
}

template <typename T> void axis2Mat(axis_t *src, T A[N][N], T B[N][N]) {
#pragma HLS inline off
  union {
    int ival;
    T oval;
  } converter;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
#pragma HLS pipeline
#pragma HLS loop_flatten off
      int k = i * N + j;
      converter.ival = src[k].data;
      A[i][j] = converter.oval;
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
#pragma HLS pipeline
#pragma HLS loop_flatten off
      int k = i * N + j;
      converter.ival = src[k + SIZE].data;
      B[i][j] = converter.oval;
    }
  }
}

template <typename T> void Mat2axis(T C[N][N], axis_t *dst) {
#pragma HLS inline off
  union {
    int oval;
    T ival;
  } converter;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
#pragma HLS pipeline
#pragma HLS loop_flatten off
      ap_uint<1> tmp = 0;
      if ((i == N - 1) && (j == N - 1)) {
        tmp = 1;
      }
      dst[i * N + j].last = tmp;
      converter.ival = C[i][j];
      dst[i * N + j].data = converter.oval;
    }
  }
}

extern "C" {
void matmult_accel(axis_t *src, axis_t *dst) {

#pragma HLS INTERFACE axis port = src
#pragma HLS INTERFACE axis port = dst
#pragma HLS INTERFACE s_axilite port = return

#pragma HLS dataflow

  float A[N][N];
  float B[N][N];
  float C[N][N];

  axis2Mat(src, A, B);

  mmult_hw(A, B, C);

  Mat2axis(C, dst);
}
}
