#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>

using namespace std;

__global__ void kernel_call(int N, float *in, float *out) {
    __shared__ float share_buf[64 * 64];

    // DO NOT CHANGE ANY CODE ABOVE THIS COMMENT

    int id = threadIdx.x; // get my id;

    for (int i = 0; i != 64 * 64 / blockDim.x; ++i) {

        /* shared memory location but made flat.*/
        int seq_access = blockDim.x * i + id; /* map each to a val from 0 to 4096 */

        /* determine transpose position */
        int transpose_row = seq_access % 64; /* the column value is the new row */
        int transpose_col = seq_access / 64; /* the row value is the new col */

        int stagger_offset = seq_access % 64; /* shift by row val so that we can wrap around w one shift for each row*/
        int stagger_col = ((transpose_col + stagger_offset) % 64);

        share_buf[transpose_row * 64 + stagger_col] = in[seq_access]; /* put the curr value into shared buffer*/
    }

    __syncthreads(); // wait till everyone is done

    for (int i = 0; i != 64 * 64 / blockDim.x; ++i) {
        int seq_access = blockDim.x * i + id; /* map each to a val from 0 to 4096 */

        /* determine transpose position in the shared buffer */
        int transpose_row = seq_access / 64; 
        int transpose_col = seq_access % 64;

        int stagger_offset = seq_access / 64; /* shift by row val so that we can wrap around w one shift for each row*/
        int stagger_col = ((transpose_col + stagger_offset) % 64);

        out[seq_access] = share_buf[transpose_row * 64 + stagger_col];
    }

}

int main() {
    float *host_in, *host_out;
    float *dev_in, *dev_out;

    /* USER SET: Set B and N here. */
    size_t matrix_side = 256; // size of matrix side (square)
    size_t N = 64; // tile size (for one)

    /*------ Set based on user input ----*/
    size_t B = int(matrix_side / N); // tiles on each side

    // create buffer on host
    host_in = (float *)malloc(B * B * N * N * sizeof(float));
    host_out = (float *)malloc(B * B * N * N * sizeof(float));

    // creates a matrix stored in row major order
    for (int ii = 0; ii < B; ++ii) {
        for (int jj = 0; jj < B; ++jj) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    host_in[(ii * B + jj) * N * N + i * N + j] = i * N + j;
                }
            }
        }
    }

    // create buffer on device
    cudaError_t err = cudaMalloc(&dev_in, B * B * N * N * sizeof(float));
    if (err != cudaSuccess) {
        cout << "Dev Memory not allocated" << endl;
        exit(-1);
    }

    err = cudaMalloc(&dev_out, B * B * N * N * sizeof(float));
    if (err != cudaSuccess) {
        cout << "Dev Memory not allocated" << endl;
        exit(-1);
    }

    cudaMemcpy(dev_in, host_in, B * B * N * N * sizeof(float),
               cudaMemcpyHostToDevice);

    // create GPU timing events for timing the GPU
    cudaEvent_t st2, et2;
    cudaEventCreate(&st2);
    cudaEventCreate(&et2);

    float curr_total_milliseconds = 0.0; 
    float curr_min_milliseconds = FLT_MAX;
    for (int i = 0; i < 1000; i++){

        cudaEventRecord(st2);    

        // STAFF STARTER: kernel_call<<<1, 128>>>(N, dev_in, dev_out);
        /* MY CHANGE: multi kernel launch */
        int num_elem_in_tile = N * N; // FIX
        for (int ii = 0; ii < B; ++ii) {
            for (int jj = 0; jj < B; ++jj) {
                /* map element to transpose elem & add offset to the dev_in and dev_out */
                kernel_call<<<1, 128>>>(N, dev_in + num_elem_in_tile * (ii * B + jj), dev_out+ num_elem_in_tile * (jj * B + ii));

            }
        }

        cudaEventRecord(et2);

        // host waits until et2 has occured
        cudaEventSynchronize(et2);
        cudaDeviceSynchronize();  /* avoid gpu optimziation betwen iter*/

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, st2, et2);
    
        curr_total_milliseconds += milliseconds;
        if(milliseconds < curr_min_milliseconds){
            curr_min_milliseconds = milliseconds;
        }

    }
    cout << "Average Kernel time: " << (curr_total_milliseconds / 1000) << "ms" << endl;
    cout << "Minimum Kernel time: " << curr_min_milliseconds << "ms" << endl;

    // cout << "Kernel time: " << milliseconds << "ms" << endl;
    // cout << "Avg kernel time: " << (milliseconds / 10000.0f) << "ms" << endl;

    // copy data out
    cudaMemcpy(host_out, dev_out, B * B * N * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int ii = 0; ii < B; ++ii) {
        for (int jj = 0; jj < B; ++jj) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    correct &= (host_out[(ii * B + jj) * N * N + i * N + j] ==
                                host_in[(jj * B + ii) * N * N + j * N + i]);
                }
            }
        }
    }
    cout << (correct ? "Yes" : "No") << endl;

    cudaEventDestroy(st2);
    cudaEventDestroy(et2);

    free(host_in);
    free(host_out);
    cudaFree(dev_in);
    cudaFree(dev_out);

    return 0;
}
