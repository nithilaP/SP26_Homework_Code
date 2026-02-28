#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

__global__ void kernel_call(int N, float *in, float *out) {
    __shared__ float share_buf[64 * 64];

    // DO NOT CHANGE ANY CODE ABOVE THIS COMMENT

    int id = threadIdx.x; // get my id;

    /* STAFF STARTER CODE
    read 2 rows at a time, write 2 columns at a time
    for (int i = 0; i != 64 * 64 / blockDim.x; ++i) {
        share_buf[i * 2 + (id % 64) * 64 + (id / 64)] = in[blockDim.x * i + id];
    }
    */

    /* MY CHANGE: 
    -> From lecture GPU Memory, Transpose Exercise
    -> read input from coalesced memory (buffer[row][col] = row * 64 + col)
    -> Put into shared buffer, but transpose it as we put it in.
    -> if we store tranposed row & col in shared buffer, cause bank conflicts within warp
    -> SO need intermediate layout: can stagger inputs: 
        -> In slides, we stagger inputs by shifting in each row such that no two values in one 
        column previous stay in a colum. They are now staggered. Columns can be put back together 
        in diagonal. 
    
    NOTES: 
    -> in: 64 x 64 (row major)
    -> out: 64 x 64 */
    for (int i = 0; i != 64 * 64 / blockDim.x; ++i) {

        /* shared memory location but made flat.*/
        int seq_access = blockDim.x * i + id; /* map each to a val from 0 to 4096 */
        // cout << "seq_access:" << seq_access;

        /* determine transpose position */
        int transpose_row = seq_access % 64; /* the column value is the new row */
        int transpose_col = seq_access / 64; /* the row value is the new col */

        /* need to stagger the value, so no shared bank conflict */
        /* to shift over one place in matrix, mult by row; 
        -> mod 64 for wrap around when we shift by 1 */
        int stagger_offset = seq_access % 64; /* shift by row val so that we can wrap around w one shift for each row*/
        int stagger_col = ((transpose_col + stagger_offset) % 64);

        share_buf[transpose_row * 64 + stagger_col] = in[seq_access]; /* put the curr value into shared buffer*/
    }

    __syncthreads(); // wait till everyone is done

    /* STAFF STARTER CODE
    // copy everything to main memory
    for (int i = 0; i != 64 * 64 / blockDim.x; ++i) {
        out[id + blockDim.x * i] = share_buf[id + blockDim.x * i];
    }
    */

    /* MY CHANGES: revert shared memory changes from when we store earlier. */
    // copy everything to main memory
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

    size_t B = 1;
    size_t N = 64;

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

    cudaEventRecord(st2);
    kernel_call<<<1, 128>>>(N, dev_in, dev_out);
    cudaEventRecord(et2);

    // host waits until et2 has occured
    cudaEventSynchronize(et2);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, st2, et2);

    cout << "Kernel time: " << milliseconds << "ms" << endl;

    // copy data out
    cudaMemcpy(host_out, dev_out, B * B * N * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int ii = 0; ii < B; ++ii) {
        for (int jj = 0; jj < B; ++jj) {
            for (int i = 0; i != N; ++i) {
                for (int j = 0; j != N; ++j) {
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
