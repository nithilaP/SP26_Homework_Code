#include <iostream>

using namespace std;

__global__ void kernel_call(float *c) {

    __shared__ float buffer[12 * 1024];

    float *s_c = buffer;
    float *s_a = buffer + 4096;
    float *s_b = buffer + 4096 * 2;

    // 1 threadblock only
    int id = threadIdx.x;
    int p = blockDim.x;

    /**** Do Not Change Code Above This ****/

    /* Notes: Assume A, B, C in shared memory. Limited to only 1 thread block but can change number of threads. */
    /* STAFF STARTER CODE.
    for (int i = id; i < 64; i += p) {
        for (int j = id; j < 64; j += p) {
            s_c[i * 64 + j] = i * 64 + j + 1.0;
            s_a[i * 64 + j] = i * 64 + j + 2.0;
            s_b[i * 64 + j] = i * 64 + j + 1.0;
        }
    }
    */

    /* MY CHANGE:
    -> Save computation by storing value each iteration */
    for (int i = id; i < 64; i += p) {
        for (int j = id; j < 64; j += p) {
            int base = i * 64 + j; /* save computation by storing*/
            s_c[base] = base + 1.0;
            s_a[base] = base + 2.0;
            s_b[base] = base + 1.0;
        }
    }

    // ensure all threads are done initializing the buffer
    __syncthreads();

    /* STAFF STARTER CODE.
    // Computes C += A * B using only 1 thread
    // A is column major order, the other 2 matrices are row major order
    for (int i = 0; i < 64; ++i) {         // 64 rows of C
        for (int j = 0; j < 64; ++j) {     // 64 columns of C
            for (int p = 0; p < 64; ++p) { // 64 columns of A
                s_c[i * 64 + j] += s_a[p * 64 + i] * s_b[p * 64 + j];
            }
        }
    }
    */
    
    // Computes C += A * B 
    /* MY CHANGE: 
    -> COMPUTE in PARALLEL based on elements in C
    -> each thread would just compute C[id], C[id + p], C[id + 2 * p], etc */
    int total_C_elem = 64 * 64;
    for (int c_i = id; c_i < total_C_elem; c_i += p) {

        /* calc i & j based on c_i */
        int i = c_i / 64; /* iterate through elements in the row */
        int j = c_i % 64; /* iterate through elements in the column */
        int curr_c_i = i * 64 + j;

        /* local sum: each thread do this own indp element calcs. */
        float elem_sum = s_c[curr_c_i];
        for (int a_col = 0; a_col < 64; ++a_col) { // 64 columns of A
            elem_sum += s_a[a_col * 64 + i] * s_b[a_col * 64 + j];
        }
        s_c[curr_c_i] = elem_sum;
    }
    __syncthreads(); /* do before copy out*/
    
    /**** Do Not Change Code Below This ****/

    // copy C out such that C is in row major order
    for (int i = id; i < 64 * 64; i += p) {
        c[i] = s_c[i];
    }
}

int main() {

    float *host_out;
    float *dev_out;

    cudaEvent_t st1, et1, st2, et2;
    cudaEventCreate(&st1);
    cudaEventCreate(&et1);
    cudaEventCreate(&st2);
    cudaEventCreate(&et2);

    float ms1, ms2;

    // create buffer on host
    host_out = (float *)malloc(64 * 64 * sizeof(float));

    // create buffer on device
    cudaError_t err = cudaMalloc(&dev_out, 64 * 64 * sizeof(float));
    if (err != cudaSuccess) {
        cout << "Dev Memory not allocated" << endl;
        exit(-1);
    }

    // record time at start
    cudaEventRecord(st2);

    // change number of threads here
    kernel_call<<<1, 128>>>(dev_out); /* launch 256 threads here. */

    // wait until kernel is done start timing
    cudaDeviceSynchronize(); /* kernel sync */
    cudaEventRecord(et2);
    cudaEventSynchronize(et2); /* need to do this before comuting elapsed time when kernel call and GPU */
    // /* ^ need to sync GPU and wait and it has reached et2 in order to get correct measurement*/

    cudaEventElapsedTime(&ms2, st2, et2);
    cout << "Kernel:\t\t\t" << ms2 << "ms" << endl;

    cudaMemcpy(host_out, dev_out, sizeof(float) * 64 * 64,
               cudaMemcpyDeviceToHost);

    free(host_out);
    cudaFree(dev_out);

    return 0;
}
