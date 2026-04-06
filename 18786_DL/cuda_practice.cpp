#include <omp.h>
#include <iostream>
/** Big Idea: Use Streams to Do the operations for each buffer of data (allocate multiple buffers).
 * 
 */
#define LAYERS 10
#define NUM_STREAMS 4
using namespace std;

__device__ float warpReduce(float init_val) {
  float value = init_val;
  for (int i=16; i>=1; i/=2)
    value += __shfl_xor_sync(0xffffffff, value, i, 32);
  return value;
}

__global__ void gpu_kernel(float *in_buff, float *weights, float *out_cat)
{
  __shared__ float buffer[256];
  int id = threadIdx.x;
  buffer[id] = in_buff[id];
  __syncthreads();
  
  float tmp = 0.0;
  for (int layer = 0; layer < LAYERS; ++layer)
  {
    for (int i = 0; i < 256; ++i)
    {
      tmp += buffer[i] * weights[layer*256*256 + id*256 + i];
    }
    __syncthreads();
    buffer[id] = tmp;
    __syncthreads();
  }
  
  float min_result = warpReduce(buffer[id]);
  if (id % 32 == 0)
    atomicAdd(out_cat, min_result);
}

int main(int argc, char *argv[])
{
  float *weight, *dev_weight;
  double st, et;
  if (argc <= 1) exit(-1);
  int runlen = atoi(argv[1]);
  if (!runlen) exit(-1);

  // ── per-stream buffers (replacing single recv_buffer, dev_buff, dev_cat) ──
  float *recv_buffer[NUM_STREAMS];  // pinned host input buffers
  float *result[NUM_STREAMS];       // pinned host output buffers
  float *dev_buff[NUM_STREAMS];     // device input buffers
  float *dev_cat[NUM_STREAMS];      // device output buffers
  cudaStream_t streams[NUM_STREAMS];

  // flags: 0 = slot free, 1 = buffer ready for dispatch
  volatile int buffer_ready[NUM_STREAMS];
  // flags: 0 = not dispatched yet, 1 = dispatched, ready to receive
  volatile int dispatch_done[NUM_STREAMS];

  // initialize weights (unchanged from starter)
  weight = (float*)malloc(sizeof(float)*256*256*LAYERS);
  for (int i = 0; i < LAYERS; ++i)
    for (int j = 0; j < 256*256; ++j)
      weight[i*256*256 + j] = (i + 1.0)*1e-5;

  // allocate device weights (unchanged from starter)
  cudaMalloc(&dev_weight, sizeof(float)*256*256*LAYERS);
  cudaMemcpy(dev_weight, weight, sizeof(float)*256*256*LAYERS,
             cudaMemcpyHostToDevice);

  // allocate per-stream buffers and create streams
  for (int i = 0; i < NUM_STREAMS; i++) {
    // pinned memory required for cudaMemcpyAsync to actually be async
    cudaMallocHost(&recv_buffer[i], sizeof(float)*256);
    cudaMallocHost(&result[i], sizeof(float));
    cudaMalloc(&dev_buff[i], sizeof(float)*256);
    cudaMalloc(&dev_cat[i], sizeof(float));
    // NonBlocking so streams dont serialize against default stream
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    buffer_ready[i] = 0;
    dispatch_done[i] = 0;
  }

  // ── create any streams or parallelization data structure above this line ──
  st = omp_get_wtime();

  #pragma omp parallel num_threads(3)
  {
    int tid = omp_get_thread_num();

    // ── THREAD 0: sensor reads (must stay sequential) ──────────────
    if (tid == 0) {
      for (int run = 0; run < runlen; run++) {
        int s = run % NUM_STREAMS;

        // spin until Thread B has consumed this slot from last cycle
        // prevents overwriting a buffer Thread B hasn't dispatched yet
        while (buffer_ready[s] != 0);

        // simulate receiving input from sensor — sequential, one at a time
        for (int i = 0; i < 256; ++i)
          recv_buffer[s][i] = (run + i) * 1e-3;

        // signal Thread B that buffer[s] is filled and ready
        buffer_ready[s] = 1;
      }
    }

    // ── THREAD 1: GPU dispatch ──────────────────────────────────────
    else if (tid == 1) {
      for (int run = 0; run < runlen; run++) {
        int s = run % NUM_STREAMS;

        // spin until Thread A fills buffer[s]
        while (buffer_ready[s] != 1);

        // reset device output for this slot before kernel writes to it
        // must be on same stream so it runs before the kernel
        cudaMemsetAsync(dev_cat[s], 0, sizeof(float), streams[s]);

        // async H2D — returns immediately, GPU queues the copy on stream s
        cudaMemcpyAsync(dev_buff[s], recv_buffer[s],
                        sizeof(float)*256,
                        cudaMemcpyHostToDevice, streams[s]);

        // kernel launch — queued on stream s after H2D copy
        gpu_kernel<<<1, 256, 0, streams[s]>>>(
            dev_buff[s], dev_weight, dev_cat[s]);

        // async D2H — queued on stream s after kernel
        cudaMemcpyAsync(result[s], dev_cat[s],
                        sizeof(float),
                        cudaMemcpyDeviceToHost, streams[s]);

        // slot is consumed — signal Thread A it can refill buffer[s]
        buffer_ready[s] = 0;

        // signal Thread C that stream[s] has been fully dispatched
        dispatch_done[s] = 1;
      }
    }

    // ── THREAD 2: receive results ───────────────────────────────────
    else if (tid == 2) {
      for (int run = 0; run < runlen; run++) {
        int s = run % NUM_STREAMS;

        // spin until Thread B has dispatched stream[s]
        while (dispatch_done[s] != 1);

        // wait for all GPU ops on stream[s] to complete
        // (cudaMemsetAsync + H2D + kernel + D2H all finish here)
        cudaStreamSynchronize(streams[s]);

        // result[s] is now valid on host — print in order
        printf("%d %e\n", run, *result[s]);

        // reset flag so Thread B can reuse this slot next cycle
        dispatch_done[s] = 0;
      }
    }
  }
  // all 3 threads rejoin here
  cudaDeviceSynchronize();

  et = omp_get_wtime();

  // ── cleanup everything we introduced ───────────────────────────────
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
    cudaFreeHost(recv_buffer[i]);
    cudaFreeHost(result[i]);
    cudaFree(dev_buff[i]);
    cudaFree(dev_cat[i]);
  }

  // ── do not change the code below this line ─────────────────────────
  cout<<(et-st)<<" seconds for "<<runlen<<" runs"<<endl;
  cudaFree(dev_weight);
  free(weight);
}