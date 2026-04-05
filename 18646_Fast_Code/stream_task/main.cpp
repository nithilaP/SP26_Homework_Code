#include <omp.h>
#include <iostream>

#define LAYERS 10

using namespace std;

/* warp reduce helper: 
* __device__: func only runs on GPU (called by other CPU code)
* sums up all 32 threads in a warp 
* __shfl_xor_sync -> lets threads direclty read each other's registers (no shared memory needed)
* loop runs 5 times (16, 8, 4, 2, 1) -> each round pairds threads w XOR distance i & adds vals together
* after 5 rounds, thread holds sum of all 32 thread values 
* __shfl_xor_sync(0xffffffff, value, i, 32);
* -- 0xffffffff: participation mask -> in binary, this is 32 ones, all 32 threads in warp participate
* -- value: the variable being shared. Each thread shares its own copy of value and receives another thread's copy of value.
* -- i: the laneMask) — XORed with the calling thread's lane ID to determine which thread to read from. So thread 5 with i=4 reads from thread 5 XOR 4 = 1.
* -- 32: width, meaning operate across the full warp of 32 threads */
__device__ float warpReduce(float init_val) {
  // Seed starting value as inverse lane ID
  float value = init_val;

  // Use XOR mode to perform butterfly reduction
  for (int i=16; i>=1; i/=2)
    value += __shfl_xor_sync(0xffffffff, value, i, 32);

  return value;
}

/**
 * __global__: launched no CPU but runs on GPU 
 * Takes in 256 input buffer, weight matix and single output float 
 * 
 * 
 */
__global__ void gpu_kernel(float *in_buff, float *weights, float *out_cat)
{

  /** Allocate shared memory for 256 floats (fast, shared within a block)
   * Each of the 256 threads copies one element from global to shared memory
   * __syncthreads() -> barrier, wait until ALL threads have finished copying
   */
  __shared__ float buffer[256];
  
  int id = threadIdx.x;

  buffer[id] = in_buff[id];
  
  __syncthreads();
  
  /** Simulates 10 layer fully connected network 
   * Each thread computes 1 output neuron -> multiples every input (256 values) by neuron weights and accumulates into tmp.
   * After each layer, writes results back to shared memory for next layer
   */
  //10 layer ANN
  float tmp = 0.0; // temp var stored in register
  for (int layer = 0; layer < LAYERS; ++layer)
    {
      for (int i = 0; i < 256; ++i)
      {
        tmp += buffer[i] * weights[layer*256*256 + id*256 + i];
      }

      /* two __syncthreads() calls to ensure no thread reads buffer while another is writing to it. */
      /* before this, all threads are using old buffer values from pervious layer. */
      __syncthreads(); // Barrier 1
      /* ALL threads done reading → [safe to write] */
      
      buffer[id] = tmp; // writing new values - each thread only writes to its own unique index in buffer, so no race condition

      __syncthreads(); // Barrier 2
      /* ALL threads done writing → [safe to read] */
      /* new safe to start next layer reading buffer. */
    }

  /* after all layers, reduce 256 outputs down to a signle number. 
  *  warpReduce - sums within each group of 32 threads (256/32 = 8 warps)
  *  only thread 0 of each warp (id % 32 == 0) atomically adds warp sum into single output float out_cat
  *  ataomicAdd is needed bc 8 threads (one in each warp) are writing to the same address at the same time */
  
  float min_result = warpReduce(buffer[id]);
  
  if (id % 32 == 0)
    atomicAdd(out_cat, min_result);
    /* outcat -> ptr to global memory - slow, lives on GPU DRAM*/
}


int main(int argc, char *argv[])
{

  /** Define Variables: 
   * recv_buffer — CPU-side buffer that simulates incoming sensor data
   * dev_buff — GPU-side copy of that sensor data
   * weight / dev_weight — the neural network weights, on CPU and GPU respectively
   * category — single float where the GPU result gets copied back to
   * dev_cat — GPU-side output float
  */
  float *recv_buffer, *dev_buff, *weight, *dev_weight;
  float category, *dev_cat;
  double st, et;

  if (argc <= 1) exit(-1);

  int runlen = atoi(argv[1]);

  if (!runlen) exit(-1);
  
  /** 
   * Allocates the CPU buffers with regular malloc (pageable memory)
   * Weight matrix is 256×256 per layer × 10 layers
   */
  recv_buffer = (float*)malloc(sizeof(float)*256);
  weight = (float*)malloc(sizeof(float)*256*256*LAYERS);

  //initialized weights
  /**
   * Initializes weights — layer 1 gets 1e-5, layer 2 gets 2e-5, etc.
   * Just simple values so the network does something deterministic
   */
  for (int i = 0; i < LAYERS; ++i)
    {
      for (int j = 0; j < 256*256; ++j)
        weight[i*256*256 + j] = (i + 1.0)*1e-5;
    }
  
  /**
   * cudaMalloc allocates memory on the GPU — analogous to malloc but for GPU memory
   * Three separate GPU allocations: weights, input buffer, output scalar
   */
  cudaMalloc(&dev_weight, sizeof(float)*256*256*LAYERS);
  cudaMalloc(&dev_buff, sizeof(float)*256);
  cudaMalloc(&dev_cat, sizeof(float));

  /**
   * Copies the weights from CPU → GPU once before the loop
   * Can do bc weights will never change
   */
  cudaMemcpy(dev_weight, weight, sizeof(float)*256*256*LAYERS, 
	     cudaMemcpyHostToDevice);
  
  //create any streams or parallelization data structure above this line
  st = omp_get_wtime();
  
  /**
   * Loops runlen times (passed as command line arg), one full pipeline per iteration
   * 
   */
  for (int runs = 0; runs != runlen; ++runs)
    {
      //simulate receiving input from sensor
      /**
       * Simulates reading from a sensor — fills 256 floats with simple values
       * This must stay sequential per the assignment requirements
       */
      for (int i = 0; i < 256; ++i)
      {
        recv_buffer[i] = (runs+i)*1e-3;
      }
      
      //once buffer is full, send it to the GPU for processing
      /**
       * Copies sensor data CPU → GPU
       * This is synchronous — CPU blocks here until the copy finishes
       */
      cudaMemcpy(dev_buff, recv_buffer, sizeof(float)*256, cudaMemcpyHostToDevice);

      /**
       * Launches the kernel: 1 block, 256 threads
       * The <<<1, 256>>> syntax is CUDA's launch config — no stream specified means it uses stream 0 (default), 
       *      so it runs sequentially after everything before it
       */
      //Call GPU Kernel
      gpu_kernel<<<1, 256>>>(dev_buff, dev_weight, dev_cat);
      
      /**
       * cudaDeviceSynchronize() — CPU waits here until the GPU fully finishes
       * Then copies result back from GPU -> CPU
       * Prints the run number and result
       */
      //copy result from GPU to CPU
      cudaDeviceSynchronize();      
      cudaMemcpy(&category, dev_cat, sizeof(float), cudaMemcpyDeviceToHost);
      printf("%d %e\n", runs, category);
    }

  cudaDeviceSynchronize();
    
  et = omp_get_wtime();

  //remember to clean up your streams and data structure you have introduced. 



  
  //do not change the code below this line  

  cout<<(et-st)<<" seconds for "<<runlen<<" runs"<<endl;

  cudaFree(dev_weight);
  cudaFree(dev_buff);
  cudaFree(dev_cat);

  free(recv_buffer);
}
