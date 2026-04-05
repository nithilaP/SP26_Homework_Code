#include <omp.h>
#include <iostream>

/** Big Idea: Use Streams to Do the operations for each buffer of data (allocate multiple buffers).
 * 
 */

#define LAYERS 10
#define NUM_STREAMS 4 

using namespace std;

__device__ float warpReduce(float init_val) {
  // Seed starting value as inverse lane ID
  float value = init_val;

  // Use XOR mode to perform butterfly reduction
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
  
  //10 layer ANN
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
//   float *recv_buffer, *dev_buff, *weight, *dev_weight;
//   float category, *dev_cat;
  float *weight, *dev_weight; // used by all streams

  double st, et;

  if (argc <= 1) exit(-1);

  int runlen = atoi(argv[1]);

  if (!runlen) exit(-1);

  /* create a buffer for each stream */
  float* recv_buffer[NUM_STREAMS];
  float* result[NUM_STREAMS];
  float* dev_buff[NUM_STREAMS];
  float* dev_cat[NUM_STREAMS];

  /** cuda streams
   * 
   * cudaStream_t streams[4];
   * cudaStreamCreateWithFlags(&streams[i],
   * cudaStreamNonBlocking);
   * ...
   * kernel_call<<<grid, block, smem_size, streams[i]>>>
   * ...
   * cudaStreamSynchronize(stream[i]);
   * cudaStreamDestroy(streams[i]);
   * 
   * Copying data can be performed asynchronously per stream: cudaMemcpyAsync(dst, src, size, direction, stream
   */

  cudaStream_t streams[NUM_STREAMS];

  volatile int sens_data_buf_full[NUM_STREAMS]; // track if streams are full 
  volatile int receive_processed_data[NUM_STREAMS]; // 
  volatile int slot_free[NUM_STREAMS];

  /* allocate buffers */
  for (int i = 0; i < NUM_STREAMS; i++){
    cudaMallocHost(&recv_buffer[i], sizeof(float)*256);
    cudaMallocHost(&result[i], sizeof(float));

    cudaMalloc(&dev_buff[i], sizeof(float)*256);
    cudaMalloc(&dev_cat[i], sizeof(float));

    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

    /* set flags */
    sens_data_buf_full[i] = 0;
    receive_processed_data[i] = 0;
    slot_free[i] = 1;
  }
  
//   recv_buffer = (float*)malloc(sizeof(float)*256);
    
    weight = (float*)malloc(sizeof(float)*256*256*LAYERS);
    //initialized weights
    for (int i = 0; i < LAYERS; ++i){
        for (int j = 0; j < 256*256; ++j){
            weight[i*256*256 + j] = (i + 1.0)*1e-5;
        }
    }
  
  cudaMalloc(&dev_weight, sizeof(float)*256*256*LAYERS);
//   cudaMalloc(&dev_buff, sizeof(float)*256);
//   cudaMalloc(&dev_cat, sizeof(float));

  cudaMemcpy(dev_weight, weight, sizeof(float)*256*256*LAYERS, 
	     cudaMemcpyHostToDevice);
 
  //create any streams or parallelization data structure above this line
  st = omp_get_wtime();
  
  /* synchronize w OpemMp parallel */
  #pragma omp parallel num_threads(3)
  {

    int thread_id = omp_get_thread_num();

    /* first thread does the sensor reads sequentially */
    if (thread_id == 0){

      for (int runs = 0; runs != runlen; ++runs)
        {
        //simulate receiving input from sensor
        
        // wait until the buffer is ready
        while (sens_data_buf_full[(runs % NUM_STREAMS)] != 0){};

        /**
         * Simulates reading from a sensor — fills 256 floats with simple values
         * This must stay sequential per the assignment requirements
         */
        for (int i = 0; i < 256; ++i)
        {
            recv_buffer[(runs % NUM_STREAMS)][i] = (runs+i)*1e-3;
        }

        sens_data_buf_full[(runs % NUM_STREAMS)] =1;
        }


    } 
    else if (thread_id == 1){

      for (int runs = 0; runs < runlen; runs++){

        while (slot_free[(runs % NUM_STREAMS)] != 1){};
        slot_free[(runs % NUM_STREAMS)] =0;

        /* wait until the buffer is full*/
        while (sens_data_buf_full[(runs % NUM_STREAMS)] != 1){};

                //once buffer is full, send it to the GPU for processing
        /**
         * Copies sensor data CPU → GPU
         * This is synchronous — CPU blocks here until the copy finishes
         */
        // cudaMemcpy(dev_buff, recv_buffer, sizeof(float)*256, cudaMemcpyHostToDevice);
        // cudaMemsetAsync(dev_cat[(runs % NUM_STREAMS)], 0, sizeof(float), streams[(runs % NUM_STREAMS)]);
        cudaMemcpyAsync(dev_buff[(runs % NUM_STREAMS)], recv_buffer[(runs % NUM_STREAMS)], sizeof(float) * 256, cudaMemcpyHostToDevice, streams[(runs % NUM_STREAMS)]);

        /**
         * Launches the kernel: 1 block, 256 threads
         * The <<<1, 256>>> syntax is CUDA's launch config — no stream specified means it uses stream 0 (default), 
         *      so it runs sequentially after everything before it
         */
        //Call GPU Kernel
        gpu_kernel<<<1, 256, 0, streams[(runs % NUM_STREAMS)]>>>(dev_buff[(runs % NUM_STREAMS)], dev_weight, dev_cat[(runs % NUM_STREAMS)]);
        
        /**
         * cudaDeviceSynchronize() — CPU waits here until the GPU fully finishes
         * Then copies result back from GPU -> CPU
         * Prints the run number and result
         */
        //copy result from GPU to CPU
        // cudaDeviceSynchronize(); 
        cudaMemcpyAsync(result[(runs % NUM_STREAMS)], dev_cat[(runs % NUM_STREAMS)], sizeof(float), cudaMemcpyDeviceToHost, streams[(runs % NUM_STREAMS)]);
     
        // cudaMemcpy(&category, dev_cat, sizeof(float), cudaMemcpyDeviceToHost);
        // printf("%d %e\n", runs, category);

        sens_data_buf_full[(runs % NUM_STREAMS)] =0;
        receive_processed_data[(runs % NUM_STREAMS)] =1;
      }
    }
    else if (thread_id == 2){

      for (int runs = 0; runs != runlen; ++runs)
      {

        // float running_total = 0.0f;

        while (receive_processed_data[(runs % NUM_STREAMS)] !=1){}

        cudaStreamSynchronize(streams[(runs % NUM_STREAMS)]);

        printf("%d %e\n", runs, *result[(runs % NUM_STREAMS)]);

        // running_total += *result[(runs % NUM_STREAMS)];
        cudaMemset(dev_cat[(runs % NUM_STREAMS)], 0, sizeof(float));

        receive_processed_data[(runs % NUM_STREAMS)] =0;
        slot_free[(runs % NUM_STREAMS)] =1;

      }
    }
  }
  /* join all threads*/
  cudaDeviceSynchronize();
    
  et = omp_get_wtime();

  //remember to clean up your streams and data structure you have introduced. 

  for (int i = 0; i < NUM_STREAMS; i++){

    cudaStreamSynchronize(streams[i]);

    cudaStreamDestroy(streams[i]);

    cudaFreeHost(recv_buffer[i]);
    cudaFreeHost(result[i]);

    cudaFree(dev_buff[i]);
    cudaFree(dev_cat[i]);

  }

  
  //do not change the code below this line  

  cout<<(et-st)<<" seconds for "<<runlen<<" runs"<<endl;

  cudaFree(dev_weight);
  // cudaFree(dev_buff);
  // cudaFree(dev_cat);

  // free(recv_buffer);
}
