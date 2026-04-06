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

  /* create a buffer for each stream */
  float* recv_buffer[NUM_STREAMS]; // host mem that sensor data is written to -> one for each buffer (or stream)
  float* category[NUM_STREAMS]; // host mem that the output from CPU gets put into (by thread 1)
  float* dev_buff[NUM_STREAMS]; // gpu input buffer -> one for each sensor buffer 
  float* dev_cat[NUM_STREAMS]; // gpu output buffer -> one for each sensor buffer, need to reset in thread 2 before using slot again to avoid storing values over time. 

  /* value shared by all streams. */
  float *weight, *dev_weight; // 

  /* from starter code */
  double st, et;

  if (argc <= 1) exit(-1);

  int runlen = atoi(argv[1]);

  if (!runlen) exit(-1);

    /** cuda streams from lecture slides: 
   * 
   * cudaStream_t streams[4];
   * cudaStreamCreateWithFlags(&streams[i],
   *  cudaStreamNonBlocking);
   * ...
   * kernel_call<<<grid, block, smem_size, streams[i]>>>
   * ...
   * cudaStreamSynchronize(stream[i]);
   * cudaStreamDestroy(streams[i]);
   * 
   * Copying data can be performed asynchronously per stream: cudaMemcpyAsync(dst, src, size, direction, stream
   */
  cudaStream_t streams[NUM_STREAMS]; // declare NUM_STREAMS cude streams

  /** tracking arrays for each stream. 
   * each buffer -> has a corresponding stream. 
   * needed to add volatile since different threads editing values on stream 
  */
  /* track if corresponding buffer for each stream is full. (thread 0 -> thread 1) */
  volatile bool sens_data_buf_full[NUM_STREAMS]; // false -> buffer open for new data, true -> buffer full

  /* track if we got the processed data for each buffer. (thread 1 -> thread 2) */
  volatile bool receive_processed_data[NUM_STREAMS]; // false -> stream still running, stream processed data already

  /* track if a buffer slot is free to be filled w new data. (thread 2 -> thread 1) */
  volatile bool stream_output_slot_open[NUM_STREAMS]; // false -> stream output has not been processed & reset yet, true -> stream output has been processed

  /* allocate buffers */
  for (int i = 0; i < NUM_STREAMS; i++){

    /** First try with malloc: causes cudaMemcpyAsync sequential behavior w copy.
    * recv_buffer[i] = (float*)malloc(sizeof(float)*256);
    * category[i] = (float*)malloc(sizeof(float));
    * 
    * Need cudaMallocHost: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gab84100ae1fa1b12eaca660207ef585b
    * Need for cudaMemcpyAsync: https://stackoverflow.com/questions/23133203/cudamallochost-vs-malloc-for-better-performance-shows-no-difference
    */
    cudaMallocHost(&recv_buffer[i], sizeof(float)*256);
    cudaMallocHost(&category[i], sizeof(float));
    
    /* malloc mem on device */
    cudaMalloc(&dev_buff[i], sizeof(float)*256);
    cudaMalloc(&dev_cat[i], sizeof(float));

    /* create stream. */
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

    /* set flags */
    sens_data_buf_full[i] = false;
    receive_processed_data[i] = false;
    stream_output_slot_open[i] = true;
  }
  
  /* create CPU weights and init. */
  weight = (float*)malloc(sizeof(float)*256*256*LAYERS);
  for (int i = 0; i < LAYERS; ++i){
    for (int j = 0; j < 256*256; ++j){
        weight[i*256*256 + j] = (i + 1.0)*1e-5;
    }
  }

  /* one time copy of weights to GPU. */
  cudaMalloc(&dev_weight, sizeof(float)*256*256*LAYERS);
  cudaMemcpy(dev_weight, weight, sizeof(float)*256*256*LAYERS, cudaMemcpyHostToDevice);
 
  // create any streams or parallelization data structure above this line
  st = omp_get_wtime();
  
  /* synchronize w OpemMp parallel for 3 threads: https://rookiehpc.org/openmp/docs/num_threads/index.html*/
  #pragma omp parallel num_threads(3)
  {

    int thread_id = omp_get_thread_num();

    /* first thread does the sensor reads sequentially */
    if (thread_id == 0){

      for (int runs = 0; runs != runlen; ++runs){
        /** note: use run index to determine what buffer it gets allocated to. */
        
        /* wait for the sensor data buffer to be full to process. */
        while (sens_data_buf_full[(runs % NUM_STREAMS)] != false){};

        /**
         * Simulates reading from a sensor — fills 256 floats with simple values
         * This must stay sequential per the assignment requirements
         */
        for (int i = 0; i < 256; ++i){
          recv_buffer[(runs % NUM_STREAMS)][i] = (runs+i)*1e-3;
        }

        /* set sensor buffer full flag true for this buffer. */
        sens_data_buf_full[(runs % NUM_STREAMS)] = true;
      }
    } 
    else if (thread_id == 1){

      for (int runs = 0; runs < runlen; runs++){
        /** note: use run index to determine what which stream runs buffer. */

        /* Needed to add a wait until the value in this stream_output_slot_open is true for this stream output slot
        -> cannot process new data into this slot if not. 
         */
        while (stream_output_slot_open[(runs % NUM_STREAMS)] != true){};
        stream_output_slot_open[(runs % NUM_STREAMS)] = false;

        /* wait until the buffer is full*/
        while (sens_data_buf_full[(runs % NUM_STREAMS)] != true){};

        /**
         * 1. Clear curr value for stream output in dev_cat array
         * 2. Copies sensor data CPU → GPU
         * async — queues operation on stream and returns to cpu
         * since all ops below on same stream, each command runs seqentialaly
         */
        cudaMemsetAsync(dev_cat[(runs % NUM_STREAMS)], 0, sizeof(float), streams[(runs % NUM_STREAMS)]);
        cudaMemcpyAsync(dev_buff[(runs % NUM_STREAMS)], recv_buffer[(runs % NUM_STREAMS)], sizeof(float) * 256, cudaMemcpyHostToDevice, streams[(runs % NUM_STREAMS)]);

        /**
         * Launches the kernel: 1 block, 256 threads on each stream
         * <<<# thread blocks, # threads/ block, ant of shared mem, cuda stream>>> syntax -> CUDA's launch config
         */
        gpu_kernel<<<1, 256, 0, streams[(runs % NUM_STREAMS)]>>>(dev_buff[(runs % NUM_STREAMS)], dev_weight, dev_cat[(runs % NUM_STREAMS)]);
        
        /**
         * Copies result back from GPU -> CPU for buffer when ops above complete. 
         */
        cudaMemcpyAsync(category[(runs % NUM_STREAMS)], dev_cat[(runs % NUM_STREAMS)], sizeof(float), cudaMemcpyDeviceToHost, streams[(runs % NUM_STREAMS)]);
        
        /* update tracking variables. */
        sens_data_buf_full[(runs % NUM_STREAMS)] = false; // clear flag for next iter
        receive_processed_data[(runs % NUM_STREAMS)] = true; // set flag for recieve procssing
      }
    }
    else if (thread_id == 2){
      /** note: use run index to determine what which stream output to use for processing. */

      float curr_category_total = 0.0;

      for (int runs = 0; runs != runlen; ++runs){

        /* wait until the gpu stream has completed processing. */
        while (receive_processed_data[(runs % NUM_STREAMS)] == false){}

        /* make sure that all ops on this stream are done. */
        cudaStreamSynchronize(streams[(runs % NUM_STREAMS)]);

        /* to match the baseline output, get the sum of all previous run outputs and current output to report as category val for this stream. */
        curr_category_total += *category[(runs % NUM_STREAMS)];
        printf("%d %e\n", runs, curr_category_total);

        /* reset all flags and clear tracking variables */
        receive_processed_data[(runs % NUM_STREAMS)] = false; // clear flag for next round of processing on this stream
        stream_output_slot_open[(runs % NUM_STREAMS)] = true; // set flag so thread 1 can overwrite stream output for next round
      }
    }
  }
  /* join all threads*/
  cudaDeviceSynchronize();
    
  et = omp_get_wtime();

  //remember to clean up your streams and data structure you have introduced. 
  for (int i = 0; i < NUM_STREAMS; i++){

    /* as per the stream steps, destroy all streams when done. */
    cudaStreamDestroy(streams[i]);

    // First try w malloc: free(recv_buffer[i]);
    cudaFreeHost(recv_buffer[i]);
    recv_buffer[i] = NULL;

    // First try w malloc: free(category[i]);
    cudaFreeHost(category[i]);
    category[i] = NULL;

    cudaFree(dev_buff[i]);
    dev_buff[i] = NULL;

    cudaFree(dev_cat[i]);
    dev_cat[i] = NULL;
  }

  //do not change the code below this line  
  cout<<(et-st)<<" seconds for "<<runlen<<" runs"<<endl;

  cudaFree(dev_weight);

  /* note: modified these lines, since my code inits values below as array and does the free in loop above. */
  // cudaFree(dev_buff);
  // cudaFree(dev_cat);
  // free(recv_buffer);
}
