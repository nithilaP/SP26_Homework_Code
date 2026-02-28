2. Cost of Parallelism 

## In order to set up CUDA: 

Check if CUDA is installed: 
    ls -l /usr/local/cuda/bin/nvcc

Add CUDA to your PATH: 
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

Verify CUDA: 
    which nvcc
    nvcc --version

## In order to run any .cu file and get measurements to copy over floats:
    make
    ./startup.x

## In order to run any <file_name>.cu file and get measurements to copy over floats:
    make
    ./<file_name>.x

## In order ot run the profiler on <file_name>.cu, 
    nsys nvprof ./<file_name>.x
