2. Cost of OpenMP Task Parallelism 

For all Sequential Compilations: 

gcc -O3 triangle.c -o triangle_seq

Run it using: ./triangle_seq <runs> < inputfile
Note: start with 5 runs to get stable data. 

To run OpenMP Version: 
gcc -O3 -fopenmp triangle.c -o triangle_omp

Control the number of threads using: OMP_NUM_THREADS
Example Run: OMP_NUM_THREADS=1  ./triangle_omp 5 < smallJA.txt

Using the Makefile: 
    Sequential Runs: make run_small_sequential | tee small_sequential_t1.txt
    OpenMP Runs: 
        First run: unset OMP_PROC_BIND
                    unset OMP_PLACES
        OMP_NUM_THREADS=1  make run_small_open_mp | tee small_open_mp_t1.txt

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
