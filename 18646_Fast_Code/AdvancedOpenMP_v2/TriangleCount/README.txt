2. Cost of OpenMP Task Parallelism 

Sequential Runs: make run_<dataset_size>_sequential # where dataset_size = [small, medium, large]
    Ex: make run_small_sequential

OpenMP Runs: 
    unset OMP_PROC_BIND
    unset OMP_PLACES
    OMP_NUM_THREADS=<N>  make run_<dataset_size>_open_mp # where N is number of threads you want to run with [1,32].

    Ex: OMP_NUM_THREADS=16  make run_large_open_mp

