# Sequential Runs
run_small_sequential:
	gcc -O2 triangle.c -o triangle.x -std=c99 -DN=6474 -DNUM_A=25144
	cat small_IA.txt small_JA.txt | ./triangle.x 10

run_medium_sequential:
	gcc -O2 triangle.c -o triangle.x -std=c99 -DN=9877 -DNUM_A=51946
	cat medium_IA.txt medium_JA.txt | ./triangle.x 10

run_large_sequential:
	gcc -O2 triangle.c -o triangle.x -std=c99 -DN=22687 -DNUM_A=109410
	cat large_IA.txt large_JA.txt | ./triangle.x 10

# OpenMP Version 
run_small_open_mp:
	gcc -O2 triangle.c -o triangle_omp_ver.x -std=c99 -DN=6474 -DNUM_A=25144 -fopenmp
	cat small_IA.txt small_JA.txt | ./triangle_omp_ver.x 10

run_medium_open_mp:
	gcc -O2 triangle.c -o triangle_omp_ver.x -std=c99 -DN=9877 -DNUM_A=51946 -fopenmp
	cat medium_IA.txt medium_JA.txt | ./triangle_omp_ver.x 10

run_large_open_mp:
	gcc -O2 triangle.c -o triangle_omp_ver.x -std=c99 -DN=22687 -DNUM_A=109410 -fopenmp
	cat large_IA.txt large_JA.txt | ./triangle_omp_ver.x 10


cleanup:
	rm -rf *~
	rm -rf *.x

