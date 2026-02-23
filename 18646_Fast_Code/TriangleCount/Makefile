run_small_base:
	gcc -O2 triangle.c -o triangle.x -std=c99 -DN=6474 -DNUM_A=25144 -fopenmp
	cat small_IA.txt small_JA.txt | ./triangle.x 10

run_small_tri_for:
	gcc -O2 tri_for.c -o tri_for.x -std=c99 -DN=6474 -DNUM_A=25144 -fopenmp
	cat small_IA.txt small_JA.txt | ./tri_for.x 10

run_small_tri_par:
	gcc -O2 tri_par.c -o tri_par.x -std=c99 -DN=6474 -DNUM_A=25144 -fopenmp
	cat small_IA.txt small_JA.txt | ./tri_par.x 10

run_small_tri_par_crit:
	gcc -O2 tri_par_crit.c -o tri_par_crit.x -std=c99 -DN=6474 -DNUM_A=25144 -fopenmp
	cat small_IA.txt small_JA.txt | ./tri_par_crit.x 10

run_medium_base:
	gcc -O2 triangle.c -o triangle.x -std=c99 -DN=9877 -DNUM_A=51946 -fopenmp
	cat medium_IA.txt medium_JA.txt | ./triangle.x 10

run_medium_tri_for:
	gcc -O2 tri_for.c -o tri_for.x -std=c99 -DN=9877 -DNUM_A=51946 -fopenmp
	cat medium_IA.txt medium_JA.txt | ./tri_for.x 10

run_medium_tri_par:
	gcc -O2 tri_par.c -o tri_par.x -std=c99 -DN=9877 -DNUM_A=51946 -fopenmp
	cat medium_IA.txt medium_JA.txt | ./tri_par.x 10

run_medium_tri_par_crit:
	gcc -O2 tri_par_crit.c -o tri_par_crit.x -std=c99 -DN=9877 -DNUM_A=51946 -fopenmp
	cat medium_IA.txt medium_JA.txt | ./tri_par_crit.x 10

run_large_base:
	gcc -O2 triangle.c -o triangle.x -std=c99 -DN=22687 -DNUM_A=109410 -fopenmp
	cat large_IA.txt large_JA.txt | ./triangle.x 10

run_large_tri_for:
	gcc -O2 tri_for.c -o tri_for.x -std=c99 -DN=22687 -DNUM_A=109410 -fopenmp
	cat large_IA.txt large_JA.txt | ./tri_for.x 10

run_large_tri_par:
	gcc -O2 tri_par.c -o tri_par.x -std=c99 -DN=22687 -DNUM_A=109410 -fopenmp
	cat large_IA.txt large_JA.txt | ./tri_par.x 10

run_large_tri_par_crit:
	gcc -O2 tri_par_crit.c -o tri_par_crit.x -std=c99 -DN=22687 -DNUM_A=109410 -fopenmp
	cat large_IA.txt large_JA.txt | ./tri_par_crit.x 10

cleanup:
	rm -rf *~
	rm -rf *.x

