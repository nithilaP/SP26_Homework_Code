all:
	# running on ece024: sm_75 architecture
	nvcc -x cu main.cpp -Xcompiler -fopenmp -o main.x -arch=sm_75 -std=c++11
	nvcc -x cu main_optimized.cpp -Xcompiler -fopenmp -o main_optimized.x -arch=sm_75 -std=c++11
run:
	./main.x 20
	./main_optimized.x 20

clean:
	rm *~ *.x
