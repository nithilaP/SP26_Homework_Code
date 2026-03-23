all:
	nvcc -x cu main.cpp -Xcompiler -fopenmp -o main.x -arch=sm_70 -std=c++11

run:
	./main.x 20

clean:
	rm *~ *.x
