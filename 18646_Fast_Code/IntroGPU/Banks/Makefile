all:
	nvcc --resource-usage banks.cu -o banks.x
	nvcc --resource-usage banks_shared.cu -o banks_shared.x

clean:
	rm -f *~ *.x
