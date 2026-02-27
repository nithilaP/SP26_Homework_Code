all:
	nvcc --resource-usage banks.cu -o banks.x
	nvcc --resource-usage banks_shared.cu -o banks_shared.x
	nvcc --resource-usage banks_multi_kernel.cu -o banks_multi_kernel.x

clean:
	rm -f *~ *.x
