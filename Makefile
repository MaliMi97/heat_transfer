GPU_ARCH = sm_50

NVCC = nvcc
NVCCFLAGS = -std=c++11 --use_fast_math -O3 -arch $(GPU_ARCH)

clean:
	rm -f *.o main

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $< -c $@

main: % : %.o
	$(NVCC) $(NVCCFLAGS) $< -o $@
