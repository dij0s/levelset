all:
	nvcc main.cpp write/write.cpp initialization/init.cpp solve/solve.cu diagnostics/diagnostics.cu -o run_serpentin
	nvprof ./run_serpentin

clean:
	rm run_serpentin