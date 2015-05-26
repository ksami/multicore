
CXX=g++
CXXFLAGS=-Wall


LIBS = -lrt
LDFLAGS = ${LIBS}


all: seq pthread opencl openmp

.PHONY: all seq pthread opencl openmp clean test


seq: kmeans_seq

kmeans_seq: kmeans_seq.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS}


pthread: kmeans_pthread

kmeans_pthread: kmeans_pthread.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS} -lpthread


opencl: kmeans_opencl

kmeans_opencl: kmeans_opencl.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS} -lOpenCL


openmp: kmeans_openmp

kmeans_openmp: kmeans_openmp.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS} -fopenmp


run_all:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64
	thorq --add kmeans_seq centroid.point data.point result_seq.class final_centroid_seq.point 1024
	thorq --add kmeans_pthread centroid.point data.point result_par.class final_centroid_par.point 1024

run_seq:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64
	thorq --add kmeans_seq centroid.point data.point result_seq.class final_centroid_seq.point 1024

run_cpu:
	./gen_data.py centroid 16 centroid.point
	./gen_data.py data 1048576 data.point 16
	thorq --add kmeans_opencl centroid.point data.point result_par.class final_centroid_par.point 1024

run:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64
	thorq --add --device gpu kmeans_openmp centroid.point data.point result_par.class final_centroid_par.point 1024
	

clean:
	rm -f kmeans_seq kmeans_pthread kmeans_opencl kmeans_openmp kmeans_main.o kmeans_seq.o kmeans_pthread.o kmeans_opencl.o kmeans_openmp.o centroid.point data.point final_centroid.point result.class task*
