
CXX=mpicc
CXXFLAGS=-Wall


LIBS = -lrt
LDFLAGS = ${LIBS}


all: mpi

.PHONY: all mpi


mpi: kmeans_mpi

kmeans_mpi: kmeans_mpi.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS}


run:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64
	thorq --add --mode mpi --slots 16 --nodes 4 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	

clean:
	rm -f kmeans_mpi kmeans_main.o kmeans_mpi.o centroid.point data.point final_centroid.point result.class task*
