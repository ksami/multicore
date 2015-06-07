
CXX=mpic++
CXXFLAGS=-Wall


LIBS = -lrt
LDFLAGS = ${LIBS}


all: mpi

.PHONY: all mpi


mpi: kmeans_mpi

kmeans_mpi: kmeans_mpi.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS}

expt:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 262144 data.point 64
	thorq --add --name 101  --mode mpi --slots 1  --nodes 1 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 102  --mode mpi --slots 2  --nodes 1 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 104  --mode mpi --slots 4  --nodes 1 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 108  --mode mpi --slots 8  --nodes 1 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 1016 --mode mpi --slots 16 --nodes 1 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 201  --mode mpi --slots 1  --nodes 2 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 202  --mode mpi --slots 2  --nodes 2 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 204  --mode mpi --slots 4  --nodes 2 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 208  --mode mpi --slots 8  --nodes 2 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 2016 --mode mpi --slots 16 --nodes 2 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 301  --mode mpi --slots 1  --nodes 3 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 302  --mode mpi --slots 2  --nodes 3 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 304  --mode mpi --slots 4  --nodes 3 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 308  --mode mpi --slots 8  --nodes 3 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 3016 --mode mpi --slots 16 --nodes 3 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 4016 --mode mpi --slots 1  --nodes 4 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 4016 --mode mpi --slots 2  --nodes 4 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 4016 --mode mpi --slots 4  --nodes 4 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 4016 --mode mpi --slots 8  --nodes 4 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	thorq --add --name 4016 --mode mpi --slots 16 --nodes 4 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	

run:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 262144 data.point 64
	thorq --add --mode mpi --slots 16 --nodes 4 kmeans_mpi centroid.point data.point result_par.class final_centroid_par.point 1024
	

clean:
	rm -f kmeans_mpi kmeans_main.o kmeans_mpi.o centroid.point data.point final_centroid.point result.class task*
