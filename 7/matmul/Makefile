TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=mpicc
CFLAGS=-g -Wall
LDFLAGS=

mat_mul: mat_mul.c 
	$(CC) $^ $(CFLAGS) -o $@ ${LDFLAGS}

clean:
	rm -rf $(TARGET) $(OBJS) task*

expt: $(TARGET)
	thorq --add --name 101  --mode mpi --slots 1  --nodes 1 ./$(TARGET)
	thorq --add --name 102  --mode mpi --slots 2  --nodes 1 ./$(TARGET)
	thorq --add --name 104  --mode mpi --slots 4  --nodes 1 ./$(TARGET)
	thorq --add --name 108  --mode mpi --slots 8  --nodes 1 ./$(TARGET)
	thorq --add --name 1016 --mode mpi --slots 16 --nodes 1 ./$(TARGET)
	thorq --add --name 201  --mode mpi --slots 1  --nodes 2 ./$(TARGET)
	thorq --add --name 202  --mode mpi --slots 2  --nodes 2 ./$(TARGET)
	thorq --add --name 204  --mode mpi --slots 4  --nodes 2 ./$(TARGET)
	thorq --add --name 208  --mode mpi --slots 8  --nodes 2 ./$(TARGET)
	thorq --add --name 2016 --mode mpi --slots 16 --nodes 2 ./$(TARGET)
	thorq --add --name 301  --mode mpi --slots 1  --nodes 3 ./$(TARGET)
	thorq --add --name 302  --mode mpi --slots 2  --nodes 3 ./$(TARGET)
	thorq --add --name 304  --mode mpi --slots 4  --nodes 3 ./$(TARGET)
	thorq --add --name 308  --mode mpi --slots 8  --nodes 3 ./$(TARGET)
	thorq --add --name 3016 --mode mpi --slots 16 --nodes 3 ./$(TARGET)
	thorq --add --name 401  --mode mpi --slots 1  --nodes 4 ./$(TARGET)
	thorq --add --name 402  --mode mpi --slots 2  --nodes 4 ./$(TARGET)
	thorq --add --name 404  --mode mpi --slots 4  --nodes 4 ./$(TARGET)
	thorq --add --name 408  --mode mpi --slots 8  --nodes 4 ./$(TARGET)
	thorq --add --name 4016 --mode mpi --slots 16 --nodes 4 ./$(TARGET)
	
run: $(TARGET)
	thorq --add --mode mpi --slots 16 --nodes 4 ./$(TARGET)
