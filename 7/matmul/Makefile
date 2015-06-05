TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=mpicc
CFLAGS=-g -Wall
LDFLAGS=

mat_mul: mat_mul.c 
	$(CC) $^ $(CFLAGS) -o $@ ${LDFLAGS}

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add --mode mpi --slots 16 --nodes 4 ./$(TARGET)
