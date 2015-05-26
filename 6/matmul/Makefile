TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=gcc
CFLAGS=-g -Wall
LDFLAGS=-fopenmp

mat_mul: mat_mul.c 
	$(CC) $^ $(CFLAGS) -o $@ ${LDFLAGS}

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add --device gpu ./$(TARGET) 1
