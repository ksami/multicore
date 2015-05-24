TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=gcc
CFLAGS=-g -Wall
LDFLAGS=-fopenmp

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

openmp: mat_mul_openmp

mat_mul_openmp: mat_mul.o
	$(CC) $^ -o $@ ${LDFLAGS}

clean:
	rm -rf $(TARGET) $(OBJS) task*

run_cpu: $(TARGET)
	thorq --add ./$(TARGET) 32

run: $(TARGET)
	thorq --add --device gpu ./$(TARGET) 32
