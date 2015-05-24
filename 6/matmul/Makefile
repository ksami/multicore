TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=gcc
CFLAGS=-g -O2 -Wall
LDFLAGS=-fopenmp

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

clean:
	rm -rf $(TARGET) $(OBJS) task*

run_cpu: $(TARGET)
	thorq --add ./$(TARGET) -v

run: $(TARGET)
	thorq --add --device gpu ./$(TARGET) -v
