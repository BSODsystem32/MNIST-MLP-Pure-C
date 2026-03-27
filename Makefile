CC      = gcc
CFLAGS  = -std=c99 -O2 -Wall -Wextra -Wpedantic
LDFLAGS = -lm

TARGET = mnist_mlp
SRCS   = main.c mnist.c matrix.c network.c
OBJS   = $(SRCS:.c=.o)

.PHONY: all clean run download

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

download:
	mkdir -p data
	cd data && for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
	  t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do \
	    [ -f $$f ] || (curl -LO "http://yann.lecun.com/exdb/mnist/$${f}.gz" && gunzip "$${f}.gz"); \
	done

run: all
	./$(TARGET) ./data weights_v2.bin

clean:
	rm -f $(OBJS) $(TARGET) weights_v2.bin
