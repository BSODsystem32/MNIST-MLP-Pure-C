# Makefile — training binary

CC      = gcc
# -O3 -march=native: full loop vectorisation + target-specific SIMD
# -ffast-math:       allows reassociation, reciprocal sqrt, etc.
#                    (safe here: we don't rely on strict IEEE exceptions)
# -funroll-loops:    unroll small inner loops (BN, dropout)
CFLAGS  = -std=c99 -O3 -march=native -ffast-math -funroll-loops \
          -Wall -Wextra -Wpedantic
LDFLAGS = -lm

# ---------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------
ifeq ($(OS),Windows_NT)
  EXE     := .exe
  NULL     = NUL
  CURL     = curl -L --output
  DATADIR  = ..\data
  SEP      = $(strip \)
else
  EXE     :=
  NULL     = /dev/null
  CURL     = curl -L -o
  DATADIR  = ../data
  SEP      = /
endif

# ---------------------------------------------------------------
# OpenMP — enabled by default, disable with: make OPENMP=0
# macOS ships Clang without OpenMP; install via: brew install libomp
# ---------------------------------------------------------------
OPENMP ?= 1
ifeq ($(OPENMP),1)
  CFLAGS  += -fopenmp
  LDFLAGS += -fopenmp
endif

TARGET = mnist_mlp$(EXE)
OBJS   = main.o mnist.o matrix.o network.o augment.o

.PHONY: all clean download

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ---------------------------------------------------------------
# Explicit compile rules
# ---------------------------------------------------------------
main.o:    main.c    network.h mnist.h augment.h
	$(CC) $(CFLAGS) -c -o $@ $<

mnist.o:   mnist.c   mnist.h
	$(CC) $(CFLAGS) -c -o $@ $<

matrix.o:  matrix.c  matrix.h
	$(CC) $(CFLAGS) -c -o $@ $<

network.o: network.c network.h matrix.h
	$(CC) $(CFLAGS) -c -o $@ $<

augment.o: augment.c augment.h
	$(CC) $(CFLAGS) -c -o $@ $<

# ---------------------------------------------------------------
# clean
# ---------------------------------------------------------------
clean:
ifeq ($(OS),Windows_NT)
	-del /Q /F $(OBJS) 2>$(NULL)
	-del /Q /F $(TARGET) 2>$(NULL)
else
	rm -f $(OBJS) $(TARGET)
endif

# ---------------------------------------------------------------
# Download MNIST
# ---------------------------------------------------------------
MNIST_URL   = http://yann.lecun.com/exdb/mnist
MNIST_FILES = \
  train-images-idx3-ubyte \
  train-labels-idx1-ubyte \
  t10k-images-idx3-ubyte  \
  t10k-labels-idx1-ubyte

download:
ifeq ($(OS),Windows_NT)
	if not exist "$(DATADIR)" mkdir "$(DATADIR)"
	$(foreach f,$(MNIST_FILES),\
	  if not exist "$(DATADIR)$(SEP)$(f)" (\
	    curl -L --output "$(DATADIR)$(SEP)$(f).gz" "$(MNIST_URL)/$(f).gz" &&\
	    tar -xzf "$(DATADIR)$(SEP)$(f).gz" -C "$(DATADIR)" &&\
	    del "$(DATADIR)$(SEP)$(f).gz") &)
else
	mkdir -p $(DATADIR)
	$(foreach f,$(MNIST_FILES),\
	  test -f $(DATADIR)/$(f) ||\
	    ($(CURL) $(DATADIR)/$(f).gz $(MNIST_URL)/$(f).gz &&\
	     gunzip $(DATADIR)/$(f).gz);)
endif
