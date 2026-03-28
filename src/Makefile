# src/Makefile — training binary

CC      = gcc
CFLAGS  = -std=c99 -O2 -Wall -Wextra -Wpedantic
LDFLAGS = -lm

# ---------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------
ifeq ($(OS),Windows_NT)
  EXE     := .exe
  RM_F     = del /Q /F
  NULL     = NUL
  CURL     = curl -L --output
  DATADIR  = ..\data
  SEP      = $(strip \)
else
  EXE     :=
  RM_F     = rm -f
  NULL     = /dev/null
  CURL     = curl -L -o
  DATADIR  = ../data
  SEP      = /
endif

TARGET = mnist_mlp$(EXE)
SRCS   = main.c mnist.c matrix.c network.c
OBJS   = $(SRCS:.c=.o)

.PHONY: all clean download

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# ---------------------------------------------------------------
# clean — single target, branch on $(OS) inside the recipe
# using cmd.exe syntax on Windows, sh syntax elsewhere
# ---------------------------------------------------------------
clean:
ifeq ($(OS),Windows_NT)
	-del /Q /F $(OBJS) 2>$(NULL)
	-del /Q /F $(TARGET) 2>$(NULL)
else
	$(RM_F) $(OBJS) $(TARGET)
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
