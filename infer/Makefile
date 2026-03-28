# infer/Makefile — inference binaries
#
# Cross-compile for ARM Linux (e.g. Raspberry Pi):
#   make CC=arm-linux-gnueabihf-gcc
#
# Cross-compile for Cortex-M bare-metal:
#   make CC=arm-none-eabi-gcc \
#        CFLAGS="-std=c99 -O2 -mthumb -mcpu=cortex-m7 -ffunction-sections"

CC      = gcc
CFLAGS  = -std=c99 -O2 -Wall -Wextra -Wpedantic
LDFLAGS = -lm

# ---------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------
ifeq ($(OS),Windows_NT)
  EXE  := .exe
  NULL  = NUL
else
  EXE  :=
  NULL  = /dev/null
endif

COMMON = infer.o

EXPORT = infer_export$(EXE)
BENCH  = infer_bench$(EXE)
SINGLE = infer_single$(EXE)

TARGETS = $(EXPORT) $(BENCH) $(SINGLE)

.PHONY: all clean

all: $(TARGETS)

infer.o: infer.c infer.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(EXPORT): export.o $(COMMON)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BENCH): bench.o $(COMMON)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(SINGLE): single.o $(COMMON)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c infer.h
	$(CC) $(CFLAGS) -c -o $@ $<

# ---------------------------------------------------------------
# clean — single target, branch inside recipe
# ---------------------------------------------------------------
clean:
ifeq ($(OS),Windows_NT)
	-del /Q /F *.o 2>$(NULL)
	-del /Q /F $(TARGETS) 2>$(NULL)
else
	rm -f *.o $(TARGETS)
endif
