# Root Makefile
#
# Targets:
#   make            build everything (training + inference)
#   make train      build training binary only
#   make infer      build inference binaries only
#   make export     train then fold BN, writes models/model.inf
#   make bench      run inference benchmark on test set
#   make download   fetch MNIST into data/
#   make clean      remove all build artifacts

# ---------------------------------------------------------------
# OS detection — $(OS) is set to "Windows_NT" by Windows itself
# (works in cmd.exe, PowerShell, MinGW, MSYS2, Cygwin)
# ---------------------------------------------------------------
ifeq ($(OS),Windows_NT)
  D       := $(strip \)
  EXE     := .exe
  MKDIR_P  = if not exist "$(1)" mkdir "$(1)"
else
  D       := /
  EXE     :=
  MKDIR_P  = mkdir -p $(1)
endif

MODELS_DIR = models
DATA_DIR   = data

.PHONY: all train infer export bench download clean

all: train infer

train:
	$(MAKE) -C src

infer:
	$(MAKE) -C infer

export: train infer
	$(call MKDIR_P,$(MODELS_DIR))
	src$(D)mnist_mlp$(EXE) $(DATA_DIR) $(MODELS_DIR)$(D)weights_v2.bin
	infer$(D)infer_export$(EXE) $(MODELS_DIR)$(D)weights_v2.bin $(MODELS_DIR)$(D)model.inf

bench: infer
	infer$(D)infer_bench$(EXE) $(MODELS_DIR)$(D)model.inf $(DATA_DIR)

download:
	$(MAKE) -C src download

clean:
	$(MAKE) -C src clean
	$(MAKE) -C infer clean
