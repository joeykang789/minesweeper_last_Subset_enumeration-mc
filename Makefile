# CUDA Minesweeper - Makefile (Unix/WSL/Linux compatible)

# CUDA compiler
NVCC = nvcc

# Project paths
SRC_DIR = src
INC_DIR = include
LIB_DIR = lib

# Source files
SRCS = $(SRC_DIR)/minesweeper_kernel.cu

# Output
ifeq ($(OS),Windows_NT)
  LIB_TARGET = $(LIB_DIR)/minesweeper.dll
  RM = rmdir /s /q
  MKDIR = mkdir
else
  LIB_TARGET = $(LIB_DIR)/libminesweeper.so
  RM = rm -rf
  MKDIR = mkdir -p
endif

# Architecture: RTX 5080 Laptop = Compute Capability 12.0
ARCH = -arch=sm_120

# Include paths
INCLUDES = -I$(INC_DIR)

# Compiler flags
NVCC_FLAGS = $(ARCH) $(INCLUDES) --use_fast_math -Xptxas -O3 -lineinfo -maxrregcount=128

# Build rules
.PHONY: all clean

all: $(LIB_TARGET)

$(LIB_DIR):
	$(MKDIR) $(LIB_DIR)

$(LIB_TARGET): $(SRCS) | $(LIB_DIR)
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $^

clean:
	$(RM) $(LIB_DIR)
