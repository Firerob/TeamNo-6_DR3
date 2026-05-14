# ============================================================
# TeamNoX-D3 — Makefile CUDA
# Compilador: nvcc (CUDA Toolkit)
# ============================================================

NVCC      = nvcc
# Ajustar -arch según la GPU disponible:
#   sm_60 → Pascal (GTX 1080, P100)
#   sm_70 → Volta  (V100)
#   sm_75 → Turing (RTX 2080)
#   sm_80 → Ampere (A100, RTX 3090)
ARCH      = sm_60
NVCCFLAGS = -O2 -arch=$(ARCH) -Xcompiler -Wall
LDFLAGS   = -lm
TARGET    = matmul_cuda
SRC       = main.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) < input.txt

clean:
	rm -f $(TARGET)