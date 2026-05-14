# ============================================================
# TeamNoX-D3 — Makefile MPI
# Compilador: mpicc (OpenMPI / MPICH)
# ============================================================

CC       = mpicc
CFLAGS   = -O2 -Wall -Wextra -std=c99
LDFLAGS  = -lm
TARGET   = matmul_mpi
SRC      = main.c

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# Ejecutar con 4 procesos y archivo de prueba
run: $(TARGET)
	mpirun -np 4 ./$(TARGET) < input.txt

clean:
	rm -f $(TARGET)