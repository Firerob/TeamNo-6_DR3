/*
 * ============================================================
 * TeamNo6-D3 — CUDA Matrix Multiplication
 * ============================================================
 * Proyecto universitario: Multiplicación de matrices NxN en
 * GPU usando CUDA C con:
 *   - Tiling con shared memory para reducir accesos a DRAM
 *   - __syncthreads() para sincronización de barrera
 *   - Eventos CUDA para medición de tiempo precisa
 *   - Validación contra referencia secuencial (golden ref)
 *
 * Compilar : nvcc -O2 -arch=sm_60 -o matmul_cuda main.cu -lm
 * Ejecutar : ./matmul_cuda < input.txt
 *
 * Formato entrada (stdin):
 *   <dim>
 *   <a00>,<a01>,...  (N*N valores separados por coma)
 *   <b00>,<b01>,...  (N*N valores separados por coma)
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

/* ──────────────────────────────────────────────────────────
 * MACRO: tamaño del tile (bloque de hilos 2D en CUDA).
 * TILE_SIZE × TILE_SIZE hilos por bloque.
 * Debe ser ≤ sqrt(max_threads_per_block) y potencia de 2.
 * Para la mayoría de GPUs: TILE_SIZE = 16 o 32.
 * ────────────────────────────────────────────────────────── */
#define TILE_SIZE 16

/* Tolerancia para comparación de flotantes */
#define EPSILON 1e-3

/* ============================================================
 * MACRO DE VERIFICACIÓN CUDA
 * Envuelve cada llamada CUDA para detectar errores.
 * Si hay error imprime archivo, línea y mensaje, luego aborta.
 * ============================================================ */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error en %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* ============================================================
 * SECCIÓN 1 — PARSER DE ENTRADA (idéntico a versión MPI)
 * ============================================================ */

static int read_int_line(void)
{
    char buf[64];
    if (fgets(buf, sizeof(buf), stdin) == NULL) return -1;
    return atoi(buf);
}

static int read_matrix_csv(double *mat, int N)
{
    size_t buf_size = (size_t)N * N * 24 + 4;
    char *buf = (char *)malloc(buf_size);
    if (!buf) { fprintf(stderr, "read_matrix_csv: malloc fallo\n"); return -1; }

    if (fgets(buf, (int)buf_size, stdin) == NULL) { free(buf); return -1; }

    size_t len = strlen(buf);
    if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';

    char *token = strtok(buf, ",");
    int idx = 0;
    while (token != NULL && idx < N * N) {
        mat[idx++] = atof(token);
        token = strtok(NULL, ",");
    }
    free(buf);
    if (idx != N * N) {
        fprintf(stderr, "read_matrix_csv: esperados %d valores, leidos %d\n", N*N, idx);
        return -1;
    }
    return 0;
}

/* ============================================================
 * SECCIÓN 2 — REFERENCIA SECUENCIAL (GOLDEN REFERENCE)
 * ============================================================ */

static void matmul_sequential(const double *A, const double *B, double *C, int N)
{
    int i, j, k;
    memset(C, 0, sizeof(double) * N * N);
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

/* ============================================================
 * SECCIÓN 3 — VALIDACIÓN
 * ============================================================ */

static int validate(const double *C_seq, const double *C_gpu, int N)
{
    int i;
    for (i = 0; i < N * N; i++) {
        double diff = fabs(C_seq[i] - C_gpu[i]);
        if (diff > EPSILON) {
            fprintf(stderr,
                    "VALIDACION FALLO en [%d]: seq=%.6f gpu=%.6f diff=%.2e\n",
                    i, C_seq[i], C_gpu[i], diff);
            return 0;
        }
    }
    return 1;
}

/* ============================================================
 * SECCIÓN 4 — KERNEL CUDA: MULTIPLICACIÓN CON TILING
 *
 * Algoritmo de tiling con shared memory:
 * ─────────────────────────────────────────────────────────
 * Cada bloque de hilos 2D (TILE_SIZE × TILE_SIZE) calcula
 * un sub-bloque (tile) de la matriz resultado C.
 *
 * Para cada tile de C[blockRow][blockCol]:
 *   Iterar sobre los tiles de A (fila blockRow) y B (columna blockCol):
 *     1. Cargar tile de A en shared memory (sA)
 *     2. Cargar tile de B en shared memory (sB)
 *     3. __syncthreads() — barrera: esperar que todos los hilos
 *        del bloque hayan terminado de cargar
 *     4. Acumular producto sA × sB en variable local
 *     5. __syncthreads() — barrera: esperar antes de sobreescribir
 *        la shared memory con el siguiente tile
 *   Escribir resultado acumulado en C[row][col]
 *
 * Por qué mejora el rendimiento:
 *   - Acceso a DRAM global: O(N^3 / TILE_SIZE) en lugar de O(N^3)
 *   - Shared memory (~100× más rápida que DRAM global)
 *   - Cada valor de A y B se carga TILE_SIZE veces en total
 *     para el tile, en lugar de TILE_SIZE veces por hilo
 *
 * Parámetros del kernel:
 *   A, B : matrices en device memory (row-major)
 *   C    : matriz resultado en device memory
 *   N    : dimensión (cuadrada)
 * ============================================================ */
__global__ void matmul_kernel(const double *A, const double *B, double *C, int N)
{
    /*
     * SHARED MEMORY: dos tiles de TILE_SIZE × TILE_SIZE doubles.
     * Declaración estática (tamaño conocido en tiempo de compilación).
     * Shared memory es compartida por TODOS los hilos de un bloque.
     */
    __shared__ double sA[TILE_SIZE][TILE_SIZE];
    __shared__ double sB[TILE_SIZE][TILE_SIZE];

    /* ── Índices del hilo dentro del bloque ── */
    int tx = threadIdx.x;   /* columna local del hilo */
    int ty = threadIdx.y;   /* fila    local del hilo */

    /* ── Índices globales de la fila y columna que calcula este hilo ── */
    int row = blockIdx.y * TILE_SIZE + ty;  /* fila    de C / A */
    int col = blockIdx.x * TILE_SIZE + tx;  /* columna de C / B */

    /*
     * Acumulador local: cada hilo suma las contribuciones
     * de todos los tiles antes de escribir en C.
     */
    double sum = 0.0;

    /* Número de tiles en la dimensión k */
    int num_tiles = N / TILE_SIZE;
    int t;

    for (t = 0; t < num_tiles; t++) {

        /* ──────────────────────────────────────────────────────
         * CARGA COLABORATIVA DE TILES EN SHARED MEMORY
         *
         * Cada hilo (ty, tx) carga exactamente un elemento:
         *   sA[ty][tx] ← A[row][t*TILE_SIZE + tx]
         *   sB[ty][tx] ← B[t*TILE_SIZE + ty][col]
         *
         * De esta forma, los TILE_SIZE² hilos del bloque
         * cargan el tile completo en paralelo.
         * ────────────────────────────────────────────────────── */
        sA[ty][tx] = A[row * N + (t * TILE_SIZE + tx)];
        sB[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];

        /* ──────────────────────────────────────────────────────
         * __syncthreads(): BARRERA DE SINCRONIZACIÓN
         *
         * Garantiza que TODOS los hilos del bloque han terminado
         * de escribir en sA y sB antes de que CUALQUIERA hilo
         * comience a leer de ellas.
         *
         * Sin esta barrera: un hilo podría leer datos del tile
         * anterior mientras otro aún está escribiendo el nuevo.
         * ────────────────────────────────────────────────────── */
        __syncthreads();

        /* ──────────────────────────────────────────────────────
         * PRODUCTO PUNTO DEL TILE
         * Cada hilo acumula TILE_SIZE multiplicaciones usando
         * datos de shared memory (latencia ~5 ciclos vs ~200
         * ciclos de DRAM global).
         * ────────────────────────────────────────────────────── */
        int k;
        for (k = 0; k < TILE_SIZE; k++) {
            sum += sA[ty][k] * sB[k][tx];
        }

        /* ──────────────────────────────────────────────────────
         * Segunda barrera: esperar que todos los hilos terminen
         * de LEER sA/sB antes de sobreescribirlas en la siguiente
         * iteración del loop de tiles.
         * ────────────────────────────────────────────────────── */
        __syncthreads();
    }

    /* ──────────────────────────────────────────────────────────
     * ESCRITURA DEL RESULTADO EN MEMORIA GLOBAL
     * Solo escribir si los índices están dentro de [0, N).
     * Necesario cuando N no es múltiplo de TILE_SIZE (aunque
     * en este proyecto se asume que sí lo es).
     * ────────────────────────────────────────────────────────── */
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

/* ============================================================
 * SECCIÓN 5 — PROGRAMA PRINCIPAL CUDA
 * ============================================================ */
int main(void)
{
    int N = 0;

    /* ── Punteros en host (CPU) ── */
    double *h_A   = NULL;  /* Matriz A en host */
    double *h_B   = NULL;  /* Matriz B en host */
    double *h_C   = NULL;  /* Resultado CUDA en host (copiado desde GPU) */
    double *h_Cseq= NULL;  /* Resultado secuencial (golden reference) */

    /* ── Punteros en device (GPU) ── */
    double *d_A   = NULL;
    double *d_B   = NULL;
    double *d_C   = NULL;

    /* ── Leer dimensión ── */
    N = read_int_line();
    if (N <= 0 || N % TILE_SIZE != 0) {
        fprintf(stderr, "Error: N=%d debe ser positivo y multiplo de TILE_SIZE=%d\n",
                N, TILE_SIZE);
        return EXIT_FAILURE;
    }

    size_t mat_bytes = sizeof(double) * (size_t)N * N;

    /* ─────────────────────────────────────────────
     * RESERVA DE MEMORIA EN HOST
     * malloc en CPU para A, B, C, C_seq
     * ───────────────────────────────────────────── */
    h_A    = (double *)malloc(mat_bytes);
    h_B    = (double *)malloc(mat_bytes);
    h_C    = (double *)malloc(mat_bytes);
    h_Cseq = (double *)malloc(mat_bytes);

    if (!h_A || !h_B || !h_C || !h_Cseq) {
        fprintf(stderr, "malloc fallo para matrices host\n");
        return EXIT_FAILURE;
    }

    /* ── Leer matrices desde stdin ── */
    if (read_matrix_csv(h_A, N) != 0 || read_matrix_csv(h_B, N) != 0) {
        fprintf(stderr, "Error leyendo matrices de entrada\n");
        return EXIT_FAILURE;
    }

    /* ─────────────────────────────────────────────
     * RESERVA DE MEMORIA EN DEVICE (GPU)
     * cudaMalloc reserva en DRAM global de la GPU.
     * ───────────────────────────────────────────── */
    CUDA_CHECK(cudaMalloc((void **)&d_A, mat_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, mat_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, mat_bytes));

    /* ─────────────────────────────────────────────
     * EVENTOS CUDA PARA MEDICIÓN DE TIEMPO
     * cudaEvent_t ofrece resolución de microsegundos
     * sincronizado con el stream de la GPU.
     * ───────────────────────────────────────────── */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    /* ─────────────────────────────────────────────
     * COPIAR A Y B DE HOST A DEVICE
     * cudaMemcpy(dst, src, bytes, dirección)
     * H2D = Host to Device
     * ───────────────────────────────────────────── */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, mat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, mat_bytes, cudaMemcpyHostToDevice));

    /* ─────────────────────────────────────────────
     * CONFIGURACIÓN DE LANZAMIENTO DEL KERNEL
     *
     * blockDim: bloque 2D de TILE_SIZE × TILE_SIZE hilos
     * gridDim : cuántos bloques necesitamos para cubrir NxN
     *
     * Grid de bloques: ceil(N/TILE_SIZE) × ceil(N/TILE_SIZE)
     * Como N es múltiplo de TILE_SIZE: exactamente N/TILE_SIZE bloques
     * en cada dimensión.
     *
     * Total hilos = (N/TILE_SIZE)^2 × TILE_SIZE^2 = N^2
     * Cada hilo calcula exactamente un elemento de C.
     * ───────────────────────────────────────────── */
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(N / TILE_SIZE, N / TILE_SIZE);

    /* Registrar tiempo de inicio */
    CUDA_CHECK(cudaEventRecord(ev_start, 0));

    /* ─────────────────────────────────────────────
     * LANZAR KERNEL
     * <<<gridDim, blockDim>>> es la sintaxis CUDA C
     * para especificar la configuración de lanzamiento.
     * ───────────────────────────────────────────── */
    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    /* Verificar errores en el lanzamiento del kernel */
    CUDA_CHECK(cudaGetLastError());

    /* Esperar a que el kernel termine antes de detener el cronómetro */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Registrar tiempo de fin */
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms_gpu;
    CUDA_CHECK(cudaEventElapsedTime(&ms_gpu, ev_start, ev_stop));

    /* ─────────────────────────────────────────────
     * COPIAR RESULTADO DE DEVICE A HOST
     * D2H = Device to Host
     * ───────────────────────────────────────────── */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, mat_bytes, cudaMemcpyDeviceToHost));

    /* ─────────────────────────────────────────────
     * REFERENCIA SECUENCIAL Y VALIDACIÓN
     * ───────────────────────────────────────────── */
    matmul_sequential(h_A, h_B, h_Cseq, N);
    int ok = validate(h_Cseq, h_C, N);

    /* ─────────────────────────────────────────────
     * REPORTE
     * ───────────────────────────────────────────── */
    printf("==============================================\n");
    printf("  CUDA Matrix Multiplication — TeamNo6-D3\n");
    printf("==============================================\n");
    printf("  N          : %d\n", N);
    printf("  TILE_SIZE  : %d\n", TILE_SIZE);
    printf("  blockDim   : %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("  gridDim    : %d x %d\n", N/TILE_SIZE, N/TILE_SIZE);
    printf("  Tiempo GPU : %.4f ms\n", ms_gpu);
    printf("  Validacion : %s\n", ok ? "PASO ✓" : "FALLO ✗");
    printf("==============================================\n");

    /* Imprimir fragmento del resultado */
    if (N >= 4) {
        int i, j;
        printf("\n  C[0..3][0..3] (resultado CUDA):\n");
        for (i = 0; i < 4; i++) {
            printf("  ");
            for (j = 0; j < 4; j++)
                printf("%10.4f ", h_C[i * N + j]);
            printf("\n");
        }
    }

    /* ─────────────────────────────────────────────
     * LIBERAR MEMORIA — evitar memory leaks
     * ───────────────────────────────────────────── */

    /* Liberar en device */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    /* Liberar en host */
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_Cseq);

    /* Liberar eventos */
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    /* Resetear device (limpieza final) */
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
