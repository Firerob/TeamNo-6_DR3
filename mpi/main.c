/*
 * ============================================================
 * TeamNo6-D3 — MPI Matrix Multiplication
 * ============================================================
 * Proyecto universitario: Multiplicación de matrices NxN en
 * paralelo usando MPI con tiling local, operaciones colectivas
 * (MPI_Bcast, MPI_Scatter, MPI_Gather) y referencia secuencial
 * (golden reference) para validación de correctitud.
 *
 * Compilar : mpicc -O2 -o matmul_mpi main.c -lm
 * Ejecutar : mpirun -np 4 ./matmul_mpi < input.txt
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
#include <mpi.h>

/* ──────────────────────────────────────────────
 * MACRO: tamaño del tile para la multiplicación
 * local en cada proceso. Debe ser potencia de 2.
 * ────────────────────────────────────────────── */
#define TILE_SIZE 32

/* Tolerancia para comparación de flotantes */
#define EPSILON 1e-3

/* ============================================================
 * SECCIÓN 1 — PARSER DE ENTRADA
 * Lee desde stdin: dimensión, datos de A, datos de B.
 * Usa fgets + strtok para evitar buffer overflows.
 * ============================================================ */

/*
 * read_int_line: lee una línea de stdin y la convierte a int.
 * Retorna el entero leído o -1 en caso de error.
 */
static int read_int_line(void)
{
    char buf[64];
    if (fgets(buf, sizeof(buf), stdin) == NULL) return -1;
    return atoi(buf);
}

/*
 * read_matrix_csv: lee N*N valores flotantes separados por coma
 * desde una única línea de stdin y los almacena en 'mat' (row-major).
 * Retorna 0 en éxito, -1 en error.
 */
static int read_matrix_csv(double *mat, int N)
{
    /* Buffer dinámico: worst case ~20 chars por número */
    size_t buf_size = (size_t)N * N * 24 + 4;
    char *buf = (char *)malloc(buf_size);
    if (!buf) { fprintf(stderr, "read_matrix_csv: malloc fallo\n"); return -1; }

    if (fgets(buf, (int)buf_size, stdin) == NULL) {
        free(buf);
        return -1;
    }

    /* Eliminar salto de línea final */
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
        fprintf(stderr, "read_matrix_csv: se esperaban %d valores, se leyeron %d\n", N * N, idx);
        return -1;
    }
    return 0;
}

/* ============================================================
 * SECCIÓN 2 — MULTIPLICACIÓN SECUENCIAL (GOLDEN REFERENCE)
 * C = A * B en O(N^3). Almacenamiento row-major.
 * Se usa como referencia para validar los resultados paralelos.
 * ============================================================ */

/*
 * matmul_sequential: multiplica A[N×N] por B[N×N] y guarda en C[N×N].
 * Todos los arreglos están en formato row-major: elemento (i,j) → arr[i*N+j].
 */
static void matmul_sequential(const double *A, const double *B, double *C, int N)
{
    int i, j, k;
    /* Inicializar C a cero */
    memset(C, 0, sizeof(double) * N * N);

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/* ============================================================
 * SECCIÓN 3 — MULTIPLICACIÓN CON TILING LOCAL
 *
 * El tiling divide las matrices en bloques (tiles) de TILE_SIZE×TILE_SIZE.
 * Ventajas de cache:
 *   - Cada tile cabe en caché L1/L2 → menos cache misses.
 *   - Reutilización espacial y temporal de datos.
 *
 * Algoritmo:
 *   Para cada bloque (bi, bj) de C, iterar sobre los bloques
 *   de columnas de A / filas de B (índice bk), acumular la
 *   contribución parcial del tile A[bi][bk] * B[bk][bj].
 * ============================================================ */

/*
 * matmul_tiled: versión con tiling de la multiplicación A * B = C.
 * 'rows_A': número de filas de A que este proceso maneja
 *           (puede ser menor que N para el proceso root).
 * 'N'     : dimensión completa (columnas de A = filas de B = columnas de B).
 */
static void matmul_tiled(const double *A, const double *B, double *C,
                          int rows_A, int N)
{
    int i, j, k;
    int ii, jj, kk;

    /* Inicializar resultado parcial a cero */
    memset(C, 0, sizeof(double) * rows_A * N);

    /*
     * Bucle externo por tiles:
     *   ii → bloque de filas    de A (y C)
     *   jj → bloque de columnas de B (y C)
     *   kk → bloque de columnas de A / filas de B
     */
    for (ii = 0; ii < rows_A; ii += TILE_SIZE) {
        for (jj = 0; jj < N; jj += TILE_SIZE) {
            for (kk = 0; kk < N; kk += TILE_SIZE) {

                /* ── Límites del tile actual ── */
                int i_end = ii + TILE_SIZE < rows_A ? ii + TILE_SIZE : rows_A;
                int j_end = jj + TILE_SIZE < N     ? jj + TILE_SIZE : N;
                int k_end = kk + TILE_SIZE < N     ? kk + TILE_SIZE : N;

                /* ── Multiplicación dentro del tile ── */
                for (i = ii; i < i_end; i++) {
                    for (k = kk; k < k_end; k++) {
                        double a_ik = A[i * N + k]; /* cargar A una vez */
                        for (j = jj; j < j_end; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

/* ============================================================
 * SECCIÓN 4 — VALIDACIÓN (golden reference vs resultado MPI)
 *
 * Compara elemento a elemento C_seq y C_mpi con tolerancia EPSILON.
 * Retorna 1 si son equivalentes, 0 si difieren.
 * ============================================================ */
static int validate(const double *C_seq, const double *C_mpi, int N)
{
    int i;
    for (i = 0; i < N * N; i++) {
        double diff = fabs(C_seq[i] - C_mpi[i]);
        if (diff > EPSILON) {
            fprintf(stderr,
                    "VALIDACION FALLO en indice [%d]: seq=%.6f mpi=%.6f diff=%.2e\n",
                    i, C_seq[i], C_mpi[i], diff);
            return 0;
        }
    }
    return 1;
}

/* ============================================================
 * SECCIÓN 5 — PROGRAMA PRINCIPAL MPI
 * ============================================================ */
int main(int argc, char *argv[])
{
    int rank, size;
    int N = 0;

    /* ── Matrices globales (solo en root) y resultado final ── */
    double *A       = NULL;  /* Matriz A completa, solo root */
    double *B       = NULL;  /* Matriz B completa, todos los procesos */
    double *C_mpi   = NULL;  /* Resultado MPI reconstruido, solo root */
    double *C_seq   = NULL;  /* Resultado secuencial, solo root */

    /* ── Buffers locales de cada proceso ── */
    double *local_A = NULL;  /* Filas de A que corresponden a este proceso */
    double *local_C = NULL;  /* Filas de C calculadas por este proceso */

    /* ────────────────────────────────────────
     * INICIALIZACIÓN MPI
     * MPI_Init configura el entorno de comunicación.
     * MPI_Comm_rank → identificador del proceso (0 = root).
     * MPI_Comm_size → número total de procesos.
     * ──────────────────────────────────────── */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ────────────────────────────────────────
     * LECTURA DE DATOS (solo proceso root=0)
     * Solo el proceso 0 tiene acceso a stdin.
     * Los demás procesos esperan en MPI_Bcast.
     * ──────────────────────────────────────── */
    if (rank == 0) {
        N = read_int_line();
        if (N <= 0) {
            fprintf(stderr, "Error: dimension invalida N=%d\n", N);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (N % size != 0) {
            fprintf(stderr,
                    "Error: N=%d no es divisible entre %d procesos\n", N, size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Reservar matrices A, B, C_mpi, C_seq */
        A     = (double *)malloc(sizeof(double) * N * N);
        B     = (double *)malloc(sizeof(double) * N * N);
        C_mpi = (double *)malloc(sizeof(double) * N * N);
        C_seq = (double *)malloc(sizeof(double) * N * N);

        if (!A || !B || !C_mpi || !C_seq) {
            fprintf(stderr, "root: malloc fallo para matrices globales\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (read_matrix_csv(A, N) != 0 || read_matrix_csv(B, N) != 0) {
            fprintf(stderr, "root: error leyendo matrices\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* ────────────────────────────────────────
     * MPI_Bcast: DIFUNDIR DIMENSIÓN N
     * root envía N a todos los procesos.
     * Todos los procesos (incluyendo root) participan.
     * Después de esta llamada, todos conocen N.
     * ──────────────────────────────────────── */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Filas de A que le corresponden a cada proceso */
    int rows_per_proc = N / size;

    /* Reservar B en todos los procesos (se llenará con Bcast) */
    if (rank != 0) {
        B = (double *)malloc(sizeof(double) * N * N);
        if (!B) {
            fprintf(stderr, "rank %d: malloc fallo para B\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Reservar buffers locales */
    local_A = (double *)malloc(sizeof(double) * rows_per_proc * N);
    local_C = (double *)malloc(sizeof(double) * rows_per_proc * N);
    if (!local_A || !local_C) {
        fprintf(stderr, "rank %d: malloc fallo para buffers locales\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ────────────────────────────────────────
     * INICIO MEDICIÓN DE TIEMPO
     * MPI_Wtime() devuelve tiempo en segundos
     * con resolución de microsegundos.
     * ──────────────────────────────────────── */
    double t_start = MPI_Wtime();

    /* ────────────────────────────────────────
     * MPI_Bcast: DIFUNDIR MATRIZ B COMPLETA
     * root envía B (N*N doubles) a todos.
     * Operación colectiva: todos los procesos
     * reciben la misma copia de B.
     * Ventaja: un solo mensaje de root a todos,
     * el runtime MPI usa árbol binario interno
     * reduciendo la latencia respecto a N envíos
     * punto a punto.
     * ──────────────────────────────────────── */
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ────────────────────────────────────────
     * MPI_Scatter: DISTRIBUIR FILAS DE A
     * root divide A en bloques de rows_per_proc*N
     * doubles y envía uno a cada proceso.
     * Operación colectiva: root envía fragmentos
     * distintos a cada proceso.
     * Ventaja: comunicación estructurada y
     * eficiente; el runtime puede solapar envíos.
     * ──────────────────────────────────────── */
    MPI_Scatter(
        A,                          /* buffer de envío (solo root) */
        rows_per_proc * N,          /* count por proceso */
        MPI_DOUBLE,
        local_A,                    /* buffer de recepción local */
        rows_per_proc * N,
        MPI_DOUBLE,
        0,                          /* root */
        MPI_COMM_WORLD
    );

    /* ────────────────────────────────────────
     * CÓMPUTO LOCAL CON TILING
     * Cada proceso multiplica sus 'rows_per_proc'
     * filas de A contra la matriz B completa.
     * El tiling mejora la localidad de caché:
     * - Se carga un tile de A (rows×TILE_SIZE)
     * - Se carga un tile de B (TILE_SIZE×cols)
     * - Ambos caben en caché L1/L2
     * - Se reduce el número de cache misses
     * ──────────────────────────────────────── */
    matmul_tiled(local_A, B, local_C, rows_per_proc, N);

    /* ────────────────────────────────────────
     * MPI_Gather: RECOLECTAR RESULTADOS EN ROOT
     * Cada proceso envía su local_C a root.
     * root ensambla todos los fragmentos en C_mpi.
     * Operación colectiva inversa a Scatter.
     * ──────────────────────────────────────── */
    MPI_Gather(
        local_C,                    /* buffer de envío local */
        rows_per_proc * N,
        MPI_DOUBLE,
        C_mpi,                      /* buffer de recepción (solo root) */
        rows_per_proc * N,
        MPI_DOUBLE,
        0,                          /* root */
        MPI_COMM_WORLD
    );

    /* ────────────────────────────────────────
     * FIN MEDICIÓN DE TIEMPO
     * ──────────────────────────────────────── */
    double t_end = MPI_Wtime();
    double t_elapsed = t_end - t_start;

    /* ────────────────────────────────────────
     * VALIDACIÓN Y REPORTE (solo root)
     * ──────────────────────────────────────── */
    if (rank == 0) {
        printf("==============================================\n");
        printf("  MPI Matrix Multiplication — TeamNo6-D3\n");
        printf("==============================================\n");
        printf("  N          : %d\n", N);
        printf("  Procesos   : %d\n", size);
        printf("  Filas/proc : %d\n", rows_per_proc);
        printf("  TILE_SIZE  : %d\n", TILE_SIZE);
        printf("  Tiempo MPI : %.6f segundos\n", t_elapsed);

        /* Calcular referencia secuencial */
        matmul_sequential(A, B, C_seq, N);

        /* Comparar MPI vs secuencial */
        int ok = validate(C_seq, C_mpi, N);
        if (ok)
            printf("  Validacion : PASO ✓\n");
        else
            printf("  Validacion : FALLO ✗\n");
        printf("==============================================\n");

        /* Imprimir fragmento del resultado (primeros 4×4) */
        if (N >= 4) {
            int i, j;
            printf("\n  C[0..3][0..3] (resultado MPI):\n");
            for (i = 0; i < 4; i++) {
                printf("  ");
                for (j = 0; j < 4; j++)
                    printf("%10.4f ", C_mpi[i * N + j]);
                printf("\n");
            }
        }

        /* Liberar memoria del root */
        free(A);
        free(C_mpi);
        free(C_seq);
    }

    /* Liberar memoria de todos los procesos */
    free(B);
    free(local_A);
    free(local_C);

    /* ── Finalizar MPI ── */
    MPI_Finalize();
    return 0;
}