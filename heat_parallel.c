#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Macro para acceder a una celda de la matriz en memoria lineal
#define IDX(i, j, N) ((i) * (N) + (j))

// Función para inicializar la matriz con 100°C en los bordes y 0°C en el interior
void inicializar_matriz(float *matriz, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
                matriz[IDX(i, j, N)] = 100.0f; // Bordes a 100°C
            else
                matriz[IDX(i, j, N)] = 0.0f;   // Interior a 0°C
        }
    }
}

// Función que simula la difusión del calor con OpenMP
void simular_difusion(float *matriz, float *nueva, int N, float umbral, int max_iters) {
    int iter = 0;
    float cambio_max;

    double tiempo_inicio = omp_get_wtime();

    do {
        cambio_max = 0.0f;

        // Bucle paralelo con reducción para calcular el cambio máximo
        #pragma omp parallel for reduction(max:cambio_max) collapse(2)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                // Promedio de las temperaturas vecinas
                nueva[IDX(i, j, N)] = 0.25f * (
                    matriz[IDX(i - 1, j, N)] +
                    matriz[IDX(i + 1, j, N)] +
                    matriz[IDX(i, j - 1, N)] +
                    matriz[IDX(i, j + 1, N)]
                );

                float cambio_local = fabsf(nueva[IDX(i, j, N)] - matriz[IDX(i, j, N)]);
                if (cambio_local > cambio_max)
                    cambio_max = cambio_local;
            }
        }

        // Intercambio de punteros: matriz ahora apunta a nueva
        float *temp = matriz;
        matriz = nueva;
        nueva = temp;

        iter++;
    } while (cambio_max > umbral && iter < max_iters);

    double tiempo_final = omp_get_wtime();

    printf("Tiempo de simulación: %.4f segundos\n", tiempo_final - tiempo_inicio);
    printf("Iteraciones realizadas: %d\n", iter);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: %s N umbral max_iteraciones\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);             // Tamaño de la matriz
    float umbral = atof(argv[2]);      // Umbral de convergencia
    int max_iters = atoi(argv[3]);     // Máximo número de iteraciones

    float *matriz = (float *) malloc(N * N * sizeof(float));
    float *nueva  = (float *) malloc(N * N * sizeof(float));

    inicializar_matriz(matriz, N);

    simular_difusion(matriz, nueva, N, umbral, max_iters);

    free(matriz);
    free(nueva);
    return 0;
}
