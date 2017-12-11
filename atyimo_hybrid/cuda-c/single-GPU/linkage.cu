/* FEDERAL UNIVERSITY OF BAHIA (UFBA)
   ATYIMOLAB (www.atyimolab.ufba.br)
   University College London (UCL)
   Denaxas Lab (www.denaxaslab.org)
*/
/*
@(#)File:           $linkage.cu$
@(#)Last changed:   $Date: 2017/07/01 17:54:00 $
@(#)Purpose:        AtyImo version for single GPU
@(#)Author:         Robespierre Pita and Clicia Pinto and Marcos Barreto and Spiros Denaxas

@(#)Comment:
 (*) Pass arguments (name of *.bloom file) for command-line interface
 (*) Get time with omp_get_wtime() in seconds
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
#include <omp.h>

#define NCOL 101

__device__ int contador = 0;

void fill_matrix(int *matrix, int pos, char *line);
void process_file(FILE *fp, int *matrix);
void print_matrix(int *matrix, int nlines);
int get_num_of_lines(FILE *fp);
int *divide(int *source_matrix, int lower_threshold, int upper_threshold);
__global__ void kernel(int *matrixA, int *matrixB, int nlines_a, int nlines_b);
__device__ float dice(int *bloomA, int *bloomB);

int main(int argc, char const *argv[]) {
    double t1, t2;
    t1 = omp_get_wtime();

    FILE *base_a, *base_b;
    char file1[30];
    strcpy(file1, "input_");
    strcat(file1, argv[2]);
    strcat(file1, ".bloom");

    int nlines_a = 0, nlines_b = 0;
    int threads_per_block = atoi(argv[1]);

    base_a = fopen(file1, "r");
    base_b = fopen(file1, "r");

    // --------------------- OPERATIONS WITH BASE A --------------------- //
    // getting the number of lines
    nlines_a = get_num_of_lines(base_a);
    int *matrixA = (int *)malloc(nlines_a * NCOL * sizeof(int));

    printf("[PROCESSING BASE A ... ]\n");
    process_file(base_a, matrixA);
    // print_matrix(matrixA, nlines_a);

    // --------------------- OPERATIONS WITH BASE B --------------------- //
    nlines_b = get_num_of_lines(base_b);
    int *matrixB = (int *)malloc(nlines_b * NCOL * sizeof(int));

    // processing base_b to fill matrixB
    process_file(base_b, matrixB);

    // ------------------------- CUDA OPERATIONS ------------------------ //
    int *matrixA_d, *matrixB_d;

    // allocating device memory
    cudaMalloc((int **)&matrixA_d, nlines_a * NCOL * sizeof(int));
    cudaMalloc((int **)&matrixB_d, nlines_b * NCOL * sizeof(int));

    // copying host memory to device memory
    cudaMemcpy(matrixA_d, matrixA, nlines_a * NCOL * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_d, matrixB, nlines_b * NCOL * sizeof(int), cudaMemcpyHostToDevice);

    // kernel operations
    dim3 dimGrid = (int) ceil( (int) nlines_a / (int) threads_per_block);
    dim3 dimBlock = threads_per_block;
    kernel<<<dimGrid, dimBlock>>>(matrixA_d, matrixB_d, nlines_a, nlines_b);

    cudaDeviceSynchronize();

    // releasing device memory
    cudaFree(matrixA_d);
    cudaFree(matrixB_d);

    free(matrixA);
    free(matrixB);

    // closing files
    fclose(base_a);
    fclose(base_b);

    t2 = omp_get_wtime();

    int length_problem = atoi(argv[2]);
    printf("%d\t%f\n", (length_problem ), (t2-t1));

    return 0;
}

// return the number of lines for a given file
int get_num_of_lines(FILE *fp) {
    int lines = 0;
    char line[256];
    if(!fgets(line, 255, fp))
        printf("fp = NULL");

    while (!feof(fp)) {
        lines++;
        if(!fgets(line, 255, fp))
            break;
    }

    return lines;
}

// return line by line for a given file
void process_file(FILE *fp, int *matrix) {
    char line[256];
    int pos_to_insert = 0;

    rewind(fp);

    // insert each line into the matrix
    if(!fgets(line, 255, fp))
        printf("fp = NULL");
    while (!feof(fp)) {
        line[strlen(line) - 1] = '\0';
        fill_matrix(matrix, pos_to_insert, line);

        pos_to_insert++;
        if(!fgets(line, 255, fp))
            break;
    }
}

// function to split a line and insert the elements into the matrix
void fill_matrix(int *matrix, int pos, char *line) {
    int i = 0, j = 0;
    char c, id[10];

    do {
        c = line[j];
        id[j] = c;
        j++;
    } while (c != ';');
    id[j-1] = '\0';
    matrix[NCOL * pos] = atoi(id);

    for (i = 1; i < NCOL; i++) {
        matrix[NCOL * pos + i] = line[j] - '0';
        j++;
    }
}

// function to divide matrixA into smaller matrices, given a lower threshold
// and a upper threshold. Each one will be executed on a GPU
int *divide(int *source_matrix, int lower_threshold, int upper_threshold) {
    static int *destination_matrix;
    destination_matrix = (int *)malloc((upper_threshold - lower_threshold) * NCOL * sizeof(int));

    int i, j = 0;

    for (i = (lower_threshold * NCOL); i < (upper_threshold * NCOL); i++) {
        destination_matrix[j] = source_matrix[i];
        j++;
    }

    return destination_matrix;
}

void print_matrix(int *matrix, int nlines) {
    int i, j;

    for (i = 0; i < nlines; i++) {
        for (j = 0; j < NCOL; j++) {
            printf("%d", matrix[i * NCOL + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// CUDA kernel to compute linkage between matrixA and matrixB using Dice
__global__ void kernel(int *matrixA, int *matrixB, int nlines_a, int nlines_b){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int bloomA[100], bloomB[100];

    if (i < nlines_a) {

        // getting Bloom filter for each matrixA register
        for (int j = 1; j < 101; j++) {
            bloomA[j - 1] = matrixA[i * NCOL + j];
        }

        // getting Bloom filter for each matrixB register
        for (int k = 0; k < nlines_b; k++) {
            for (int l = 1; l < 101; l++) {
                bloomB[l - 1] = matrixB[k * NCOL + l];
            }
            dice(bloomA, bloomB);
        }
    }
}

// device function to calculate Dice coefficient using Bloom filter
__device__ float dice(int *bloomA, int *bloomB) {
    float a = 0, b = 0, h = 0;
    int i;

    for (i = 0; i < 100; i++) {
        if (bloomA[i] == 1) {
            a++;
            if (bloomB[i] == 1)
                h++;
        }
        if (bloomB[i] == 1) {
            b++;
        }
    }
    float dice = ((h * 2.0) / (a + b)) * 10000;
    return dice;
}
