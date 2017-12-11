/* FEDERAL UNIVERSITY OF BAHIA (UFBA)
   ATYIMOLAB (www.atyimolab.ufba.br)
   University College London (UCL)
   Denaxas Lab (www.denaxaslab.org)
*/
/*
@(#)File:           $linkage.cu$
@(#)Version:        $v3$
@(#)Last changed:   $Date: 2017/02/10 09:05:00 $
@(#)Purpose:        AtyImo version for multiple GPU
@(#)Author:         Robespierre Pita and Clicia Pinto and Marcos Barreto and Spiros Denaxas

@(#)Usage:
 (*) Hotocompile:   make clean; make
 (*) Hotoexecute:  ./object <num_threads_per_block> <file1> <threads_openmp> <percentage_each_gpu> <qtd_gpu>
 (*) Hotoexecute:  ./linkage 64 100 32 40 2

@(#)Comment:
 (*) Pass arguments (name of file *.bloom) for command-line interface
 (*) Get time with omp_get_wtime() in seconds
 (*) Inaccurate Divide dimGrid
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <cuda.h>

#define NCOL 101

void fill_matrix(int *matrix, int pos, char *line);
int get_num_of_lines(FILE *fp);
void process_file(FILE *fp, int *matrix);
void print_matrix(int *matrix, int nlines);
int *divide(int *source_matrix, int lower_threshold, int upper_threshold);
int *get_pu_threshold(int lines, int qtd_gpu, int percentage_each_gpu);
void executa_multicore(int *matrixA, int *matrixB, int nlines_b, int id_nested, int quantum, int leftover);
float dice_multicore(int *bA, int *bB);
__global__ void kernel(int *matrixA, int *matrixB, int nlines_a, int nlines_b);
__device__ float dice(int *bloomA, int *bloomB);

int main(int argc, char const *argv[]) {
    FILE *base_a, *base_b;
    double t1, t2;

    int nlines_a = 0, nlines_b = 0, lower_threshold, upper_threshold, *pu_threshold;

    char file1[30];
    strcpy(file1, "input_");
    strcat(file1, argv[2]);
    strcat(file1, ".bloom");

    int threads_per_block = atoi(argv[1]);
    int threads_openmp = atoi(argv[3]);
    int percentage_each_gpu = atoi(argv[4]);
    int qtd_gpu = atoi(argv[5]);

    // --------------------- START: opening and loading files A and B --------------------- //
    base_a = fopen(file1, "r");
    base_b = fopen(file1, "r");

    nlines_a = get_num_of_lines(base_a);
    int *matrixA = (int *)malloc(nlines_a * NCOL * sizeof(int));
    process_file(base_a, matrixA);
    // print_matrix(matrixA, nlines_a);

    nlines_b = get_num_of_lines(base_b);
    int *matrixB = (int *)malloc(nlines_b * NCOL * sizeof(int));
    process_file(base_b, matrixB);
    // print_matrix(matrixB, nlines_b);
    // --------------------- END: opening and loading files A and B --------------------- //

    pu_threshold = get_pu_threshold(nlines_a, qtd_gpu, percentage_each_gpu);
    t1 = omp_get_wtime();
    omp_set_nested(1);
    #pragma omp parallel num_threads(qtd_gpu+1) private(lower_threshold, upper_threshold)
    {
        int id;
        id = omp_get_thread_num();

        if((id == 0) && (pu_threshold[0]!= -1)){
            double quantum, leftover;
            lower_threshold = pu_threshold[id * 2];
            upper_threshold = pu_threshold[(id * 2)+1];
            quantum = (float)upper_threshold / threads_openmp;
            leftover = (quantum - (int)quantum) * threads_openmp;
            int *matrixA_tmp;
            matrixA_tmp = divide(matrixA, lower_threshold, upper_threshold);
            #pragma omp parallel num_threads(threads_openmp)
            {
                int id_nested;
                id_nested = omp_get_thread_num();
                executa_multicore(matrixA_tmp, matrixB, nlines_b, id_nested, (int) quantum, (int) leftover);
            }
        }

        else if((id != 0) && (pu_threshold[2]!= -1)){
            int gpu_id = -1;
            cudaSetDevice(id);
            cudaGetDevice(&gpu_id);
            int *matrixA_d, *matrixB_d, *matrixA_tmp;
            lower_threshold = pu_threshold[id * 2];
            upper_threshold = pu_threshold[(id * 2)+1];
            matrixA_tmp = divide(matrixA, lower_threshold, upper_threshold);
            cudaMalloc((int **)&matrixA_d, (upper_threshold - lower_threshold) * NCOL * sizeof(int));
            cudaMemcpy(matrixA_d, matrixA_tmp, (upper_threshold - lower_threshold) * NCOL * sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((int **)&matrixB_d, nlines_b * NCOL * sizeof(int));
            cudaMemcpy(matrixB_d, matrixB, nlines_b * NCOL * sizeof(int), cudaMemcpyHostToDevice);
            dim3 dimGrid = (int) ceil( (int) (upper_threshold - lower_threshold) / (int) threads_per_block);
            dim3 dimBlock = threads_per_block;
            kernel<<<dimGrid, dimBlock>>>(matrixA_d, matrixB_d, (upper_threshold - lower_threshold), nlines_b);
            cudaDeviceSynchronize();
            cudaFree(matrixA_d);
            cudaFree(matrixB_d);
        }
    }

    t2 = omp_get_wtime();
    free(matrixA);
    free(matrixB);
    fclose(base_a);
    fclose(base_b);

    int length_problem = atoi(argv[2]);
    printf("%d\t%f\n", (length_problem * 1000), (t2-t1));

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

// function to indicate the upper and lower thresholds for each GPU and CPU
// according to the number of lines in matrixA
int *get_pu_threshold(int problem_size, int qtd_gpu, int percentage_each_gpu) {
    static int *threshold_vector;
    int i, border = 0;
    float quantum_cpu, quantum_gpu, leftover = 0.0;
    int percentage_cpu = 100 - (percentage_each_gpu * qtd_gpu);
    threshold_vector = (int *)malloc((2 + (qtd_gpu * 2)) * sizeof(int));

    quantum_cpu = (percentage_cpu/100.0)*problem_size;
    quantum_gpu = (percentage_each_gpu/100.0)*problem_size;
    leftover = quantum_cpu - ((int)quantum_cpu);
    leftover += (quantum_gpu - ((int)quantum_gpu))*2.0;

    if (percentage_cpu == 0) {
        threshold_vector[0] = -1;
        threshold_vector[1] = -1;
    }
    else {
        threshold_vector[0] = border;
        border = border+(int)quantum_cpu;
	if ((int)leftover != 0){
		border++;
		leftover-= 1.0;
	}
        threshold_vector[1] = border;
    }


    for (i = 2; i < (2 + qtd_gpu * 2); i = i + 2) {
        if(percentage_each_gpu == 0){
                border = -1;
        }

        threshold_vector[i] = border;
	if ((int)leftover != 0){
                border++;
                leftover-= 1.0;
	}
        if(percentage_each_gpu == 0){
                border = -1;
        }

        border = border+(int)quantum_gpu;
        threshold_vector[i + 1] = border;
    }

    return threshold_vector;
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

void executa_multicore(int *matrixA, int *matrixB, int nlines_b, int id_nested, int quantum, int leftover){
	int bloomA[100], bloomB[100], inicio, fim;
	if(id_nested < leftover){
		inicio = id_nested * (quantum +1);
		fim = quantum + 1 + inicio;
	}
	else{
		if(id_nested ==leftover){
			if(leftover == 0){
				inicio = id_nested * quantum;
				fim = inicio + quantum;
			}
			else{
				inicio = id_nested * (quantum +1);
				fim = inicio + quantum;
			}
		}
		else{
			if(leftover==0){
				inicio = id_nested * (quantum);
				fim = inicio + quantum;
			}
			else{
				inicio = id_nested * (quantum+1) - (id_nested-leftover);
				fim = inicio + quantum;
			}
		}
	}

    int i = inicio;
    while (i < fim) {
        for (int j = 1; j < 101; j++) {
            bloomA[j - 1] = matrixA[i * NCOL + j];
        }
        // getting bloom filter for each matrixB register
        for (int k = 0; k < nlines_b; k++) {
            for (int l = 1; l < 101; l++) {
                bloomB[l - 1] = matrixB[k * NCOL + l];
            }
            dice_multicore(bloomA, bloomB);
        }
        i++;
    }
}

// CUDA kernel to compute linkage between matrixA and matrixB using a Dice
// function as similarity measure
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

// device function to calculate the Dice coefficient using Bloom filter
__device__ float dice(int *bloomA, int *bloomB) {
    double a = 0, b = 0, h = 0;
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
    double dice = ((h * 2.0) / (a + b)) * 10000;
    // printf("%.1f\n", dice);
    return dice;
}

float dice_multicore(int *bloomA, int *bloomB) {
    double a = 0, b = 0, h = 0;
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
    double dice = ((h * 2.0) / (a + b)) * 10000;
    return dice;
}
