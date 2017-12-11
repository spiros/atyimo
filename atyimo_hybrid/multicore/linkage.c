/* FEDERAL UNIVERSITY OF BAHIA (UFBA)
   ATYIMOLAB (www.atyimolab.ufba.br)
   University College London (UCL)
   Denaxas Lab (www.denaxaslab.org)
*/
/*
@(#)File:           $linkage.c$
@(#)Last changed:   $Date: 2017/07/01 17:54:00 $
@(#)Purpose:        AtyImo version for multicore processors
@(#)Authors:        Robespierre Pita and Clicia Pinto and Marcos Barreto and Spiros Denaxas
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define NCOL 101

void fill_matrix(int *, int , char *);
int  get_num_of_lines(FILE *);
void process_file(FILE *, int *);
void print_matrix(int *, int );
void multicore_execution(int *, int *, int , int , int , int );
float dice_multicore(int *, int *);
void print_matrix(int *matrix, int nlines);

int comparacoes = 0;

int main(int argc, char const *argv[]) {
    FILE *base_a, *base_b;
    char file1[30];
    double t1, t2;
    double quantum, leftover;
    int threads_openmp = atoi(argv[2]);
    int nlines_a, nlines_b;

    strcpy(file1, "input_");
    strcat(file1, argv[1]);
    strcat(file1, ".bloom");

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

    quantum = (float)nlines_a / threads_openmp;
	  leftover = (quantum - (int)quantum) * threads_openmp;
	  t1 = omp_get_wtime();

#pragma omp parallel num_threads(threads_openmp)
{
		int thread_id = omp_get_thread_num();
		multicore_execution(matrixA, matrixB, nlines_b, thread_id, (int) quantum, (int) leftover);
}

    t2 = omp_get_wtime();
	  int length_problem = atoi(argv[1]);
	  printf("%d\t%f\n", (length_problem), (t2 - t1));
    return 0;
}

void multicore_execution(int *matrixA, int *matrixB, int nlines_b, int thread_id, int quantum, int leftover){
	int bloomA[100], bloomB[100], inicio, fim;
    float dice;
    FILE *result[thread_id];
    char file_name[30];
    char str[10];
    sprintf(str, "%d", thread_id);

    strcpy(file_name, "results_thread");
    strcat(file_name, str);
    strcat(file_name, ".dice");

	if(thread_id < leftover) {
        inicio = thread_id * (quantum + 1);
        fim = quantum + 1 + inicio;
    }
    else {
        if (thread_id == leftover) {
            if (leftover == 0) {
                inicio = thread_id * quantum;
                fim = inicio + quantum;
            }
            else {
                inicio = thread_id * (quantum + 1);
                fim = inicio + quantum;
            }
        }
        else {
            if (leftover == 0) {
                inicio = thread_id * quantum;
                fim = inicio + quantum;
            }
            else {
                inicio = thread_id * (quantum + 1) - (thread_id - leftover);
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
			dice = dice_multicore(bloomA, bloomB);
		}
		i++;
	}
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

void process_file(FILE *fp, int *matrix) {
    char line[256];
    int pos_to_insert = 0;

    rewind(fp);
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

void print_matrix(int *matrix, int nlines) {
    int i, j;
    for (i = 0; i < nlines; i++) {
		printf("imprimindo a linha %d: \n", i);
        for (j = 0; j < NCOL; j++) {
            printf("%d", matrix[i * NCOL + j]);
        }
        printf("\n");
    }
    printf("\n");
}
