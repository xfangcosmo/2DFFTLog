#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>

#include <time.h>

#include <fftw3.h>

#include "utils.h"
#include "twobessel.h"

int main(int argc, char const *argv[])
{
	config my_config;
	my_config.l1 = 0;
	my_config.l2 = 0;
	my_config.nu1 = 1.01;
	my_config.nu2 = 1.01;
	my_config.c_window_width = 0.25;
	my_config.sys_Flag = 0;

	char filename[] = "../python/Pk_test";
	FILE *IN = fopen(filename, "r");

	// double *ell, *fl;
	long Nk = 3000;
	double ell[Nk], fl[Nk];

	long linenum = 0;
	while(!feof(IN) && (linenum<Nk)) {
		fscanf(IN, "%lg %lg", &ell[linenum], &fl[linenum]);
		linenum++;
	}
	double dlnk = log(ell[1]/ell[0]);
	int i,j;
	double **fk1k2;
	fk1k2 = malloc(Nk * sizeof(double *));
	for(i=0; i<Nk; i++) {
		fk1k2[i] = malloc(Nk * sizeof(double));
		for(j=0; j<Nk; j++) {
			fk1k2[i][j] = 0.;
		}
		fk1k2[i][i] = fl[i]/dlnk;
	}

	double *r1, *r2, **result;
	r1 = malloc(Nk * sizeof(double));
	r2 = malloc(Nk * sizeof(double));
	result = malloc(Nk * sizeof(double *));
	for(i=0; i<Nk; i++) {
		result[i] = malloc(Nk * sizeof(double));
	}
	double smooth_dlnr = dlnk;
	int dimension = 2;
	clock_t start = clock();
	two_Bessel_binave(ell, ell, fk1k2, Nk, Nk, &my_config, smooth_dlnr, dimension, r1, r2, result);
	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("time:%f\n", seconds);

	// char outfilename[] = "test_output/output_binave.txt";
	// FILE *OUT = fopen(outfilename, "w");
	
	// for(i=0; i<Nk; i++) {
	// 	for(j=0; j<Nk; j++){
	// 		fprintf(OUT, "%lg %lg %lg", r1[i], r1[j], result[i][j]);
	// 		fprintf(OUT, "\n");
	// 	}
	// }
	char outfilename2[]= "test_output/output_bin_ave_diag.txt";
	FILE *OUT2 = fopen(outfilename2, "w");
	
	for(i=0; i<Nk; i++) {
		fprintf(OUT2, "%lg %lg", r1[i], result[i][i]);
		fprintf(OUT2, "\n");
	}
	// fclose(OUT);
	fclose(IN);
	fclose(OUT2);
	free(r1);
	free(r2);
	free(fk1k2);
	free(result);

	return 0;
}