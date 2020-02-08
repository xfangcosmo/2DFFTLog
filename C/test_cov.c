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
	my_config.l1 = -0.5;
	my_config.l2 = -0.5;
	my_config.nu1 = 1.7;
	my_config.nu2 = 1.7;
	my_config.c_window_width = 0.25;
	my_config.sys_Flag = 0;

	char filename[] = "extra_file.txt";
	FILE *IN = fopen(filename, "r");

	// double *ell, *fl;
	long Nk = 3200;
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

	clock_t start = clock();
	twobessel(ell, ell, fk1k2, Nk, Nk, &my_config, r1, r2, result);
	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("time:%f\n", seconds);

	char outfilename[] = "test_cov_output/newoutput.txt";
	FILE *OUT = fopen(outfilename, "w");
	
	for(i=0; i<Nk; i++) {
		fprintf(OUT, "%lg %lg", r1[i], result[i][i]/2./pow(M_PI,3)/5000.*41253.*r1[i]);
		fprintf(OUT, "\n");
	}
	fclose(OUT);
	fclose(IN);
	free(r1);
	free(r2);
	free(fk1k2);
	free(result);

	return 0;
}