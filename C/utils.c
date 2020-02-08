#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "utils.h"

double complex gamma_lanczos(double complex z) {
/* Lanczos coefficients for g = 7 */
	static double p[] = {
		0.99999999999980993227684700473478,
		676.520368121885098567009190444019,
		-1259.13921672240287047156078755283,
		771.3234287776530788486528258894,
		-176.61502916214059906584551354,
		12.507343278686904814458936853,
		-0.13857109526572011689554707,
		9.984369578019570859563e-6,
		1.50563273514931155834e-7};

	if(creal(z) < 0.5) {return M_PI / (csin(M_PI*z)*gamma_lanczos(1. - z));}
	z -= 1;
	double complex x = p[0];
	for(int n = 1; n < 9; n++){ x += p[n] / (z + (double)(n));}

	double complex t = z + 7.5;
	return sqrt(2*M_PI) * cpow(t, z+0.5) * cexp(-t) * x;
}

void g_l(double l, double nu, double *eta, double complex *gl, long N) {
/* z = nu + I*eta
Calculate g_l = zln2 + lngamma( (l+nu)/2 + I*eta/2 ) - lngamma( (3+l-nu)/2 - I*eta/2 ) */
	long i;
	double complex z;
	for(i=0; i<N; i++) {
		z = nu+I*eta[i];
		gl[i] = cexp(z*log(2.) + clog(gamma_lanczos((l+z)/2.) ) - clog(gamma_lanczos((3.+l-z)/2.)));
	}
}

void c_window_2d(double complex *out, double c_window_width, long halfN1, long halfN2) {
	// 'out' is N1*(halfN2+1) complex array
	long Ncut1, Ncut2;
	long N1;
	N1 = 2*halfN1;
	Ncut1 = (long)(halfN1 * c_window_width);
	Ncut2 = (long)(halfN2 * c_window_width);
	long i,j;
	double W;
	for(j=0; j<=Ncut2; j++) { // window for right-side
		W = (double)(j)/Ncut2 - 1./(2.*M_PI) * sin(2.*j*M_PI/Ncut2);
		for(i=0; i<N1; i++){
			out[i*(halfN2+1)+ halfN2-j] *= W;
		}
	}
	for(i=0; i<=Ncut1; i++){ // window for center part (equivalent for up-down edges when rearanged)
		W = (double)(i)/Ncut1 - 1./(2.*M_PI) * sin(2.*i*M_PI/Ncut1);
		for(j=0; j<=halfN2; j++) {
			out[(halfN1-i)*(halfN2+1) + j] *= W;
			out[(halfN1+1+i)*(halfN2+1) + j] *= W;
		}
	}
}


void mk_diag_g_to_ng(double *in, long N, double dlnk, double **out) {
	long i, j;
	for(i=0;i<N;i++) {
		for(j=0;j<N;j++) {
			out[i][j] = 0.;
		}
		out[i][i] = in[i] / dlnk;
	}
}

void extrap_log_linear(double *fk, int N_origin, int N_extra, double *large_fk) {
	double dln_left = log(fk[1]/fk[0]);
	double dln_right = log(fk[N_origin-1]/fk[N_origin-2]);
	int i;
	for(i=0; i<N_extra; i++) {
		large_fk[i] = exp(log(fk[0]) + (i - N_extra) * dln_left);
	}
	for(i=N_extra; i< N_extra+N_origin; i++) {
		large_fk[i] = fk[i - N_extra];
	}
	for(i=N_extra+N_origin; i< 2*N_extra+N_origin; i++) {
		large_fk[i] = exp(log(fk[N_origin-1]) + (i - N_extra - N_origin +1) * dln_right);
	}
}

// void resample_fourier_gauss(double *k, double *fk, config *config, double *k_sample, double *fk_sample) {
// 	long i;
// 	double dlnk = log(k[sizeof(k)-1]/k[0]) / (config->Nk_sample-1.);
// 	for(i=0; i<config->Nk_sample; i++) {
// 		k_sample[i] = k[0] * exp(i*dlnk);
// 		fk_sample[i] = 
// 	}
// }