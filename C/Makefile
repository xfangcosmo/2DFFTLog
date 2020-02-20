opt_home := -std=c99 -Wno-missing-braces -Wno-missing-field-initializers -I/usr/local/include -L/usr/local/lib -lgsl -lfftw3 -lgslcblas -lm -g -O3 -std=gnu99 -ffast-math -funroll-loops


test:
	gcc -o test_cov test_cov.c twobessel.c utils.c $(opt_home)