opt_home := -std=c99 -Wno-missing-braces -Wno-missing-field-initializers -I/usr/local/include -L/usr/local/lib -lgsl -lfftw3 -lgslcblas -lm -g -O3 -std=gnu99 -ffast-math -funroll-loops


tests:
	make test1
	make test2

test1:
	gcc -o test1 test1.c twobessel.c utils.c $(opt_home)

test2:
	gcc -o test2 test2.c twobessel.c utils.c $(opt_home)