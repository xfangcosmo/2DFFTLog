# 2D-FFTLog
Xiao Fang

2D-FFTLog code for efficiently computing integrals containing two Bessel or spherical Bessel functions, in the context of transforming covariance matrices from Fourier space to real space.

_This code has been adapted and used in [**CosmoCov**](https://github.com/CosmoLike/CosmoCov)._

[-> Paper to cite](#Paper-to-Cite)

The code is *independently* written and tested in python ([./python/twobessel.py](python/twobessel.py)) and C ([./C/](C)). Examples of calling the routines are given in [./C/test1.c](C/test1.c), [./C/test2.c](C/test2.c), and [./python/test.py](python/test.py). In the examples, input arrays `k` and `P(k)` are read in, with `k` sampled logarithmically. k<sup>3</sup> P(k) is set as `f(k)` in the integrand of the Gaussian covariance. The code then builds a matrix with diagonal elements <img src="https://render.githubusercontent.com/render/math?math=f(k)/\Delta_{\ln k}">, and then performs 2D-FFTLog. For non-Gaussian covariance, one may read in the covariance and apply 2D-FFTLog directly.

For non-bin averaged case, the transformed covariances are evaluated at points given by array `1/k`. For bin-averaged case, one needs to specify bin-width in log-space, but note that the output `r` array is always at bin edges.

To run python examples, go to ([./python/](python)) directory, and run
```shell
python test.py
```
To run C examples, go to ([./C/](C)) directory, and compile with command
```shell
make tests
```
then run tests:
```shell
./test1
./test2
```

See more details of the algorithm in [Fang et al (2020); arXiv:2004.04833](https://arxiv.org/abs/2004.04833).

Please feel free to use and adapt the code for your own purpose, and let me know if you are confused or find a bug (just open an [issue](https://github.com/xfangcosmo/2DFFTLog/issues)) or throw me an email (address shown on the profile page). 2DFFTLog is open source and distributed with the
[MIT license](https://opensource.org/licenses/mit).

## Paper to Cite
Please cite the following paper if you use 2D-FFTLog in your
research:

  [X. Fang, T. Eifler, E. Krause; *2D-FFTLog: Efficient computation of
    real space covariance matrices for galaxy clustering and weak
    lensing*; arXiv:2004.04833](https://arxiv.org/abs/2004.04833)
