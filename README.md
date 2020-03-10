# 2D-FFTLog

2D-FFTLog code for efficiently computing integrals containing two Bessel or spherical Bessel functions, in the context of transforming covariance matrices from Fourier space to real space.

The code is *independently* written and tested in python ([./python/twobessel.py](python/twobessel.py)) and C ([./C/](C)). Examples of calling the routines are given in [./C/test1.c](C/test1.c), [./C/test2.c](C/test2.c), and [./python/test.py](python/test.py). In the examples, input arrays $`k`$ and $`P(k)`$ are read in, with $`k`$ sampled logarithmically. $`P(k)`$ is set as $`f(k)`$ in the integrand of the Gaussian covariance. The code then builds a matrix with diagonal elements $`P(k)/\Delta_{\ln k}`$, and then performs 2D-FFTLog. For non-Gaussian covariance, one may read in the covariance and apply 2D-FFTLog directly.

For non-bin averaged case, the transformed covariances are evaluated at points given by array $`1/k`$. For bin-averaged case, one needs to specify bin-width in log-space, but note that the output $`r`$ array is always at bin edges.

See more details of the algorithm in [Fang et al (2020); arXiv:xxxx](https://arxiv.org/abs/xxxx), and please cite it if you find the algorithm or the code useful to your research.

Please feel free to use and adapt the code for your own purpose, and let me know if you are confused or find a bug (just open an [issue](https://github.com/xfangcosmo/2DFFTLog/issues)) or throw me an email (address shown on the profile page). 2DFFTLog is open source and distributed with the
[MIT license](https://opensource.org/licenses/mit).
