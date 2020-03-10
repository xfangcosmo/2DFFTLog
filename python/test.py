"""
This module tests twobessel

by Xiao Fang
Feb 22, 2020
"""

import numpy as np
from twobessel import *
from time import time
import matplotlib.pyplot as plt

print('This is a test of twobessel module written by Xiao Fang.')
print('nu is required to be between -ell to 2.')
k, pk = np.loadtxt('Pk_test', usecols=(0,1), unpack=True)

# k = log_extrap(k, 200, 200)
# pk= log_extrap(pk, 200, 200)
# k = k[::4]
# pk= pk[::4] # downsample to reduce resoluton
dlnk = np.log(k[1]/k[0])


N = k.size
print('number of input data points: '+str(N))
ell1 = 0
ell2 = 0
nu = 1.01

pk1k2 = np.diag(k**3*pk/dlnk)

################# Test two_sph_bessel ##############
mytwo_sph_bessel = two_sph_bessel(k, k, pk1k2, nu1=nu, nu2=nu, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0)

t1 = time()
r1, r2, Fr1r2 = mytwo_sph_bessel.two_sph_bessel(ell1, ell2)
t2 = time()
print('time: %fs'%(t2-t1))
print('Testing two_sph_bessel')
fig = plt.figure(figsize=(10,5))
fig.suptitle(r'$F(y_1,y_2) = \int_0^\infty\frac{dx_1}{x_1}\int_0^\infty \frac{dx_2}{x_2} f(x_1,x_2)j_{\ell_1}(x_1y_1)j_{\ell_2}(x_2y_2), \ell_1=$%.1f, $\ell_2=$%.1f'%(ell1,ell2))

subfig1 = fig.add_subplot(1,2,1)
subfig1.set_xscale('log')
subfig1.set_yscale('log')
subfig1.set_xlabel('x')
subfig1.set_ylabel('f(x)')
subfig1.plot(k, pk)
plt.tight_layout()

subfig2 = fig.add_subplot(1,2,2)
# subfig2.set_title(r'$\nu=$%.2f'%(nu))
subfig2.set_xscale('log')
subfig2.set_yscale('log')
subfig2.set_xlabel('y')
subfig2.set_ylabel(r'diag of $F(y_1,y_2)$')
subfig2.plot(r1, np.diag(Fr1r2), label='fftlog')

plt.legend()
plt.tight_layout()
plt.show()
np.savetxt("out.txt", np.c_[r1, np.diag(Fr1r2)])
# exit()

################# Test two_sph_bessel_binave ##############
t1 = time()
r1, r2, Fr1r2 = mytwo_sph_bessel.two_sph_bessel_binave(ell1, ell2, dlnk, dlnk)
t2 = time()
print('time: %fs'%(t2-t1))
print('Testing two_sph_bessel_binave')
fig = plt.figure(figsize=(10,5))
fig.suptitle(r'$F(y_1,y_2) = \int_0^\infty \frac{dx_1}{x_1}\int_0^\infty \frac{dx_2}{x_2} f(x_1,x_2)\bar{j}_{\ell_1}(x_1y_1)\bar{j}_{\ell_2}(x_2y_2), \ell_1=$%.1f, $\ell_2=$%.1f'%(ell1,ell2))

subfig1 = fig.add_subplot(1,2,1)
subfig1.set_xscale('log')
subfig1.set_yscale('log')
subfig1.set_xlabel('x')
subfig1.set_ylabel('f(x)')
subfig1.plot(k, pk)
plt.tight_layout()

subfig2 = fig.add_subplot(1,2,2)
# subfig2.set_title(r'$\nu=$%.2f'%(nu))
subfig2.set_xscale('log')
subfig2.set_yscale('log')
subfig2.set_xlabel('y')
subfig2.set_ylabel(r'diag of $F(y_1,y_2)$')
subfig2.plot(r1, np.diag(Fr1r2), label='fftlog')

plt.legend()
plt.tight_layout()
plt.show()


################# Test two_Bessel_binave ##############
mytwo_Bessel = two_Bessel(k, k, pk1k2, nu1=nu, nu2=nu, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0)
t1 = time()
r1, r2, Fr1r2 = mytwo_Bessel.two_Bessel_binave(ell1, ell2, dlnk, dlnk)
t2 = time()
print('time: %fs'%(t2-t1))
print('Testing two_Bessel_binave')
fig = plt.figure(figsize=(10,5))
fig.suptitle(r'$F(y_1,y_2) = \int_0^\infty \frac{dx_1}{x_1}\int_0^\infty \frac{dx_2}{x_2} f(x_1,x_2)\bar{J}_{\ell_1}(x_1y_1)\bar{J}_{\ell_2}(x_2y_2), \ell_1=$%.1f, $\ell_2=$%.1f'%(ell1,ell2))

subfig1 = fig.add_subplot(1,2,1)
subfig1.set_xscale('log')
subfig1.set_yscale('log')
subfig1.set_xlabel('x')
subfig1.set_ylabel('f(x)')
subfig1.plot(k, pk)
plt.tight_layout()

subfig2 = fig.add_subplot(1,2,2)
# subfig2.set_title(r'$\nu=$%.2f'%(nu))
subfig2.set_xscale('log')
subfig2.set_yscale('log')
subfig2.set_xlabel('y')
subfig2.set_ylabel(r'diag of $F(y_1,y_2)$')
subfig2.plot(r1, np.diag(Fr1r2), label='fftlog')

plt.legend()
plt.tight_layout()
plt.show()

np.savetxt("out_twoBessel_binave.txt", np.c_[r1, np.diag(Fr1r2)])