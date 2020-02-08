import numpy as np
from numpy.fft import rfft2, irfft2
from utils import g_l, c_window

def twobessel(k1, k2, fk1k2, config_pre, config):
	sym_Flag = config['sym_Flag']
	pre_k1min, pre_k1max = config_pre['pre_k1min'],config_pre['pre_k1min']
	pre_k2min, pre_k2max = config_pre['pre_k2min'],config_pre['pre_k2max']

	l1, l2 = config['l1'], config['l2']
	nu1, nu2 = config['nu1'], config['nu2']
	c_window_width = config['c_window_width']

	N1 = k1.size
	Delta_k1 = np.log(k1[1]/k1[0])
	m = np.arange(-N1//2,N1//2+1)
	eta_m = 2*np.pi/Delta_k1 / float(N1) * m

	if(sym_Flag):
		N2 = N1
		Delta_k2 = Delta_k1
		n = m
		eta_n = eta_m
	else:
		N2 = k2.size
		Delta_k2 = np.log(k2[1]/k2[0])
		n = np.arange(-N2//2,N2//2+1)
		eta_n = 2*np.pi/Delta_k2 /float(N2) * n

	z1 = nu1 + 1j*eta_m
	z2 = nu2 + 1j*eta_n
	g1 = g_l(l1,z1)
	g2 = g_l(l2,z2)


	k10, k20 = k1[0], k2[0]
	# r1 = np.pi/k1[::-1]
	# r2 = np.pi/k2[::-1]
	r1 = 1./k1[::-1]
	r2 = 1./k2[::-1]
	r10,r20= r1[0], r2[0]


	## Calculate c_mn
	P_b=((fk1k2*k2**(-nu2)).T*k1**(-nu1)).T # N1*N2 array
	c_mn=rfft2(P_b)
	# print(P_b[0])
	print(c_mn[0][1])
	# exit()
	print(c_mn.shape)

	c_mn1 = c_mn[:N1//2+1,:]
	c_mn = np.vstack((c_mn[N1//2:,:], c_mn1[:,:]))

	c_mn_left = np.conj(np.flip(np.flip(c_mn,0),1))
	c_mn = np.hstack((c_mn_left[:,:-1], c_mn))

	c_window_array1 = c_window(m, int(c_window_width*N1//2.) )
	if(sym_Flag):
		c_window_array2 = c_window_array1
	else:
		c_window_array2 = c_window(n, int(c_window_width*N2//2.) )
	c_mn_filter = ((c_mn*c_window_array2).T*c_window_array1).T

	# c_mn = c_mn_filter

	## calculate integral result as a matrix format
	mat = np.conj((c_mn*(k20*r20)**(-1j*eta_n) * g2).T * (k10*r10)**(-1j*eta_m) * g1).T
	mat_right = mat[:,N2//2:]
	mat_adjust = np.vstack((mat_right[N1//2:,:],mat_right[1:N1//2,:]))
	# print(mat_adjust[0][1])
	result = ((irfft2(mat_adjust) *np.pi / 16./ r2**nu2).T / r1**nu1).T
	# print((irfft2(mat_adjust))[0][1])
	# print((irfft2(mat_adjust))[1][0])
	return r1, r2, result
	## postprocess
	# k1_invert = k1[::-1]
	# ind1 = np.where((k1_invert>=pre_k1min) & (k1_invert<=pre_k1max) )[0]
	# print(ind1)
	# r1_out = r1[ind1]
	# if(sym_Flag):
	# 	k2_invert = k1_invert
	# 	ind2 = ind1
	# 	r2_out = r1_out
	# else:
	# 	k2_invert = k2[::-1]
	# 	ind2 = np.where((k2_invert>=pre_k2min) & (k2_invert<=pre_k2max) )[0]
	# 	r2_out = r2[ind2]

	# result_out = result[ind1[0]:ind1[-1]+1,ind2[0]:ind2[-1]+1]

	# return r1_out, r2_out, result_out