import numpy as np
from utils import linear_extra_P, bilinear_extra_P
from scipy.interpolate import interp1d, interp2d

def resample_fourier_gauss(k, fk, config):
	Nk = config['Nk_sample']
	k_sample = np.logspace(np.log10(k[0]), np.log10(k[-1]), num=Nk, endpoint=True)
	lnfk_interpfunc = interp1d(np.log(k), np.log(fk), kind='cubic')
	fk_sample = np.exp(lnfk_interpfunc(np.log(k_sample)))
	return k_sample, fk_sample

def resample_fourier_nongauss(k1, k2, fk1fk2, config):
	Nk1 = config['Nk1_sample']
	Nk2 = config['Nk2_sample']
	k1_sample = np.logspace(np.log10(k1[0]), np.log10(k1[-1]), num=Nk1, endpoint=True)
	k2_sample = np.logspace(np.log10(k2[0]), np.log10(k2[-1]), num=Nk2, endpoint=True)
	lnfk1k2_interpfunc = interp2d(np.log(k1),np.log(k2),np.log(fk1fk2), kind='cubic')
	fk1fk2_sample = np.exp(lnfk1k2_interpfunc(np.log(k1_sample),np.log(k2_sample)))
	return k1_sample, k2_sample, fk1fk2_sample

def extrap_fourier_gauss(k, fk, config):
	'''
	k, fk are log-sampled
	'''
	N_extrap_low = config['N_extrap_low']
	N_extrap_high = config['N_extrap_high']

	k_extrap = linear_extra_P(k, N_extrap_low, N_extrap_high)
	fk_extrap = linear_extra_P(fk, N_extrap_low, N_extrap_high)

	Delta_k = np.log(k[1]/k[0])
	fk1k2_extrap = np.diag(fk_extrap/Delta_k)
	np.savetxt('extra_file.txt', np.c_[k_extrap, fk_extrap])

	config_pre = {'pre_k1min': k[0], 'pre_k1max': k[-1],
				  'pre_k2min': k[0], 'pre_k2max': k[-1]}
	return k_extrap, k_extrap, fk1k2_extrap, config_pre

def extrap_fourier_nongauss(k1, k2, fk1k2, config):
	'''
	k1, k2, fk1k2 are log-sampled
	'''
	N_extrap_low = config['N_extrap_low']
	N_extrap_high = config['N_extrap_high']
	sym_Flag = config['sym_Flag']

	k1_extrap = linear_extra_P(k1, N_extrap_low, N_extrap_high)
	if(sym_Flag):
		k2_extrap = k1_extrap
	else:
		k2_extrap = linear_extra_P(k2, N_extrap_low, N_extrap_high)
	fk1k2_extrap = bilinear_extra_P(fk1k2, N_extrap_low, N_extrap_high)

	config_pre = {'pre_k1min': k1[0], 'pre_k1max': k1[-1],\
				  'pre_k2min': k2[0], 'pre_k2max': k2[-1]}
	return k1_extrap, k2_extrap, fk1k2_extrap, config_pre

