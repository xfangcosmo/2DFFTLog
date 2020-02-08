import numpy as np
from preprocess import *
from twobessel import twobessel
from scipy.interpolate import interp2d

def resample_ell_fl(ell, fl, config_output):
	config = {'Nk_sample': config_output['Nell_sample']}
	return resample_fourier_gauss(ell, fl, config)

def resample_l1l2_fl1l2(ell1, ell2, fl1l2, config_output):
	config = {'Nk1_sample': config_output['Nell1_sample'], 'Nk2_sample': config_output['Nell2_sample']}
	return resample_fourier_nongauss(ell1, ell2, fl1fl2, config)



## General forms
def cov_G_real_noPureN(ell,fl, dav1, dav2, config_output):
	'''
	fl: 1dim cov_Fourier
	dav1, dav2:
		0: xi_+
		1: xi_-
		2: gamma_t
		3: w
	'''

	#initialize config
	config = {'N_extrap_low': 800, 'N_extrap_high': 800,\
			  'sym_Flag': False, \
			  'l1': 0, 'l2': 0,\
			  'nu1': 1.5, 'nu2': 1.5, 'c_window_width': 0.25}

	# convert from arcmin to radians
	out_thetamin, out_thetamax = config_output['out_thetamin']/60./180.*np.pi,config_output['out_thetamax']/60./180.*np.pi
	
	out_Ntheta = config_output['out_Ntheta']
	out_theta = np.logspace(np.log10(out_thetamin),np.log10(out_thetamax), num=out_Ntheta+1, endpoint=True)

	config['c_window_width'] = config_output['c_window_width']
	f_sky = config_output['f_sky']/41253. # convert from deg squared to steradian

	ell1_extrap, ell2_extrap, fl1l2_extrap, config_pre = extrap_fourier_gauss(ell,fl,config)

	# if(z1!=z2 or pm1!=pm2): config['sym_Flag']=False
	davs = {0: 0, 1: 4, 2: 2, 3: 0}
	config['l1'] = davs[dav1] - 0.5
	config['l2'] = davs[dav2] - 0.5

	# config['nu1'] = 1. - config['l1']/2.
	# config['nu2'] = 1. - config['l2']/2.

	config['nu1'] = 1.
	config['nu2'] = 1.

	# config['nu1'] = 1.7
	# config['nu2'] = -0.3
	# config['nu2'] = 1.7
	# config['nu1'] = 1.45 - config['l1']/2.
	# config['nu1'] = 1.5
	# config['nu2'] = 3.5

	theta1_out, theta2_out, result_out = twobessel(ell1_extrap, ell2_extrap, fl1l2_extrap, config_pre, config)

	# import matplotlib.pyplot as plt
	# plt.plot(theta1_out, np.diag(result_out))
	# plt.plot(theta1_out, -np.diag(result_out), '--')
	# plt.xscale('log')
	# plt.yscale('log')
	# plt.ylim(1e-22,1e-3)
	# plt.show()
	# exit()

	cov_interpfunc = interp2d(np.log(theta1_out), np.log(theta2_out), result_out, kind='linear')
	cov_interp = cov_interpfunc(np.log(out_theta), np.log(out_theta))

	cov_out = cov_interp /4./np.pi**3/f_sky
	cov_out = (cov_out.T * np.sqrt(out_theta)).T * np.sqrt(out_theta)
	return out_theta, cov_out
	# result_out /= (4.*np.pi**3 *f_sky)
	# return theta1_out, (result_out.T * np.sqrt(theta1_out)).T * np.sqrt(theta1_out)

def cov_NG_real(ell1,ell2,fl1l2, dav1, dav2, config_output):
	'''
	fl1l2: 2dim cov_Fourier
	dav1, dav2:
		0: xi_+
		1: xi_-
		2: gamma_t
		3: w
	'''

	#initialize config
	config = {'N_extrap_low': 1000, 'N_extrap_high': 1000,\
			  'sym_Flag': False, \
			  'l1': 0, 'l2': 0,\
			  'nu1': 1.5, 'nu2': 1.5, 'c_window_width': 0.25}

	out_thetamin, out_thetamax = config_output['out_thetamin']/60./180.*np.pi,config_output['out_thetamax']/60./180.*np.pi
	out_Ntheta = config_output['out_Ntheta']
	out_theta = np.logspace(np.log10(out_thetamin),np.log10(out_thetamax), num=out_Ntheta, endpoint=True)

	config['c_window_width'] = config_output['c_window_width']
	f_sky = config_output['f_sky']/41253.

	ell1_extrap, ell2_extrap, fl1l2_extrap, config_pre = extrap_fourier_nongauss(ell1,ell2,fl1l2,config)

	# if(z1!=z2 or pm1!=pm2): config['sym_Flag']=False
	davs = {0: 0, 1: 4, 2: 2, 3: 0}
	config['l1'] = davs[dav1] - 0.5
	config['l2'] = davs[dav2] - 0.5

	config['nu1'] = 1.45 - config['l1']/2.
	config['nu2'] = 1.45 - config['l2']/2.

	theta1_out, theta2_out, result_out = twobessel(ell1_extrap, ell2_extrap, fl1l2_extrap, config_pre, config)


	cov_interpfunc = interp2d(np.log(theta1_out), np.log(theta2_out), result_out, kind='linear')
	cov_interp = cov_interpfunc(np.log(out_theta), np.log(out_theta))

	cov_out = cov_interp /2./np.pi**3
	cov_out = (cov_out.T * np.sqrt(out_theta)).T * np.sqrt(out_theta)
	return out_theta, cov_out

########################
## Individual covariance pieces

##### shear-shear
def cov_G_shear_shear_real_noPureN(ell,fl, pm1, pm2, config_output):
	'''
	fl: 1dim cov_Fourier
	(pm1,pm2)=(1,1): ++, (1,0): +-, (0,0): --
	'''
	dav1 = (pm1 + 1)%2
	dav2 = (pm2 + 1)%2
	return cov_G_real_noPureN(ell,fl, dav1, dav2, config_output)

def cov_NG_shear_shear_real(ell1,ell2,fl1l2, pm1, pm2, config_output):
	'''
	fl1l2: 2dim cov_Fourier
	(pm1,pm2)=(1,1): ++, (1,0): +-, (0,0): --
	'''
	dav1 = (pm1 + 1)%2
	dav2 = (pm2 + 1)%2
	return cov_NG_real(ell1,ell2,fl1l2, dav1, dav2, config_output)


##### gl-shear
def cov_G_gl_shear_real_noPureN(ell,fl, pm, config_output):
	'''
	fl: 1dim cov_Fourier
	pm = 1:+, 0:-
	'''
	dav1 = 2
	dav2 = (pm + 1)%2
	return cov_G_real_noPureN(ell,fl, dav1, dav2, config_output)

def cov_NG_gl_shear_real(ell1,ell2,fl1l2, pm, config_output):
	'''
	fl1l2: 2dim cov_Fourier
	pm = 1:+, 0:-
	'''
	dav1 = 2
	dav2 = (pm + 1)%2
	return cov_NG_real(ell1,ell2,fl1l2, dav1, dav2, config_output)



##### cl-shear
def cov_G_cl_shear_real_noPureN(ell,fl, pm, config_output):
	'''
	fl: 1dim cov_Fourier
	pm = 1:+, 0:-
	'''
	dav1 = 2
	dav2 = (pm + 1)%2
	return cov_G_real_noPureN(ell,fl, dav1, dav2, config_output)

def cov_NG_cl_shear_real(ell1,ell2,fl1l2, pm, config_output):
	'''
	fl1l2: 2dim cov_Fourier
	pm = 1:+, 0:-
	'''
	dav1 = 3
	dav2 = (pm + 1)%2
	return cov_NG_real(ell1,ell2,fl1l2, dav1, dav2, config_output)


##### gl-gl
def cov_G_gl_gl_real_noPureN(ell,fl, config_output):
	'''
	fl: 1dim cov_Fourier
	'''
	dav1 = dav2 = 2
	return cov_G_real_noPureN(ell,fl, dav1, dav2, config_output)

def cov_NG_gl_gl_real(ell1,ell2,fl1l2, config_output):
	'''
	fl1l2: 2dim cov_Fourier
	'''
	dav1 = dav2 = 2
	return cov_NG_real(ell1,ell2,fl1l2, dav1, dav2, config_output)


##### cl-gl
def cov_G_cl_gl_real_noPureN(ell,fl, config_output):
	'''
	fl: 1dim cov_Fourier
	'''
	dav1 = 3
	dav2 = 2
	return cov_G_real_noPureN(ell,fl, dav1, dav2, config_output)

def cov_NG_cl_gl_real(ell1,ell2,fl1l2, config_output):
	'''
	fl1l2: 2dim cov_Fourier
	'''
	dav1 = 3
	dav2 = 2
	return cov_NG_real(ell1,ell2,fl1l2, dav1, dav2, config_output)


##### cl-cl
def cov_G_cl_cl_real_noPureN(ell,fl, config_output):
	'''
	fl: 1dim cov_Fourier
	'''
	dav1 = dav2 = 3
	return cov_G_real_noPureN(ell,fl, dav1, dav2, config_output)

def cov_NG_cl_cl_real(ell1,ell2,fl1l2, config_output):
	'''
	fl1l2: 2dim cov_Fourier
	'''
	dav1 = dav2 = 3
	return cov_NG_real(ell1,ell2,fl1l2, dav1, dav2, config_output)


