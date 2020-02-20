import numpy as np
from preprocess import *
from twobessel import twobessel
from scipy.interpolate import interp2d
from scipy.linalg import block_diag


## General forms
def bin_cov_G_real_noPureN(ell,fl, dav1, dav2, config_output):
	'''
	fl: 1dim cov_Fourier
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
			  'nu1': 1.01, 'nu2': 1.01, 'c_window_width': 0.25}

	# convert from arcmin to radians
	out_thetamin, out_thetamax = config_output['out_thetamin']/60./180.*np.pi,config_output['out_thetamax']/60./180.*np.pi
	
	out_Ntheta = config_output['out_Ntheta']
	out_theta = np.logspace(np.log10(out_thetamin),np.log10(out_thetamax), num=out_Ntheta, endpoint=True)

	config['c_window_width'] = config_output['c_window_width']
	f_sky = config_output['f_sky']/41253. # convert from deg squared to steradian

	ell1_extrap, ell2_extrap, fl1l2_extrap, config_pre = extrap_fourier_gauss(ell,fl,config)

	# if(z1!=z2 or pm1!=pm2): config['sym_Flag']=False
	davs = {0: 0, 1: 4, 2: 2, 3: 0}
	config['l1'] = davs[dav1] - 0.5
	config['l2'] = davs[dav2] - 0.5

	theta1_out, theta2_out, result_out = twobessel(ell1_extrap, ell2_extrap, fl1l2_extrap, config_pre, config)

	log_out_theta = np.log(out_theta)
	dlog_out_theta=log_out_theta[1]-log_out_theta[0]

	cov_interpfunc = interp2d(np.log(theta1_out), np.log(theta2_out), result_out, kind = 'cubic')
	rebin_factor = config_output['rebin_factor']
	N_rebin = rebin_factor*(out_Ntheta-1)
	logtheta_rebin = np.linspace(log_out_theta[0], log_out_theta[-1], num=N_rebin, endpoint=False)
	dlogtheta_rebin = logtheta_rebin[1]-logtheta_rebin[0]
	logtheta_rebin += dlogtheta_rebin/2.
	theta_rebin = np.exp(logtheta_rebin)
	cov_rebinned = cov_interpfunc(logtheta_rebin,logtheta_rebin) ## array with shape: N_rebin*N_rebin

	cov_rebinned /= (2.*np.pi**3*f_sky)
	cov_rebinned_weighted = (cov_rebinned.T * theta_rebin**(2.5)).T * theta_rebin**(2.5)

	cov_squeezed_temp = cov_rebinned_weighted[0::rebin_factor,:]
	for idx in xrange(rebin_factor-1):
		idx+=1
		cov_squeezed_temp += cov_rebinned_weighted[idx::rebin_factor,:]
	cov_squeezed = cov_squeezed_temp[:,0::rebin_factor]
	for idy in xrange(rebin_factor-1):
		idy+=1
		cov_squeezed += cov_squeezed_temp[:,idy::rebin_factor]
	
	cov_squeezed *= (2.*dlogtheta_rebin/(np.exp(2.*dlog_out_theta)-1.))**2
	theta_bin_min = out_theta[:-1]
	theta_bin_min_2 = theta_bin_min**2
	cov_bin = ((cov_squeezed / theta_bin_min_2).T / theta_bin_min_2).T

	theta_bin_ave = np.diff(out_theta**3)/np.diff(out_theta**2)*2./3.
	# theta_bin_center = np.exp( log_out_theta[:-1] + dlog_out_theta/2.)
	return theta_bin_ave, cov_bin


def bin_cov_NG_real(ell1, ell2, fl1l2, dav1, dav2, config_output):
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
			  'nu1': 1.01, 'nu2': 1.01, 'c_window_width': 0.25}

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

	theta1_out, theta2_out, result_out = twobessel(ell1_extrap, ell2_extrap, fl1l2_extrap, config_pre, config)

	log_out_theta = np.log(out_theta)
	dlog_out_theta=log_out_theta[1]-log_out_theta[0]

	cov_interpfunc = interp2d(np.log(theta1_out), np.log(theta2_out), result_out, kind = 'cubic')
	rebin_factor = config_output['rebin_factor']
	N_rebin = rebin_factor*(out_Ntheta-1)
	logtheta_rebin = np.linspace(log_out_theta[0], log_out_theta[-1], num=N_rebin, endpoint=False)
	dlogtheta_rebin = logtheta_rebin[1]-logtheta_rebin[0]
	logtheta_rebin += dlogtheta_rebin/2.
	theta_rebin = np.exp(logtheta_rebin)
	cov_rebinned = cov_interpfunc(logtheta_rebin,logtheta_rebin) ## array with shape: N_rebin*N_rebin

	cov_rebinned /= (2.*np.pi**3*f_sky)
	cov_rebinned_weighted = ((cov_rebinned.T * theta_rebin**(2.5)).T * theta_rebin**(2.5)).T

	filter_matrix = np.ones((N_rebin, out_Ntheta-1))
	cov_squeezed = filter_matrix.T.dot(cov_rebinned_weighted).dot(filter_matrix)
	cov_squeezed *= (dlogtheta_rebin/rebin_factor)**2
	
	cov_squeezed *= (2./(np.exp(2.*dlog_out_theta)-1.))**2
	theta_bin_min = out_theta[:-1]
	theta_bin_min_2 = theta_bin_min**2
	cov_bin = ((cov_squeezed / theta_bin_min_2).T / theta_bin_min_2).T

	return theta_bin_min, cov_bin




########################
## Individual covariance pieces

##### shear-shear
def bin_cov_G_shear_shear_real_noPureN(ell,fl, pm1, pm2, config_output):
	'''
	fl: 1dim cov_Fourier
	(pm1,pm2)=(1,1): ++, (1,0): +-, (0,0): --
	'''
	dav1 = (pm1 + 1)%2
	dav2 = (pm2 + 1)%2
	return bin_cov_G_real_noPureN(ell,fl, dav1, dav2, config_output)

def bin_cov_NG_shear_shear_real(ell1,ell2,fl1l2, pm1, pm2, config_output):
	'''
	fl1l2: 2dim cov_Fourier
	(pm1,pm2)=(1,1): ++, (1,0): +-, (0,0): --
	'''
	dav1 = (pm1 + 1)%2
	dav2 = (pm2 + 1)%2
	return bin_cov_NG_real(ell1,ell2,fl1l2, dav1, dav2, config_output)


##### gl-shear
def bin_cov_G_gl_shear_real_noPureN(ell,fl, pm, config_output):
	'''
	fl: 1dim cov_Fourier
	pm = 1:+, 0:-
	'''
	dav1 = 2
	dav2 = (pm + 1)%2
	return bin_cov_G_real_noPureN(ell,fl, dav1, dav2, config_output)

def bin_cov_NG_gl_shear_real(ell1,ell2,fl1l2, pm, config_output):
	'''
	fl1l2: 2dim cov_Fourier
	pm = 1:+, 0:-
	'''
	dav1 = 2
	dav2 = (pm + 1)%2
	return bin_cov_NG_real(ell1,ell2,fl1l2, dav1, dav2, config_output)



##### cl-shear
def bin_cov_G_cl_shear_real_noPureN(ell,fl, pm, config_output):
	'''
	fl: 1dim cov_Fourier
	pm = 1:+, 0:-
	'''
	dav1 = 2
	dav2 = (pm + 1)%2
	return bin_cov_G_real_noPureN(ell,fl, dav1, dav2, config_output)

def bin_cov_NG_cl_shear_real(ell1,ell2,fl1l2, pm, config_output):
	'''
	fl1l2: 2dim cov_Fourier
	pm = 1:+, 0:-
	'''
	dav1 = 3
	dav2 = (pm + 1)%2
	return bin_cov_NG_real(ell1,ell2,fl1l2, dav1, dav2, config_output)


##### gl-gl
def bin_cov_G_gl_gl_real_noPureN(ell,fl, config_output):
	'''
	fl: 1dim cov_Fourier
	'''
	dav1 = dav2 = 2
	return bin_cov_G_real_noPureN(ell,fl, dav1, dav2, config_output)

def bin_cov_NG_gl_gl_real(ell1,ell2,fl1l2, config_output):
	'''
	fl1l2: 2dim cov_Fourier
	'''
	dav1 = dav2 = 2
	return bin_cov_NG_real(ell1,ell2,fl1l2, dav1, dav2, config_output)


##### cl-gl
def bin_cov_G_cl_gl_real_noPureN(ell,fl, config_output):
	'''
	fl: 1dim cov_Fourier
	'''
	dav1 = 3
	dav2 = 2
	return bin_cov_G_real_noPureN(ell,fl, dav1, dav2, config_output)

def bin_cov_NG_cl_gl_real(ell1,ell2,fl1l2, config_output):
	'''
	fl1l2: 2dim cov_Fourier
	'''
	dav1 = 3
	dav2 = 2
	return bin_cov_NG_real(ell1,ell2,fl1l2, dav1, dav2, config_output)


##### cl-cl
def bin_cov_G_cl_cl_real_noPureN(ell,fl, config_output):
	'''
	fl: 1dim cov_Fourier
	'''
	dav1 = dav2 = 3
	return bin_cov_G_real_noPureN(ell,fl, dav1, dav2, config_output)

def bin_cov_NG_cl_cl_real(ell1,ell2,fl1l2, config_output):
	'''
	fl1l2: 2dim cov_Fourier
	'''
	dav1 = dav2 = 3
	return bin_cov_NG_real(ell1,ell2,fl1l2, dav1, dav2, config_output)


