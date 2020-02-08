import yaml
import sys
sys.path.append('../')
from covariances_real import *

# configfile = "config_output.yaml"
configfile = "config_output_desy1.yaml"

# Read yaml file
with open(configfile, 'r') as stream:
	config_output = yaml.load(stream)


# file_real = 'inputs/y3_mcal_ssss_++_cov_Ntheta30_Ntomo4_1'
# file_real = 'inputs/y3_mcal_ssss_--_cov_Ntheta30_Ntomo4_56'
# file_real = 'inputs/y3_mcal_ssss_+-_cov_Ntheta30_Ntomo4_113'
# file_real = 'inputs/y1_mcal_ssc_ssss_+-_cov_Ntheta20_Ntomo4_111'
file_real = 'inputs/y1_mcal_ssc_llls_cov_Ntheta20_Ntomo4_936'
data_real = np.loadtxt(file_real)
# r_vals_real = data_real[:900,2]
# r_vals_real = r_vals_real[::30]
# cov_real = data_real[:900,8]
# cov_real = cov_real[::31]
r_vals_real = data_real[:400,2]
r_vals_real = r_vals_real[::20]
cov_real = data_real[:400,8]
cov_real = cov_real[::21]

##########################################

ell_real = 1./r_vals_real[::-1]
N_r = ell_real.size
N_fill = 17
ell_real_fill = np.logspace(np.log10(ell_real[0]), np.log10(ell_real[-1]), num=(1+N_fill)*(N_r-1)+1, endpoint=True )
N_extra_left = 222
N_extra_right = 206
dlnl = np.log(ell_real_fill[1]/ell_real_fill[0])
ell_left = np.exp(dlnl * np.arange(-N_extra_left, 0) ) * ell_real_fill[0]
ell_right = np.exp(dlnl * np.arange(1, N_extra_right+1) ) * ell_real_fill[-1]
ell_full = np.hstack((ell_left, ell_real_fill, ell_right))
print(ell_full[0],ell_full[-1])
# exit()

# file = 'inputs/CL.txt'
# data = np.loadtxt(file)

# l_num = data[:-1,0]
# cl = data[:-1,1]
# noise = data[:-1,2]
# power = cl*(cl+2*noise)
# power = cl*(cl)

# func = l_num**3 * power *2.

file = 'inputs/check.txt'
data = np.loadtxt(file)

l_num = data[:-1,0]

func = np.exp(data[:-1,1])


# ell_sample, fl_sample = resample_ell_fl(l_num, func, config_output)

from scipy.interpolate import interp1d
fl_interp = interp1d(np.log(l_num), np.log(func),kind='linear')
fl_full = np.exp(fl_interp(np.log(ell_full)))



pm1, pm2 = 1,0

from time import time
t1 = time()
# out_theta, cov_out = cov_G_shear_shear_real_noPureN(ell_sample,fl_sample, pm1, pm2, config_output)
# out_theta, cov_out = cov_G_cl_gl_real_noPureN(ell_sample,fl_sample, config_output)

out_theta, cov_out = cov_G_cl_gl_real_noPureN(ell_full,fl_full, config_output)
t2 = time()
print('time spent in calculating covariance: '+str(t2-t1)+'s')
cov_diag = np.diag(cov_out)




file_real2 = 'inputs/y1_mcal_ssc_llls_cov_Ntheta20_Ntomo4_936_2'
data_real2 = np.loadtxt(file_real2)
r_vals_real2 = data_real2[:400,2]
r_vals_real2 = r_vals_real2[::20]
cov_real2 = data_real2[:400,8]
cov_real2 = cov_real2[::21]

import matplotlib.pyplot as plt
plt.imshow(cov_out)
plt.show()
# exit()

plt.xscale('log')
plt.yscale('log')
plt.plot(out_theta/np.pi*180.*60., abs(cov_diag), label='FASTCov')
plt.plot(r_vals_real/np.pi*180*60, abs(cov_real), label='CosmoLike')
plt.plot(r_vals_real2/np.pi*180*60, abs(cov_real2), label='Cfastcov')
plt.legend()
plt.xlabel(r'$\theta$')
plt.ylabel('Cov diagonal')
plt.show()

# np.savetxt('res_diag_bias7.txt', np.c_[out_theta, cov_diag])
# np.savetxt('res_cov_bias7.txt', cov_out)