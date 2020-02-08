import numpy as np
from scipy.special import gamma 

## functions related to gamma functions
def g_m_vals(mu,q):
	'''
	g_m_vals function is adapted from FAST-PT
	'''
	imag_q= np.imag(q)
	
	g_m=np.zeros(q.size, dtype=complex)

	cut =200
	asym_q=q[np.absolute(imag_q) >cut]
	asym_plus=(mu+1+asym_q)/2.
	asym_minus=(mu+1-asym_q)/2.
	
	q_good=q[ (np.absolute(imag_q) <=cut) & (q!=mu + 1 + 0.0j)]

	alpha_plus=(mu+1+q_good)/2.
	alpha_minus=(mu+1-q_good)/2.
	
	g_m[(np.absolute(imag_q) <=cut) & (q!= mu + 1 + 0.0j)] =gamma(alpha_plus)/gamma(alpha_minus)

	# asymptotic form 								
	g_m[np.absolute(imag_q)>cut] = np.exp( (asym_plus-0.5)*np.log(asym_plus) - (asym_minus-0.5)*np.log(asym_minus) - asym_q \
	    +1./12 *(1./asym_plus - 1./asym_minus) +1./360.*(1./asym_minus**3 - 1./asym_plus**3) +1./1260*(1./asym_plus**5 - 1./asym_minus**5) )

	g_m[np.where(q==mu+1+0.0j)[0]] = 0.+0.0j
	
	return g_m

def g_l(l,z_array):
	'''
	gl = 2.**z_array * gamma((l+z_array)/2.) / gamma((3.+l-z_array)/2.)
	'''
	gl = 2.**z_array * g_m_vals(l+0.5,z_array-1.5)
	return gl

## Window function
def c_window(n,n_cut):

	n_right = n[-1] - n_cut
	n_left = n[0]+ n_cut 

	n_r=n[ n[:]  > n_right ] 
	n_l=n[ n[:]  <  n_left ] 
	
	theta_right=(n[-1]-n_r)/float(n[-1]-n_right-1) 
	theta_left=(n_l - n[0])/float(n_left-n[0]-1) 

	W=np.ones(n.size)
	W[n[:] > n_right]= theta_right - 1/(2*np.pi)*np.sin(2*np.pi*theta_right)
	W[n[:] < n_left]= theta_left - 1/(2*np.pi)*np.sin(2*np.pi*theta_left)
	
	return W


## Extrapolation
def linear_extra_P(fk, N_low, N_high):
	'''
	1d linear extrapolation of the input fk array

	fk: input array
	N_low: number of points to extrapolate on the lower sides
	N_high: number of points to extrapolate on the higher sides
	'''
	logfk = np.log(fk) # This Extrapolation only works in log space	
	grad_left = logfk[1] - logfk[0]
	grad_right = logfk[-1] - logfk[-2]

	add_left = np.arange(-N_low,0)
	add_right = np.arange(1,N_high)

	left_array = add_left*grad_left + logfk[0]
	right_array= add_right*grad_right+logfk[-1]

	new_fk = np.exp(np.hstack((left_array,logfk, right_array)))
	return new_fk

def bilinear_extra_P(fk1k2, N_low, N_high):
	'''
	2d bilinear extrapolation of the input fk1k2 matrix

	fk1k2: input matrix
	N_low: number of points to extrapolate on the lower sides
	N_high: number of points to extrapolate on the higher sides
	'''
	logfk1k2 = np.log(fk1k2) # This Extrapolation only works in log space
	h_grad_left = logfk1k2[:,1]-logfk1k2[:,0] # horizontal gradient left side
	h_grad_right= logfk1k2[:,-1]-logfk1k2[:,-2] # horizontal gradient right side
	add_left = np.arange(-N_low,0)
	left_matrix = np.matrix(h_grad_left).T.dot(np.matrix(add_left)) + np.matrix(logfk1k2[:,0]).T
	add_right = np.arange(1,N_high)
	right_matrix= np.matrix(h_grad_right).T.dot(np.matrix(add_right)) + np.matrix(logfk1k2[:,-1]).T
	new_logfk1k2 = np.hstack((left_matrix,logfk1k2,right_matrix)) ## type: matrix

	v_grad_up = new_logfk1k2[1,:] - new_logfk1k2[0,:] ## type: matrix
	v_grad_down=new_logfk1k2[-1,:]- new_logfk1k2[-2,:] ## type: matrix
	up_matrix = np.matrix(add_left).T.dot(v_grad_up) + np.matrix(new_logfk1k2[0,:])
	down_matrix=np.matrix(add_right).T.dot(v_grad_down)+np.matrix(new_logfk1k2[-1,:])
	result_matrix= np.vstack((up_matrix,new_logfk1k2,down_matrix)) ## type: matrix
	return np.exp(np.array(result_matrix))## type: array