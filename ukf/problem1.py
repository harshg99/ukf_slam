import numpy as np
import matplotlib.pyplot as plt

GEN = False
# Model Parameters
a = -1

mu_x0 = 1
sig_x0 = 2

sig_eps = 1
sig_nu = 0.5


def model(x_k):
	x_k1 = a*x_k + np.random.normal(0,sig_eps,size = (1))
	y_k = np.sqrt(x_k*x_k+1) + np.random.normal(0,sig_nu,size=(1))
	return x_k1,y_k

def gen_data():
	x = np.zeros((101,))
	y = np.zeros((100,))
	x0 = np.random.normal(mu_x0,sig_x0,size=(1))
	x[0] = x0
	for j in range(x.shape[0]-1):
		x[j+1],y[j] = model(x[j])

	np.save('data_prob1.npy', y)
	np.save('x.npy',x)


def propagate(state,cov):
	state_k1 = np.copy(state)
	state_k1[0] = state[0]*state[1]
	A = np.array([[state[1],state[0]],[0,1]])
	cov_  = A@cov@A.T
	cov_[0][0] = cov_[0][0]+sig_eps
	return state_k1,cov_

def merge_meas(state,cov,meas):
	C = np.array([[state[0]/np.power(state[0]*state[0]+1,0.5),0]])
	K = cov@C.T/(C@cov@C.T+sig_nu)
	cov_ = (np.eye(2) - K@C)@cov
	print(K)
	#print(cov_)
	# print(meas-C@state)
	# print(K@(meas - C@state))
	state_ = state + K@(np.array([meas - np.sqrt(state[0]*state[0]+1)]))
	return state_,cov_

def estimate_a(data):

	states = np.zeros((101,2))
	covs = np.zeros((101,2,2))
	state = np.array([2.0,-0.5])
	cov = np.array([[4,0],[0,4]])
	state_k = state
	cov_k  = cov
	states[0,:] = state_k
	covs[0,:,:] = cov
	for j in range(data.shape[0]):
		state_k1_k,cov_k1_k = propagate(state_k,cov_k)
		state_k1_k1,cov_k1_k1 = merge_meas(state_k1_k,cov_k1_k,data[j])
		state_k = state_k1_k1
		cov_k = cov_k1_k1 
		states[j+1,:] = state_k
		covs[j+1,:,:] = cov_k
	
	print(states)
	print(covs)
	return states,covs



if GEN:
	gen_data()

y = np.load('data_prob1.npy')
x = np.load('x.npy')
print(y)
print(x)
s,sig_s = estimate_a(y)
a_ = s[:,1]
cov_a = sig_s[:,1,1]
# Plotting here
t = range(a_.shape[0])
plt.plot(t,a_-cov_a)
plt.plot(t,a_+cov_a)
plt.plot(t,a_)
plt.xlabel('T')
plt.ylabel('Estimate of a')
plt.legend(['a - cov(a)','a','a+cov(a)'])
plt.show()
	
