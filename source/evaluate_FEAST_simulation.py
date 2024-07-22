import numpy as np 
import scipy.stats as ss
import pandas as pd 
import FEAST as F
from scipy.spatial import distance # to calculate the jensen-shannon divergence between two categorical distributions
from scipy.stats import pearsonr
F.set_seed(9999) # set random seed
K = 1 # number of reference source
J = 3 # number of taxa
C_i = 100 # number of reads in a reference samples, they should be different but we will try to keep the process simple for now
C = 100 # number of reads in sink

def generate_true_alpha_gamma(K_obs: int, J:int, K_unknown: int):
	'''
	K_obs: # reference sources OBSERVED
	J: # taxa
	C_i: # reads in a source
	C: # reads in sink
	K_unknown: number of unknown sources that we will simulate. this will result in the last K_unknown rows of source_df (which this function return) being all 0s
	'''
	K = K_obs + K_obs
	return 

def generate_random_discrete_NormalLookalike(N:int, scale:int=3, lower = -10, upper = 11)-> np.array:
	'''
	Generate discrete numbers that seem to follow a normal distribution
	To get a discrete distribution that looks like the normal distribution is to draw from a multinomial distribution where the probabilities are calculated from a normal distribution.
	Source: https://stackoverflow.com/questions/37411633/how-to-generate-a-random-normal-distribution-of-integers
	'''
	x = np.arange(lower, upper) # lower and upper should be centered at 0
	xU, xL = x + 0.5, x - 0.5 
	prob = ss.norm.cdf(xU, scale = scale) - ss.norm.cdf(xL, scale = scale) # calculate the cdf P(X <=x), given that X follows normal (0, scale). This function is calculating the probabilities of numbers that fall into each range such that probabilities itself will follow a normal distribution
	prob = prob / prob.sum() # normalize the probabilities so their sum is 1
	nums = np.random.choice(x, size = N, p = prob)
	return x, prob, nums

def simulate_data(alpha, gamma, K_obs: int=1, J:int=3, C_i:int=100, C:int=100, K_unknown: int=1):
	'''
	K_obs: # reference sources OBSERVED
	J: # taxa
	C_i: # reads in a source
	C: # reads in sink
	K_unknown: number of unknown sources that we will simulate. this will result in the last K_unknown rows of source_df (which this function return) being all 0s
	'''
	# alpha = np.array([0.7, 0.3])
	Y_theory = C_i * gamma
	_, _, noise_Y = generate_random_discrete_NormalLookalike(N = gamma.shape[0]* gamma.shape[1], scale = 1, lower = -5, upper = 6)
	noise_Y = noise_Y.reshape((gamma.shape[0], gamma.shape[1]))
	Y_obs = np.round(np.clip(Y_theory + noise_Y, a_min = 0, a_max = None))
	beta = np.dot(gamma.T, alpha) # beta_j = sum_{i=1}^{K+1} alpha_i * gamma_{ij}
	X_theory = C * beta 
	_, _, noise_X = generate_random_discrete_NormalLookalike(N = len(X_theory), scale = 1, lower = -5, upper = 6)
	X_obs = np.round(np.clip(X_theory + noise_X, a_min = 0, a_max = None))
	# now we have calculated the Y_obs based on full knowledge of the unknown source, but we need to mask the unknown reference counts as 0 to feed into FEAST algorithm
	Y_obs[-K_unknown:,:] = 0
	return X_obs, Y_obs

def init_true_alpha(num_source: int, Uprop: float)-> np.array:
	'''
	For the sake of visibility for myself, I want all the alphas to have single decimal point. 
	'''
	other_source_alpha = np.round(1 - Uprop, 1)
	other_source = num_source-1
	alpha = np.random.multinomial(int(other_source_alpha*10), [1.0/ (other_source)]*other_source, size=1)
	alpha = alpha/10
	alpha = np.append(alpha, Uprop) # add unknown proportion to the list of alphas
	return alpha 

def test_prob_unknown_effects(num_source: int=2)-> None:
	gamma = np.array([[0.3,0.3,0.4], [0, 0.5, 0.5], [0.9,0.05,0.05], [0, 0.05, 0.95]]) 
	gamma = gamma[:num_source,:] # depending on how many sources the user wants to simulate, we will take gamma up to that number.
	for Uprop in np.arange(0.6, 0.7, 0.1): # Uprop: unknown alpha
		alpha = init_true_alpha(num_source, Uprop)
		X_obs, Y_obs = simulate_data(alpha, gamma)
		print('true alpha:', alpha)
		print('X_obs: ', X_obs)
		print('Y_obs: ', Y_obs)
		for i in range(10):
			Falpha, Fgamma, Fllh = F.FEAST(X_obs, Y_obs)
			jenShan = distance.jensenshannon(alpha, Falpha) # bounded 0-1, 0 means perfect similarity (Therefore, lower better)
			corr, p_val = pearsonr(Falpha, alpha)
			# print('true alpha: ', alpha)
			print('FEAST alpha: ', Falpha)
			# print('true gamma:', gamma)
			# print('FEAST gamma:', Fgamma)
			print('pearsonr, jensenshannon, Fllh:', corr, jenShan, Fllh)
			print()
	return 

if __name__ == '__main__':
	# test_prob_unknown_effects(2)
	test_prob_unknown_effects(4)
'''
- If we create the reference such that we mask out the unknown source, then when the proportion of unknown is low, the model perform very poorly (very wrong guess about alpha). But in the same setting, the proportion of the unknown source is very high, then alpha guess by FEAST is very good. 
- The reason why in cases where the unknown proportion is low compared to the known proportion (in ground truth) ANd the model seems to not be able to predict that is because the  model will tend to output that the gamma in unknown is similar to the known source's gamma. --> Is there a way that we can force the model to learn the this signal?
- If the initializations of alpha get changed such that the proportion of unknown source is small, then the model will be better at converging to the correct solution. This is important details because the unknown source gamma (gamma_{K+1}) can, in theory, be set to have perfect correlation with X_obs, and the model will still be able result in high log-likelihood. 
- When K+1 = 3, and the real alpha implies a very even spread between the sources, then the model perform really poorly
- A key feature that makes FEAST perform poorly is when there are strongly correlated gamma across different sources
- initialization of gamma is also important, especially in dealing with real data. I decided that I would initialize gamma just as the observed source_df
- multiple runs of EM is important. The algorithm itself is very fickle and different runs with different random seed can lead to wildly different results. Initialization is very important as well
- Based on simulation, there are scenarios when the correlation with the ground truth of the predicted alpha seems to be bipolar (either close to 1 or close to -1)
'''

