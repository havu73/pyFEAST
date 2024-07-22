import pandas as pd
import numpy as np
import os 
import argparse
import helper
import random
from scipy.spatial import distance # to calculate the jensen-shannon divergence between two categorical distributions

DEFAULT_EPSILON = 1e-10 # small positive constant to add to numbers to avoid the case where we divide by 0, or take log of 0
DEFAULT_NUM_ITER = 1000 # default number of iterations 
DEFAULT_CONVERGE_THRES = 1e-6 # convergen threshold for the log likelihood (difference in log likelihood between two iterations)
DEFAULT_SEED = 9999
def set_seed(seed: int = DEFAULT_SEED) -> None:
	np.random.seed(seed)
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	print(f"Random seed set as {seed}")

def log_epsilon(vector, epsilon):
	try:
		log_vector = np.log(vector)
		assert np.any(np.isinf(log_vector)) == False, 'Taking the log of 0, we will need to add some epsilon to the vector to avoid this case'
		return log_vector
	except:
		return np.log(vector+epsilon)

def normalize_to_sumOne(vector: np.array):
	'''
	normalize an array such that the sum of all entries will be 1
	'''
	vSum = np.sum(vector)
	return vector/vSum

def calculate_jensen_shannon_sink_source(sink_vector: np.array, source_df: np.array):
	'''
	Calculate the jenssen shannon divergence between the sink_vector's relative abundance of species and each of the source in source_df
	sink_vector: (num_taxa)
	source_df: (num_source, num_taxa)
	'''
	sink_vector = normalize_to_sumOne(sink_vector) # (num_taxa)
	source_df = np.apply_along_axis(normalize_to_sumOne, axis=1, arr=source_df) #(num_source, num_taxa)
	dist_w_sink = []
	for source in source_df:
		measure = distance.jensenshannon(sink_vector, source)
		dist_w_sink.append(measure)
	return np.array(dist_w_sink)


# now let's implement the FEAST algorithm itself
def initialize_params(num_source: int, num_taxa: int, source_LDA_ratio: int, sink_vector: np.array, source_df: np.array):
	'''
	Initialize alpha and gamma. Gamme initialization is quite standard: for each source, we can basically intialize gamma_i to be equally spread between taxa. The algorithm can detect the correct gamma_i quite easily based on the observed data. 
	Alpha initialization needs more precautions. The easiest way for the algorithm to fit the data is to calculate alpha_{K+1} to close to 1, and gamma_{K+1} to be highly correlated with the sink_vector. If alpha is initialied to be evenly spread among sources, the EM alogrithm will likely result in really high alpha_{K+1} because of the problem stated above.
	Therefore, initializeing alpha, we want to set alpha_{K+1}^{(0)} to be extremely small compared to other sources.
	source_LDA_ratio is a parameter such that the dirichlet parameters for sources other than the unknown source is this number, while the unknown source has value 1 as its dirichlet parameters
	'''
	# first, we will calculate the jensen shannon distance between the relative abundance of sink_vector with each of the known sources in source_df

	LDA_params = np.full(num_source-1, source_LDA_ratio) # fill a numpy matrix of the specified shape with the value in the second argument. In this case, we want all the elements of alpha associated with the known sources to have values source_LDA_ratio
	LDA_params = np.append(LDA_params, 1) # if the other sources has dirichlet value 
	alpha = np.random.dirichlet(LDA_params, size=1)[0] 
	# alpha: vector of size num_source that sum to 1, and represent the proportions of the sources that contribut to the sink
	# now, onto initialize gamma. We want to initialize gamma based on the proportion of each taxa observed each source in source_df. For sources where there's no reads for any taxa, we will change the count of all taxa in that source from 0 to 1 so that the initialized gamma values will be exactly equal for all taxa
	sum_read_per_source = np.sum(source_df, axis = 1) # --> (num_source): total sum of reads per source
	source_df[sum_read_per_source==0,:] = 1 # in rows (sources) where there are no reads (usually because we do not see anything in this source, or because this source is an unknown source artificially set up by us), then we will set the reads for all taxa in this source to be 1 instead of 0. This is useful because we will eventially extract the relative abundance of each taxa in each source, this will avoid the situation of division by 0
	sum_read_per_source[sum_read_per_source==0] = source_df.shape[1]
	gamma =  np.divide(source_df, sum_read_per_source[:, np.newaxis])# shape (num_source, num_taxa) 
	# we intialize gamma to be the observed relative abundance of taxa in each source
	return alpha, gamma

def calculate_expected_prop_of_source_in_each_sink_taxa(alpha, gamma, sink_vector, epsilon):
	'''
	alpha: (num_source) --> the CURRENT_ESTIMATE proportion of each source in the sink
	gamma: (num_source, num_taxa) --> the relative abundance of each taxa in each source. For each source, the relative abundance of taxa sum to 1 --> row_sum = 1
	sink_vector: a vector of length #species (taxa), denoting the #reads of each taxa observed in the sink
	This function will estimate the #reads of each source for each observed taxa in the sink. In particular, if x_j denotes #reads from taxa j in sink, we want to estimate x_j * E(Z_ij) --> estimated #reads of taxa j in the sink is from source i
	'''
	nom = gamma * alpha[:, np.newaxis] # alpha_i * gamma_ij = exp_Zij
	denom = np.dot(alpha.T, gamma) # sum over all i (1 to num_source), alpha_i * gamma_ij
	denom[denom==0] = epsilon
	exp_Zij = np.divide(nom, denom) # expected proportion of each taxa that are from each source
	assert np.any(np.isnan(exp_Zij) | np.isinf(exp_Zij)) == False, 'In calculating beta_j (see the derivation), the results involve either taxa that has beta_j as nan or inf. Exiting...'
	X_exp_Zij = exp_Zij * sink_vector # expected read counts of each taxa that are from each source
	return X_exp_Zij 

def calculate_MLE_alpha(X_exp_Zij):
	'''
	X_exp_Zij: result from the E-step, (num_source, num_taxa): expected number of read counts of each sink taxa that is from each source 
	'''
	alpha_nom = X_exp_Zij.sum(axis = 1) # row sum --> (num_source)
	alpha_denom = np.sum(alpha_nom)
	alpha = alpha_nom / alpha_denom
	return alpha

def calculate_MLE_gamma(X_exp_Zij, source_df):
	'''
	X_exp_Zij: result from the E-step, (num_source, num_taxa): expected number of read counts of each sink taxa that is from each source 
	source_df: a df of shape (num_source, num_taxa) : read counts of each taxa from each source. Note the reason why source_df has num_source-1 rows is bc we do not have data of the unknown source in source_df
	'''
	gamma_nom = X_exp_Zij + source_df # element-wise sum --> (num_source, num_taxa)
	gamma_denom = gamma_nom.sum(axis = 1) # row sum --> (num_source)
	gamma = gamma_nom / gamma_denom[:, np.newaxis]
	return gamma

def calculate_LLH_without_constants(alpha, gamma, sink_vector, source_df, epsilon):
	'''
	alpha: (num_source) --> the CURRENT_ESTIMATE proportion of each source in the sink --> model params
	gamma: (num_source, num_taxa) --> the relative abundance of each taxa in each source. For each source, the relative abundance of taxa sum to 1 --> row_sum = 1 --> model_params
	sink_vector: a vector of length #species (taxa), denoting the #reads of each taxa observed in the sink --> observed data
	source_df: a df of shape (num_source, num_taxa) : read counts of each taxa from each source --> observed data
	epsilon: a small positive number that we can add to numbers to avoid avoid division by 0 or taking log of 0
	'''
	# calculate beta --> a vector of length num_taxa --> relative abundance of each taxa in the sink
	beta = np.dot(alpha.T, gamma) # beta_j = sum_{i=1}^{K+1} alpha_i * gamma_ij
	log_beta = log_epsilon(beta, epsilon) # element-wise log  --> (num_taxa)
	X_log_beta = np.dot(sink_vector, log_beta) # sum_{j=1}^{num_taxa} X_j * log(beta_j)
	log_gamma = log_epsilon(gamma, epsilon) # element-wise log --> (num_source, num_taxa)
	flat_logGamma = log_gamma.flatten() # (num_source * num_taxa)
	flat_source = source_df.flatten()	# (num_source * num_taxa)
	Y_log_gamma = np.dot(flat_source, flat_logGamma) # sum_{i=1}^{num_source} sum_{j=1}^{num_taxa} Y_ij * log(gamma_ij), Y is source_df
	llh_no_constants = X_log_beta + Y_log_gamma # we calculate llh without the constants of observed data based on the first term of multinomal distribution (C choose X1, ..., XN) and (C_i choose Y_i1,...,Y_iN)
	return llh_no_constants


def FEAST(sink_vector: np.array, source_df: np.array, source_LDA_ratio:int = 10, num_iter: int=DEFAULT_NUM_ITER, converge_thres: float = DEFAULT_CONVERGE_THRES, epsilon: float=DEFAULT_EPSILON):
	'''
	sink_vector: a vector of length #species
	source_df: a df of shape (num_source-1, num_taxa) 
	source_LDA_ratio: a parameter that sets how many times the other sources should be set to a larger value compared to the unknown source's alpha, in the initialization step. This is important because we want to direct the unknown source to be really SMALL. Reason: EM can easily fit the data well if it sets unknown source's alpha to be close to 1 and unknown source's gamma to be well-correlated with the sink_vector. We don't want EM to do that, we want to force it to learn the true alpha before resorting to setting unknown's alpha to be almost 1 
	'''
	# source_unknown = np.zeros(source_df.shape[1]).astype('int')[np.newaxis,:] # shape(1, num_taxa) --> the additional row of taxa count for the unknown (K+1) source
	# source_df = np.concatenate((source_df, source_unknown)) # add a row of 0, which correspond to the read count in the unknown (K+1) source. Adding this row of 0 will create updates of gamma_ij that is consitent across all cases (whether 1<=i<=K or i=K+1). Please see the derivation in the provided pdf for details.
	num_source = source_df.shape[0] 
	num_taxa = source_df.shape[1]
	assert len(sink_vector) == num_taxa, 'Number of taxa from the source data and sink data is not consistent. Check to make sure that your source data and sink data both contain information for the same number of taxa'
	# first, inintialize the paramters
	alpha, gamma = initialize_params(num_source, num_taxa, source_LDA_ratio, source_df)
	print('init alpha:', alpha)
	# create loop until convergence
	current_llh = -np.inf # we want to MAXIMIZE llh
	for step in range(num_iter):
		# E-step
		X_exp_Zij = calculate_expected_prop_of_source_in_each_sink_taxa(alpha, gamma, sink_vector, epsilon) # tested this function, it does exactly what is expected
		# M-step
		alpha = calculate_MLE_alpha(X_exp_Zij) # tested this function, it does exactly is expected of it
		gamma = calculate_MLE_gamma(X_exp_Zij, source_df)
		# calculate the log likelihood and check if it converges. If it does then stop the algorithm
		llh = calculate_LLH_without_constants(alpha, gamma, sink_vector, source_df, epsilon)
		# if llh - current_llh <= converge_thres:
		# 	return alpha, gamma, llh
		current_llh = llh
	return alpha, gamma, llh

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'The core algorithm of FEAST to predict the proportion of the sink microbiome that is from each of the source microbiome')
	# Create argument groups
	input_group = parser.add_argument_group("Input")
	output_group = parser.add_argument_group("Output")
	hyperParam_group = parser.add_argument_group('HyperParam')
	input_group.add_argument('--sink_fn', type=str, help = "There should be two columns correpsonding to taxa and read_counts, as observed in the sink sample", required = False)
	input_group.add_argument('--source_fn', type=str, help = 'where the data of read counts in different sources should be stored, the column should correspond to taxa, followed by different sources, and the values in each cells correspond to read counts of each taxa (row) in each source (column)', required = False)
	output_group.add_argument('--output_folder', type=str,
		help = 'Where we will save the results of the deconvolution', required = False)
	hyperParam_group.add_argument('--epsilon', type=float, required = False, default = DEFAULT_EPSILON, help = 'epsilon value to add to numbers when we encounter situations of division by zero or taking log of zero')
	hyperParam_group.add_argument('--converge_thres', type = float, required = False, default = DEFAULT_CONVERGE_THRES, help = 'converge threshold for the EM algorithm')
	hyperParam_group.add_argument('--num_iter', type=int, required = False, default = DEFAULT_NUM_ITER, help = 'number of iterations of the EM algorithm')
	hyperParam_group.add_argument('--seed', type = int, required = False, default = 9999, help = 'random number generator seed, for reproducibility')
	args = parser.parse_args()
	helper.make_dir(args.output_folder)
	helper.check_file_exist(args.sink_fn)
	helper.check_file_exist(args.source_fn)
	# set_seed(args.seed)
	meta_fn = '/u/home/h/havu73/project-ernst/FEAST_study/data/cozygene/metadata_example.txt'
	count_fn = '/u/home/h/havu73/project-ernst/FEAST_study/data/cozygene/otu_example.txt'
	meta_df = pd.read_csv(meta_fn, header = 0, index_col = None, sep = '\t')
	count_df = pd.read_csv(count_fn, header = 0, index_col = None, sep = '\t')
	sources = meta_df[meta_df['SourceSink'] == 'Source']['SampleID']
	source_df = count_df[sources].values.T # (num_source, num_taxa)
	sink_vector = count_df['ERR525698'].values # (num_taxa,)
	all_zero_columns = np.all(source_df == 0, axis=0) # each entry is true of false depending on whether the taxa is all-0-count across sources
	all_zero_columns = all_zero_columns & (sink_vector == 0)
	Fsource_df = source_df[:, ~all_zero_columns]
	Fsink_vector = sink_vector[~all_zero_columns]
	Fsource_df = np.concatenate((Fsource_df, Fsink_vector[np.newaxis,:]))
	alpha, gamma, llh = FEAST(Fsink_vector, Fsource_df, DEFAULT_NUM_ITER, DEFAULT_CONVERGE_THRES, DEFAULT_EPSILON)
	print(alpha)




