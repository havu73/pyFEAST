import pandas as pd
import numpy as np
import os 
import FEAST as F
from scipy.spatial import distance
from scipy.stats import pearsonr
# meta_fn = '/u/home/h/havu73/project-ernst/FEAST_study/data/sourcetracker/data/metadata.txt'
# meta_df = pd.read_csv(meta_fn, header = 0, index_col = None, sep = '\t') # shape (305, 6)
# # there are 180 source and 125 sink in meta_df
# count_fn = '/u/home/h/havu73/project-ernst/FEAST_study/data/sourcetracker/data/otus.txt'
# count_df = pd.read_csv(count_fn, header = 0, index_col = None, sep = '\t', skiprows = 1)
# count_df.rename(columns = {'#OTU ID': 'otus_id'}, inplace = True) # shape (6750, 306) --> 305 of them correspond to the different samples, from meta_df
# otus_fn = '/u/home/h/havu73/project-ernst/FEAST_study/data/sourcetracker/data/otu_RDP_lineages.txt' # this file contains all the metadata of the different species of bacteria present in the data
# otus_df = pd.read_csv(otus_fn, header = 0, index_col = None, sep = '\t') # just two columns: otus_id and the species name
# otus_df.rename(columns = {'OTU ID': 'otus_id'}, inplace = True) # shape (6750, 2)
# sources = meta_df['#SampleID'][meta_df['SourceSink'] == 'source']
# source_df = count_df[sources]
# source_df = source_df.values.T
# sink_vector = count_df['Run20100430_H2O-1'].values
# alpha, gamma, llh = F.FEAST(sink_vector, source_df, F.DEFAULT_NUM_ITER, F.DEFAULT_CONVERGE_THRES, F.DEFAULT_EPSILON)

def calculate_jensen_shannon_distance(source_df):
	'''
	code provided by ChatGPT
	'''
	# Normalize the data to create probability distributions
	obs_gamma = source_df / np.sum(source_df, axis=1, keepdims=True)

	# Calculate Jensen-Shannon Divergence matrix
	num_samples = source_df.shape[0]
	jsd_matrix = np.zeros((num_samples, num_samples))
	for i in range(num_samples):
		for j in range(i, num_samples):
			p_avg = (obs_gamma[i] + obs_gamma[j]) / 2
			jsd = 0.5 * (distance.jensenshannon(obs_gamma[i], p_avg) +
			             distance.jensenshannon(obs_gamma[j], p_avg))
			jsd_matrix[i, j] = jsd
			jsd_matrix[j, i] = jsd
	return jsd_matrix

# cozygene example
meta_fn = '/u/home/h/havu73/project-ernst/FEAST_study/data/cozygene/metadata_example.txt'
count_fn = '/u/home/h/havu73/project-ernst/FEAST_study/data/cozygene/otu_example.txt'
meta_df = pd.read_csv(meta_fn, header = 0, index_col = None, sep = '\t')
count_df = pd.read_csv(count_fn, header = 0, index_col = None, sep = '\t')
sources = meta_df[meta_df['SourceSink'] == 'Source']['SampleID']
source_df = count_df[sources].values.T # (num_source, num_taxa)
source_unknown = np.ones(source_df.shape[1]).astype('int')[np.newaxis,:] # shape(1, num_taxa) --> the additional row of taxa count for the unknown (K+1) source
source_df = np.concatenate((source_df, source_unknown)) # add a row of 0, which correspond to the read count in the unknown (K+1) source. Adding this row of 0 will create updates of gamma_ij that is consitent across all cases (whether 1<=i<=K or i=K+1). Please see the derivation in the provided pdf for details.

all_zero_columns = np.all(source_df == 0, axis=0) # each entry is true of false depending on whether the taxa is all-0-count across sources
sink_vector = count_df['ERR525698'].values # (num_taxa,)
all_zero_columns = all_zero_columns & (sink_vector == 0)
Fsource_df = source_df[:, ~all_zero_columns]
Fsink_vector = sink_vector[~all_zero_columns]
correlation_matrix = np.corrcoef(source_df)
corr, p_val = pearsonr(sink_vector, source_df[0,:])
# print(corr)
# print(correlation_matrix)
jensenshannon_matrix = calculate_jensen_shannon_distance(source_df)
# print('jensenshannon_matrix:')
# print(jensenshannon_matrix)
for i in range(3):
	Falpha, Fgamma, Fllh = F.FEAST(sink_vector, source_df)
	print(Falpha)
print(Falpha)