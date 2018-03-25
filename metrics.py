import numpy as np
from frechet import *
from hausdorff import hausdorff
from edit_distance import *
from euclidean_distance import *

def compare_multiple_scanpaths(human_scanpaths, predicted_scanpaths):

	output = {'ED':[],'FD':[], 'HD':[], 'EUD':[]}
	
	for h_scan, p_scan in zip(human_scanpaths, predicted_scanpaths):
		
		human_str =  scanpath_to_string(h_scan, 768, 1024, 5)
		model_str =  scanpath_to_string(p_scan, 768, 1024,5)

		ED = levenshtein(human_str, model_str, substitution_cost=1)
		FD = frechetDist(h_scan, p_scan)
		HD = 1049*hausdorff(h_scan/1049, p_scan/1049)
		EUD = euclidean_distance(h_scan, p_scan)
		output['ED'].append(ED)
		output['FD'].append(FD)
		output['HD'].append(HD)
		output['EUD'].append(EUD)
		
	return output
	
if __name__=='__main__':
	
	human_scanpath = np.load('png/orig_test_4.npy')[:,:2]
	predicted_scanpath= np.load('png/test4.jpeg.npy')[:,:2]


	stat = compare_multiple_scanpaths(
									np.array([human_scanpath, human_scanpath]),
									np.array([predicted_scanpath,predicted_scanpath]))
	print('FD: ', stat['FD'], 'FD Avg: ', sum(stat['FD'])/len(stat['FD']))	
	print('HD: ', stat['HD'], 'HD Avg: ', sum(stat['HD'])/len(stat['HD']))	
	print('ED: ', stat['ED'], 'FD Avg: ', sum(stat['ED'])/len(stat['ED']))	
	print('EUD:', stat['EUD'], 'FD Avg: ', sum(stat['EUD'])/len(stat['EUD']))	

