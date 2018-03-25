import numpy as np
from hausdorff import hausdorff

import numpy as np
import math
# Euclidean distance.
def euc_dist(pt1,pt2):
    return math.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))

def _c(ca,i,j,P,Q):
    if ca[i,j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        ca[i,j] = euc_dist(P[0],Q[0])
    elif i > 0 and j == 0:
        ca[i,j] = max(_c(ca,i-1,0,P,Q),euc_dist(P[i],Q[0]))
    elif i == 0 and j > 0:
        ca[i,j] = max(_c(ca,0,j-1,P,Q),euc_dist(P[0],Q[j]))
    elif i > 0 and j > 0:
        ca[i,j] = max(min(_c(ca,i-1,j,P,Q),_c(ca,i-1,j-1,P,Q),_c(ca,i,j-1,P,Q)),euc_dist(P[i],Q[j]))
    else:
        ca[i,j] = float("inf")
    return ca[i,j]

""" Computes the discrete frechet distance between two polygonal lines
Algorithm: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
P and Q are arrays of 2-element arrays (points)
"""
def frechetDist(P,Q):
    ca = np.ones((len(P),len(Q)))
    ca = np.multiply(ca,-1)
    return _c(ca,len(P)-1,len(Q)-1,P,Q)
    
def euclidean_distance(human_scanpath, simulated_scanpath):
	min_len = min(human_scanpath.shape[0], simulated_scanpath.shape[0])
	human_scanpath = human_scanpath[:min_len,:]
	simulated_scanpath = simulated_scanpath[:min_len,:]

	if len(human_scanpath) == len(simulated_scanpath):

		dist = np.zeros(len(human_scanpath))
		for i in range(len(human_scanpath)):
			P = human_scanpath[i]
			Q = simulated_scanpath[i]
			dist[i] = np.sqrt((P[0] - Q[0]) ** 2 + (P[1] - Q[1]) ** 2)
		return np.mean(dist)

	else:

		print( 'Error: The two sequences must have the same length!')
		return False
def _Levenshtein_Dmatrix_initializer(len1, len2):

    Dmatrix = []

    for i in range(len1):
        Dmatrix.append([0] * len2)

    for i in range(len1):
        Dmatrix[i][0] = i

    for j in range(len2):
        Dmatrix[0][j] = j

    return Dmatrix


def _Levenshtein_cost_step(Dmatrix, string_1, string_2, i, j, substitution_cost=1):

    char_1 = string_1[i - 1]
    char_2 = string_2[j - 1]

    # insertion
    insertion = Dmatrix[i - 1][j] + 1
    # deletion
    deletion = Dmatrix[i][j - 1] + 1
    # substitution
    substitution = Dmatrix[i - 1][j - 1] + substitution_cost * (char_1 != char_2)

    # pick the cheapest
    Dmatrix[i][j] = min(insertion, deletion, substitution)


def levenshtein(string_1, string_2, substitution_cost=1):
    # get strings lengths and initialize Distances-matrix
    len1 = len(string_1)
    len2 = len(string_2)
    Dmatrix = _Levenshtein_Dmatrix_initializer(len1 + 1, len2 + 1)

    # compute cost for each step in dynamic programming
    for i in range(len1):
        for j in range(len2):
            _Levenshtein_cost_step(Dmatrix,
                               string_1, string_2,
                               i + 1, j + 1,
                               substitution_cost=substitution_cost)

    if substitution_cost == 1:
        max_dist = max(len1, len2)
    elif substitution_cost == 2:
        max_dist = len1 + len2

    return Dmatrix[len1][len2]

def scanpath_to_string(scanpath, height, width, n):

    height_step, width_step = height//n, width//n

    string = ''

    for i in range(np.shape(scanpath)[0]):
        fixation = scanpath[i].astype(np.int32)
        correspondent_square = (fixation[0] / width_step) + (fixation[1] / height_step) * n
        string += chr(97+int(correspondent_square))

    return string


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

