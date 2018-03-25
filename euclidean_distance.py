import numpy as np

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

if __name__ == '__main__':
	human_scanpath = np.load('png/original_test.npy')[:,:2]
	simulated_scanpath= np.load('png/test.npy')[:,:2]
	EUD = euclidean_distance(human_scanpath, simulated_scanpath)
	print('EUD {:f}'.format(EUD))
