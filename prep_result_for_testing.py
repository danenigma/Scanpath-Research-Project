import numpy as np

def decode(scanpath, vocab, stats):

	output = []
	for scan in scanpath:
	
		x_idx, y_idx, dur_idx = np.where(vocab == scan)	
		
		x_mean, x_std = stats[0][x_idx[0]], stats[3][x_idx[0]]
		y_mean, y_std = stats[1][y_idx[0]], stats[4][y_idx[0]]
		dur_mean, dur_std = stats[2][dur_idx[0]], stats[5][dur_idx[0]]
		
		x   = np.random.normal(x_mean, x_std/2)
		y   = np.random.normal(y_mean, y_std/2)
		dur = np.random.normal(dur_mean, dur_std/2)
		
		output.append([int(x), int(y), dur])

	return np.array(output)


vocab  = np.load('data/MIT1003-vocab.npy')
stats  = np.load('data/MIT1003-stat.npy', encoding='latin1')


test_subj = np.load('data/scanpaths.npy')
test_other_subjs = np.load('data/MIT1003-labels.npy', encoding='latin1')

#print(test_other_subjs[:, 1])
np.random.seed(100)
counter = 0
for i in range(100):
	index     = np.random.randint(200)
	pred_scan = test_subj[index][0]
	try:
		dur       = pred_scan[:,3]-pred_scan[:,2]*1000
		print(dur)
	except:
		pass
	 

