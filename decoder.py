import numpy as np

def decoder(scanpath, vocab, stats):
	output = []
	for scan in scanpath:
		if scan != 0 or scan !=1:
			x_idx, y_idx, dur_idx = np.where(vocab == scan)	
			x_mean, x_std = stats[0][x_idx[0]], stats[3][x_idx[0]]
			y_mean, y_std = stats[1][y_idx[0]], stats[4][y_idx[0]]
			dur_mean, dur_std = stats[2][dur_idx[0]], stats[5][dur_idx[0]]
			
			x   = np.random.normal(x_mean, x_std/2)
			y   = np.random.normal(y_mean, y_std/2)
			dur = np.random.normal(dur_mean, dur_std/2)
			
			output.append([int(x), int(y), dur])

	return np.array(output)
if __name__=='__main__':
	vocab = np.load('data/MIT1003-vocab.npy')
	stats = np.load('data/MIT1003-stat.npy', encoding='latin1')
	print(decoder([200, 1054], vocab, stats))
