import numpy as np

n_samples = 4
seq_pairs    = []
multi_matchs = []
for i in range(n_samples):
	seq_1_len = np.maximum(3, np.random.randint(15))
	seq_2_len = np.maximum(3, np.random.randint(15))
	seq_1 = np.random.randn(seq_1_len, 3)
	seq_2 = np.random.randn(seq_2_len, 3)
	muli_match_score = np.random.randn(1,5).T
	multi_matchs.append(muli_match_score)
	seq_pairs.append([seq_1,  seq_2])
seq_pairs    = np.array(seq_pairs)
multi_matchs = np.array(multi_matchs)[:,:,0]

np.save('seq_pairs.npy', seq_pairs)
np.save('multi_matchs.npy', multi_matchs)

print(seq_pairs.shape, multi_matchs.shape)
	
	
