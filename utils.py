import pandas as pd
import numpy  as np 
import os

def valid_scan(scanpath, valid_max_duration=0.705):
	if scanpath[2]<valid_max_duration:
		return True
	else:
		return False

def get_scanpath(file_path):

	start_token = 1
	end_token   = 2

	vocab    = np.load('data/MIT1003-vocab.npy', encoding='latin1')
	binedges = np.load('data/MIT1003-binedges.npy', encoding='latin1')
	
	scanpath_file = open(file_path)
	scanpath_file_lines = scanpath_file.readlines()
	scanpath = np.zeros((len(scanpath_file_lines),4))
	for i in range(len(scanpath)):
		scanpath[i] = np.array(scanpath_file_lines[i].split()).astype(np.float)
	
	scanpath[:, 2] = scanpath[:,3]-scanpath[:,2]
		
	trans=[]
	trans.append(start_token)
	trans.extend(bin(scanpath[:, :3], vocab, binedges))
	trans.append(end_token)
	return trans
			
def bin(scanpath, vocab, binedges, valid_max_duration=0.705):
	words = []
	for scan in scanpath:
	
		if valid_scan(scan, valid_max_duration):
			x_digitized = np.digitize(scan[0], binedges[0])-1
			y_digitized = np.digitize(scan[1], binedges[1])-1
			duration_digitized = np.digitize(scan[2], binedges[2])-1
			words.append(vocab[x_digitized, y_digitized, duration_digitized])

	return words
	


if __name__=='__main__':
	
	img_dir        = 'data/FixaTons/MIT1003/STIMULI'
	scanpath_dir   = 'data/FixaTons/MIT1003/SCANPATHS'

	img_paths = os.listdir(img_dir)
	data_table = []
	for img_name in img_paths:
		img_      = img_name.split('.')[0]
		scanpaths  = os.listdir(os.path.join(scanpath_dir, img_))
	
		for scan in scanpaths:
			data_table.append([os.path.join(img_dir, img_name), os.path.join(scanpath_dir, img_,scan)])

	df = pd.DataFrame(np.array(data_table), columns=['image_path', 'scanpath'])
	train_table = df.sample(frac=0.80) #70-30 split
	val_table   = df[~df['scanpath'].isin(train_table['scanpath'])]
	train_table_name = 'data/train_table.pkl'
	val_table_name   = 'data/val_table.pkl'

	train_table.to_pickle(train_table_name) 
	val_table.to_pickle(val_table_name)


	scanpath_path = val_table.iloc[0, 1]
	print(scanpath_path)
	trans=get_scanpath(scanpath_path)
	print(trans)

