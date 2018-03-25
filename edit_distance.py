import numpy as np

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




if __name__=='__main__':
	human_scanpath = np.load('png/original_test.npy')[:,:2]
	predicted_scanpath = np.load('png/test.npy')[:,:2]
	
	human_str =  scanpath_to_string(human_scanpath, 768, 1024, 5)
	model_str =  scanpath_to_string(predicted_scanpath, 768, 1024, 5)
	
	ED = levenshtein(human_str, model_str, substitution_cost=1)
	
	print('ED: {:f}'.format(ED))
	
	
