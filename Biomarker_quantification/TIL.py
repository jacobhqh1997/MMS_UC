import numpy as np

def TIL_Score(prob_map_path, cell_size):
    # prob_map: MxNx8 numpy array contains the probabilities
    # cell_size: number of patch to be consider as one grid-cell

    prob_map = np.load(prob_map_path)
    pred_map = np.zeros((prob_map.shape[0],prob_map.shape[1]))
    for i in range(prob_map.shape[0]):
        for j in range(prob_map.shape[1]):
            pred_map[i][j] = np.argmax(prob_map[i,j,:])
            # print(multi_classes[i][j])
            if prob_map[i,j,0] == 0 and pred_map[i][j] == 0:#for 'empty' class is it necessary to be 0  
                pred_map[i][j] = 1

    T = np.int8(pred_map == 0)  
    L = np.int8(pred_map == 6)  

    [rows, cols] = T.shape
    stride = np.int32(cell_size / 2)

    t = np.zeros(len(range(0, rows - cell_size + 1, stride))*len(range(0, cols - cell_size + 1, stride)))
    l = np.zeros(len(range(0, rows - cell_size + 1, stride))*len(range(0, cols - cell_size + 1, stride)))
    k = 0

    for i in range(0, rows - cell_size + 1, stride):
        for j in range(0, cols - cell_size + 1, stride):
            t[k] = np.mean(np.mean(T[i:i + cell_size, j:j + cell_size]))
            l[k] = np.mean(np.mean(L[i:i + cell_size, j:j + cell_size]))
            k += 1

    index = np.logical_and(t == 0, l == 0)
    index = np.where(index)[0]

    t = np.delete(t, index)
    l = np.delete(l, index)

    til_score = 0.0
    coloc_score_l = 0.0 
    if len(t) == 0:  # ideally each WSI should have some tumour o
        til_score = 1  # if there is no tumour then its good for patients long term survival
    else:
        # normalizaing the percentage of tumour and lymphocyte range to [0-1] in a grid-cell
        t = t/(t + l)
        l = l/(t +l)

        # Morisita-Horn Index based colocalization socre
        coloc_score_l = (2 * sum(t*l)) / (sum(t**2) + sum(l**2))
        if np.sum(t) == 0:
            til_score = 1 # w
        else:
            l2t_ratio = np.sum(l) / np.sum(t)  #
            til_score = 0.5 * coloc_score_l * l2t_ratio  
    return til_score, coloc_score_l

prob_map_path = "path/to/your/prob_map.npy"  
cell_size = 10  


til_score, coloc_score = TIL_Score(prob_map_path, cell_size)

print(f"TIL: {til_score}, coloc_score: {coloc_score}")
