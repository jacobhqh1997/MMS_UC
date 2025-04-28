import pandas as pd
import os
def TIM_Score(prob_map_path, cell_size):
    # prob_map: MxNx8 numpy array contains the probabilities
    # cell_size: number of patch to be consider as one grid-cell
    prob_map = np.load(prob_map_path)
    pred_map = np.zeros((prob_map.shape[0],prob_map.shape[1]))
    for i in range(prob_map.shape[0]):
        for j in range(prob_map.shape[1]):
            pred_map[i][j] = np.argmax(prob_map[i,j,:])
            # print(multi_classes[i][j])
            if prob_map[i,j,0] == 0 and pred_map[i][j] == 0:
                pred_map[i][j] = 1
    T = np.int8(pred_map == 0)  # patches predicted as tumour
    M = np.int8(pred_map == 4)  # patches predicted as musele
    [rows, cols] = T.shape
    stride = np.int32(cell_size / 2)
    t = np.zeros(len(range(0, rows - cell_size + 1, stride))*len(range(0, cols - cell_size + 1, stride)))
    m = np.zeros(len(range(0, rows - cell_size + 1, stride))*len(range(0, cols - cell_size + 1, stride)))
    k = 0
 
    for i in range(0, rows - cell_size + 1, stride):
        for j in range(0, cols - cell_size + 1, stride):
            t[k] = np.mean(np.mean(T[i:i + cell_size, j:j + cell_size]))
            m[k] = np.mean(np.mean(M[i:i + cell_size, j:j + cell_size]))
            k += 1

    index = np.logical_and(t == 0, m == 0)
    index = np.where(index)[0]
    t = np.delete(t, index)
    m = np.delete(m, index)
    tim_score = 0.0
    coloc_score = 0.0 
    if len(t) == 0:  
        tim_score = 0  
    else:
        
        t = t/(t + m)
        m = m/(t + m)
   
        coloc_score = (2 * sum(t*m)) / (sum(t**2) + sum(m**2))
        if np.sum(t) == 0:
            tim_score = 1 
        else:
            l2t_ratio = np.sum(m) / np.sum(t)  
            tim_score = 0.5 * coloc_score * l2t_ratio  
    return tim_score,coloc_score

prob_map_path = "path/to/your/prob_map.npy"  
cell_size = 10  


TIR_score, Coloc_M = TIM_Score(prob_map_path, cell_size)

print("Colocalization Score:", Coloc_M )
