import numpy as np
import math
import scipy.sparse
from scipy.sparse import *


def multiway_merge(condensed_inputs, N_WAY = 64):
    total_columns = len(condensed_inputs)
    N_ROUND = int(math.ceil(total_columns/N_WAY))
    condensed_outputs = []

    for i in range(N_ROUND):
        current_inputs = condensed_inputs[i*N_WAY: (i+1)*N_WAY]
        list_of_arrays = [np.array(current_inputs[j]) for j in range(len(current_inputs))]
        # merge the input lists
        current_outputs = np.concatenate(list_of_arrays)
        current_outputs = current_outputs[current_outputs[:, 0].argsort(), :]
        # add the elements with same row_index together
        _, start_ind_list = np.unique(current_outputs[:, 0], return_index=True)
        for j in range(len(start_ind_list)):

            if j != len(start_ind_list)-1:
                current_outputs[start_ind_list[j], 1] = current_outputs[start_ind_list[j]:start_ind_list[j+1], 1].sum()
            else:
                current_outputs[start_ind_list[j], 1] = current_outputs[start_ind_list[j]:, 1].sum()
        current_outputs = current_outputs[start_ind_list, :]
        condensed_outputs.append(current_outputs)
    if N_ROUND == 1:
        return condensed_outputs
    else:
        return multiway_merge(condensed_outputs, N_WAY)

def compute_ref_results(mat_csc,vector):
    output_vector = np.zeros((mat_csc.shape[0], 1))
    for i in range(vector.shape[0]):
        vector_val = vector[i, 1]
        col_idx = int(vector[i, 0])
        start_ind = mat_csc.indptr[col_idx]
        end_ind = mat_csc.indptr[col_idx+1]
        for j in range(start_ind, end_ind):
            row_id = mat_csc.indices[j]
            data = mat_csc.data[j]
            output_vector[row_id] += data*vector_val
    return output_vector

def checking_results(merged_r1, merged_r2, golden_results):
    merged_r1 = merged_r1[0]
    merged_r2 = merged_r2[0]
    results_r1 = np.zeros((golden_results.size, 1))
    results_r2 = np.zeros((golden_results.size, 1))
    for i in range(merged_r1.shape[0]):
        results_r1[int(merged_r1[i, 0])] = merged_r1[i,1]
    for i in range(merged_r2.shape[0]):
        results_r2[int(merged_r2[i, 0])] = merged_r2[i, 1]
    for i in range(golden_results.size):
        if( abs( results_r1[i] - golden_results[i])>0.0001 or abs( results_r2[i] - golden_results[i])>0.0001 ):
            print("mode0_r: {}, mode1_r: {}, golden_r {}".format( results_r1[i], results_r2[i] , golden_results[i] ))
            print("Check failed...")
            assert False
    print("Check passed...")

def read_file(filename, type):
    data_list = []
    with open(filename) as f:
        curr_line = f.readline()
        while curr_line:
            data_list.append([float(x) for x in curr_line.split()])
            curr_line = f.readline()
    if type == "matrix":
        input_matrix = csc_matrix((data_list[0], data_list[1], data_list[2]))
        return input_matrix
    if type == "vector":
        return np.asarray(data_list, dtype=np.float32)
    if type == "packing":
        return data_list
