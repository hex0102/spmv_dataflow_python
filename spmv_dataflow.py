# This code is intended to show the dataflow of SPMV
# feature 1: matrix condensing
# feature 2: inserted chasing pointer entry
# 2 modes

import numpy as np
import sys
import scipy.sparse
from scipy.sparse import *
from spmv_utils import *
import optparse

if __name__ == "__main__":

    parser = optparse.OptionParser("usage: python3 spmv_dataflow.py \
                                   matrix_filename vector_filename packing_filename")

    parser.add_option("-f", "--matrix", dest="matrix_filename", default = "twitter_combined.csr",\
                      type = "string", help = "specify csc matrix")
    parser.add_option("-v", "--vector", dest="vector_filename", default = "twitter_combined.vector",\
                      type = "string", help = "specify vector")
    parser.add_option("-p", "--packing", dest="packing_filename", default = "twitter_combined.packing",\
                      type = "string", help = "specify packing file")
    (options, args) = parser.parse_args()


    matrix_filename = options.matrix_filename
    vector_filename = options.vector_filename
    packing_filename = options.packing_filename

    mat_csc = read_file(matrix_filename, 'matrix')
    n_unique = mat_csc.shape[0]
    vector = read_file(vector_filename, 'vector').transpose()
    packing_group = read_file(packing_filename, 'packing')

    golden_results = compute_ref_results(mat_csc,vector)

    condensed_groups = []
    # Create condense column groups using CSC matrix and the packing info
    # Output:  condensed_groups   (format: [ list0( [row, col, val],[row, col, val]... ),  list1, ... listn-1])
    for i in range(len(packing_group)):
        curr_group = np.zeros((0, 3))
        current_column_group_idx = packing_group[i]
        for j in range(len(current_column_group_idx)):
            curr_column = int(current_column_group_idx[j])
            start_ind = mat_csc.indptr[curr_column]
            end_ind = mat_csc.indptr[curr_column+1]
            column_length = end_ind - start_ind
            if column_length != 0:
                vals = mat_csc.data[start_ind:end_ind].reshape(column_length, 1)
                row_idx = mat_csc.indices[start_ind:end_ind].reshape(column_length, 1).astype(float)
                col_idx = (np.ones(end_ind-start_ind)*curr_column).reshape(column_length, 1).astype(float)
                current_coo = np.concatenate((row_idx, col_idx, vals), axis = 1)
                curr_group = np.vstack([curr_group, current_coo])
        curr_group = curr_group[curr_group[:, 0].argsort(),:]
        condensed_groups.append(curr_group)

    print("Build the pointer array for the purpose of skipping entries")
    # print(len(condensed_groups))
    pointer_list = []
    for i in range(len(condensed_groups)):
        current_group = condensed_groups[i]
        current_pointer = []
        condense_group_col_idx = current_group[:, 1]
        current_column_idx  = np.unique(condense_group_col_idx)
        for j in range(current_column_idx.size):
            curr_column = current_column_idx[j]
            matching_ind = (condense_group_col_idx==curr_column).nonzero()[0]
            idx = ((matching_ind[1:] - matching_ind[:-1])>1).nonzero()[0] + 1
            jump_pointer = np.append(matching_ind[0], matching_ind[idx])
            current_pointer.append(jump_pointer)
        pointer_list.append(current_pointer)

    print_frequency = 500
    print("Mode0: Streaming down the condensed columns")
    enable_mode0 = 1
    if enable_mode0:
        multiplied_groups_m0 = []
        for i in range(len(condensed_groups)):
            if i%print_frequency == 0:
                print("processing condensed group number: {}".format(i))
            current_group = condensed_groups[i]
            current_multiplied_group = []
            # load vector entries into avaiable_vector_elements
            avaiable_vector_elements = np.zeros((len(packing_group[i]), 2))
            avaiable_vector_elements[:, 0] = np.array(packing_group[i]).astype(int)
            intersect_idx_array = (np.in1d(vector[:, 0], avaiable_vector_elements[:, 0])).nonzero()[0]

            used_vector_entries = vector[intersect_idx_array, :]
            idx_array = used_vector_entries[:, 0]
            val_array = used_vector_entries[:, 1]
            matched_ind = np.in1d(avaiable_vector_elements[:, 0], idx_array).nonzero()[0]
            avaiable_vector_elements[matched_ind, 1] = val_array

            for j in range(current_group.shape[0]):
                row_idx = current_group[j, 0]
                col_idx = current_group[j, 1]
                val = current_group[j, 2]
                used_idx = np.argwhere(avaiable_vector_elements[:, 0]==col_idx)[0][0]
                val = avaiable_vector_elements[used_idx, 1] *val
                current_multiplied_group.append([row_idx, val])
            multiplied_groups_m0.append(current_multiplied_group)

        final_merge = multiway_merge(multiplied_groups_m0, 64)

    print("Mode1: using sparse vector to select the NNZs in the condensed format")
    # mode 1 for sparse vector:
    multiplied_groups_m1 = []
    for i in range(len(condensed_groups)):
        if i % print_frequency == 0:
            print("processing condensed group number: {}".format(i))
        current_multiplied_group = []
        current_group = condensed_groups[i]
        xy, x_ind, y_ind = np.intersect1d(vector[:,0], packing_group[i], return_indices=True)

        # pointers
        current_pointer_set = [ pointer_list[i][index] for index in list(y_ind) ]
        # corresponding vector element value
        useful_vector_elements = vector[x_ind, :] #need to check if x_ind correlates to y_ind

        current_pointer_cc = np.zeros(len(current_pointer_set)).astype(int)
        current_ind = np.zeros(len(current_pointer_set)).astype(int)
        out_of_bound = np.ones(len(current_pointer_set))
        pointer_bound = [len(current_pointer_set[m]) for m in range(len(current_pointer_set))]
        while out_of_bound.sum() != 0:
            for m in range(current_ind.size):
                current_ind[m] = current_pointer_set[m][current_pointer_cc[m]]
            current_column = np.argmin(current_ind)
            current_value = useful_vector_elements[current_column, 1]
            start_ind = np.min(current_ind)
            while True:
                current_multiplied_group.append([current_group[start_ind, 0], current_group[start_ind, 2]*current_value])

                if start_ind == current_group.shape[0]-1:
                    break
                if current_group[start_ind, 1] != current_group[start_ind + 1, 1]:
                    break
                else:
                    start_ind += 1

            if(current_pointer_cc[current_column] == pointer_bound[current_column]-1):
                out_of_bound[current_column] = 0
                current_pointer_set[current_column][current_pointer_cc[current_column]] = n_unique + 100
            else:
                current_pointer_cc[current_column] += 1
        if( x_ind.shape[0] != 0 ):  #no intersection
            multiplied_groups_m1.append(current_multiplied_group)

    final_merge2 = multiway_merge(multiplied_groups_m1, 64)

    checking_results(final_merge,final_merge2,golden_results)
    print("finished...")



























    #Merge:
    N_WAY_MERGE = 128










