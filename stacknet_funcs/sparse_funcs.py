def from_sparse_to_file(filename, array, deli1=" ", deli2=":", ytarget=None):

    from scipy.sparse import csr_matrix
    import numpy as np

    zsparse = csr_matrix(array)
    indptr = zsparse.indptr
    indices = zsparse.indices
    data = zsparse.data
    print(" data lenth %d" % (len(data)))
    print(" indices lenth %d" % (len(indices)))
    print(" indptr lenth %d" % (len(indptr)))

    f = open(filename, "w")
    counter_row = 0
    for b in range(0, len(indptr) - 1):
        # if there is a target, print it else , print nothing
        if ytarget is not None:
            f.write(str(ytarget[b]) + deli1)

        for k in range(indptr[b], indptr[b + 1]):
            if (k == indptr[b]):
                if np.isnan(data[k]):
                    f.write("%d%s%f" % (indices[k], deli2, -1))
                else:
                    f.write("%d%s%f" % (indices[k], deli2, data[k]))
            else:
                if np.isnan(data[k]):
                    f.write("%s%d%s%f" % (deli1, indices[k], deli2, -1))
                else:
                    f.write("%s%d%s%f" % (deli1, indices[k], deli2, data[k]))
        f.write("\n")
        counter_row += 1
        if counter_row % 10000 == 0:
            print(" row : %d " % (counter_row))
    f.close()

