import numpy as np


def to_list(input):
    if isinstance(input, (list, tuple)):
        return list(input)
    else:
        return [input]

def split(v, num_slices):
    """
    Split v as evently as possible
    :param v: An 1-D vector
    :return: list of vectors
    """
    # np.split(grads, self.num_worker would raise exception if grads cannot be evenly divided.
    #
    vshape = v.shape
    assert len(vshape) == 1, "we only accept 1D vector here"
    vector_len = vshape[0]
    slice_len = vector_len // num_slices
    slice_extra_len = vector_len % num_slices
    # i.e vector_len = 10, num_slices=20, then we should return [10]
    if slice_len == 0:
        return [v]
    else:
        result_slices = []
        for i in range(0, num_slices):
            len_tmp = slice_len + (1 if i < slice_extra_len else 0)
            # each time, it would accumulate 1 same as i, and the max value is slice_extra_len
            offset = i * slice_len + min(i, slice_extra_len)
            result_slices.append(v[offset:(offset + len_tmp)])
        return result_slices
