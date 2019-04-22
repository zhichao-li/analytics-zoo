import numpy as np

def split(v, num_slices):
    """
    Split v as evently as possible
    :param v: An 1-D vector
    :return: list of vectors
    """
    # the split method here as it would raise exception if cannot be evently divided.
    # np.split(grads, self.num_worker)
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
            # each time ,it would accumulate 1 same as i, and the max value is slice_extra_len
            offset = i * slice_len + min(i, slice_extra_len)
            result_slices.append(v[offset:(offset + len_tmp)])
        return result_slices



# def split_by_shapes(v, shapes):
#     i = 0
#     arrays = []
#     for shape in shapes:
#         size = np.prod(shape, dtype=np.int)
#         array = v[i:(i + size)].reshape(shape)
#         arrays.append(array)
#         i += size
#     assert len(v) == i, "Passed vector does not have the correct shape."
#     return arrays