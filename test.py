import torch
import numpy as np

a1 = torch.randn(3, 6, dtype=torch.float)
a2 = torch.randn(6, 12, dtype=torch.float)


def flatten_all(t_list):
    # result = torch.empty(0,0) # result
    sizes = []  # preserve sizes
    flats = []
    for a in t_list:
        size = a.size()
        sizes.append(size)
        a_f = a.flatten()
        flats.append(a_f)
    flattened = np.concatenate(flats)
    return flattened, sizes


def unflatten_all(t_list, sizes):
    tensors = []
    offset = 0
    for i in range(0, len(sizes)):
        size = sizes[i]
        total = 1
        for n in size:
            total = total * n
        next_offset = offset + total
        slice_ = t_list[offset:next_offset]
        t_flat = torch.FloatTensor(slice_)
        t_shaped = t_flat.view(size)
        tensors.append(t_shaped)
        offset = next_offset
    return tensors


raw_tensors = [a1, a2]
print("flattening...")
flats, sizes = flatten_all(raw_tensors)
print("flats: ", flats)
print("sizes: ", sizes)

print("unflattening...")
tensors = unflatten_all(flats, sizes)
print(tensors)

for i in range(0, len(tensors)):
    print("is equal: ", i, " -> ", raw_tensors[i].equal(tensors[i]))
