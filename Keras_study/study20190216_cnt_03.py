
import h5py
f = h5py.File('cnt_01_W.h5', 'r')

print("Key: ", list(f.keys()))

# https://pypi.org/project/h5pyViewer/
# d = f['concatenate_1', 'dense_1', 'dense_2', 'dense_3', 'dense_4', 'dense_5', 'dense_6', 'dense_7', 'dense_8', 'dense_9', 'input_1', 'input_2', 'input_3']
# print(d)

# print(f['dense_1'].shape)
# print(f['dense_1'][:])

# w, b= model.get_weights()
# print(w, b)

