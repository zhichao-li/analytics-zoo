from keras.models import *
from keras.layers import *
model = Sequential()
model.add(Dense(10, kernel_initializer="glorot_uniform", input_shape=(8, )))
model.add(Dense(12))
from keras.models import load_model
model.save('/tmp/seq.h5')  # creates a HDF5 file 'my_model.h5'
