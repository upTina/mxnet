import numpy as np 
import os,sys
import mxnet as mx
import gzip
import struct
import logging

def read_data(label_url, image_url):
	with gzip.open(label_url) as flbl:
		magic, num = struct.unpack(">II", flbl.read(8))
		label = np.fromstring(flbl.read(), dtype = np.int8)
	with gzip.open(image_url, 'rb') as fimg:
		magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
		image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
	return (label, image)

(train_lbl, train_img) = read_data('/home/shang/dataset/mnist/train-labels-idx1-ubyte.gz', '/home/shang/dataset/mnist/train-images-idx3-ubyte.gz')
(val_lbl, val_img) = read_data('/home/shang/dataset/mnist/t10k-labels-idx1-ubyte.gz','/home/shang/dataset/mnist/t10k-images-idx3-ubyte.gz')

def to4d(img):
	return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32) / 255

batch_size = 100
train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)

data = mx.sym.Variable('data')
data = mx.sym.Flatten(data=data)
fc1 = mx.sym.FullyConnected(data = data, name = 'fc1', num_hidden = 128)
act1 = mx.sym.Activation(data=fc1, name='relu1', act_type='relu')
fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden = 64)
act2 = mx.sym.Activation(data=fc2, name='relu2', act_type='relu')
fc3 = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden = 10)
mlp = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

logging.getLogger().setLevel(logging.DEBUG)
model = mx.model.FeedForward(
	ctx = mx.gpu(),
	symbol = mlp,
	num_epoch = 10,
	learning_rate = 0.1
)

model.fit(
	X = train_iter,
	eval_data = val_iter,
	batch_end_callback = mx.callback.Speedometer(batch_size, 200)
)
