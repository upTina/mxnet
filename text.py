import mxnet.ndarray as nd
import mxnet.autograd as ag
x = nd.array([[1,2],[3,4]])
x.attach_grad()
with ag.record():
	y = x * 2
	z = y * x
z.backward()
print('x.grad:',x.grad)
x.grad == 4*x
