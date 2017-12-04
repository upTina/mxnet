def SGD(params, lr):
	for param in params:
		param[:] = param - lr*param.grad
