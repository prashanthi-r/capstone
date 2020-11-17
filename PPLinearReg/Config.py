import socket
import numpy as np

class Config:
	PORT = -1
	partyNum = -1
	IP = socket.gethostbyname(socket.gethostname())
	# IP='192.168.0.152'
	#IP=(socket.gethostbyname_ex(socket.gethostname())[-1])[1]
	advIP = IP 
	advPORT = -1
	l = 64
	lby2 = 32
	modl = 2**l
	precision = 13
	converttoint64 = (1<<precision)
	trunc_parameter = (1>>precision)
	epochs = 1

	# data specific
	n=-1
	d=-1
	t=-1
	batchsize=1
	alpha = 0.01 # learning rate
	alpha_inv = (1/alpha)
	train = int(506*(80/100))
	test = 506 - train

	# def floattoint64(x):
	# 	# print(conf.converttoint64*(x))
	# 	x = ((1<<16)*(x))
	# 	return np.uint64(x)

	# def int64tofloat(x):
	# 	x = (float(np.int64(x))/(1<<16))
	# 	return x