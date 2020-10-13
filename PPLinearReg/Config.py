import socket

class Config:
	PORT = -1
	partyNum = -1
	IP = socket.gethostbyname(socket.gethostname())
	advIP = IP 
	advPORT = -1
	l = 64
	lby2 = 32
	modl = 2**l
	precision = 16
	converttoint64 = (1<<precision)
	trunc_parameter = (1>>precision)

	# data specific
	n=-1
	d=-1
	t=-1
	batchsize=1
	alpha = 0.01 # learning rate
	train = int(506*(80/100))
	test = 506 - train