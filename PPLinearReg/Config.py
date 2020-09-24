import socket

class Config:
	PORT = -1
	partyNum = -1
	IP = socket.gethostbyname(socket.gethostname())
	advIP = IP 
	advPORT = -1
	l = 32
	modl = 2**l
	n=-1
	d=-1
	t=-1
	batchsize=1
	alpha = 0.01 # learning rate