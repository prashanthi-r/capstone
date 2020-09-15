import socket

class Config:
	PORT = -1
	partyNum = -1
	IP = socket.gethostbyname(socket.gethostname())
	advIP = IP 
	advPORT = -1
	l = 64