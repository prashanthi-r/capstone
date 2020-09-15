#!/usr/bin/python
import sys
from Config import Config as config
import socket

def send_val(sum1):
	
	if(config.partyNum == 0):
		ssock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		ssock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		ssock.bind((config.IP, config.PORT))
		ssock.listen(1)
		client, addr = ssock.accept()
		sum2 = float(client.recv(1024).decode())
		client.send(str(sum1).encode())
		client.close()
		ssock.close()
	else: 
		csock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		csock.connect((config.advIP,config.advPORT))
		csock.send(str(sum1).encode())
		sum2 = float(csock.recv(1024).decode())
		csock.close()
	return sum2

def addshares(a, b, mask):
	sum1 = (a + b) 
	sum2 = send_val(sum1)
	
	return sum1+sum2


# def multiplyshares():


# def matrixmul():


# command line argument - partyNum, integer input a, integer input b, mask value
# main
print(sys.argv)
config.partyNum = int(sys.argv[1])
a = float(sys.argv[2])
b = float(sys.argv[3])
mask = float(sys.argv[4])

if(config.partyNum == 0):
	config.PORT = 8002
	config.advPORT = 8003
else: 
	config.PORT = 8003
	config.advPORT = 8002


output = addshares(a,b,mask)
print(output)