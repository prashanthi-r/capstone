#!/usr/bin/python
import sys
from Config import Config as config
import socket
import pickle 

def send_val(send_info):
	
	if(config.partyNum == 0):
		ssock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		ssock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		ssock.bind((config.IP, config.PORT))
		ssock.listen(1)
		client, addr = ssock.accept()
		recv_info = pickle.loads(client.recv(1024))
		client.send(pickle.dumps(send_info))
		client.close()
		ssock.close()
	else: 
		csock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		csock.connect((config.advIP,config.advPORT))
		csock.send(pickle.dumps(send_info))
		recv_info = pickle.loads(csock.recv(1024))
		csock.close()
	return recv_info

def addshares(a, b, mask):
	sendlist = []
	sum1 = (a + b) 
	sendlist.append(sum1)
	sum2 = send_val(sendlist)
	
	return sum1+sum2[0]

def reconstruct(c):
	sendlist=[]
	sendlist.append(c)
	C = send_val(sendlist)
	return C[0]

def multiplyshares(a,b,u,v,z):
	sendlist = []
	e = a - u
	f = b - v
	sendlist.append(e)
	sendlist.append(f)
	recv_info = send_val(sendlist)
	E = e + recv_info[0]
	F = f + recv_info[1]
	c = (-1 * config.partyNum * E * F) + (a * F) + (E * b) + z
	C = reconstruct(c)
	return c+C
# def matrixmul():


# command line argument - partyNum, integer input a, integer input b, mask value
# main
print(sys.argv)
config.partyNum = int(sys.argv[1])
a = float(sys.argv[2])
b = float(sys.argv[3])
u = float(sys.argv[4])
v = float(sys.argv[5])
z = float(sys.argv[6])

if(config.partyNum == 0):
	config.PORT = 8002
	config.advPORT = 8003
else: 
	config.PORT = 8003
	config.advPORT = 8002


#output = addshares(a,b,u)
output_mul = multiplyshares(a,b,u,v,z)
print(output_mul)