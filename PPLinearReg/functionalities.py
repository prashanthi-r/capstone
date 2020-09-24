#!/usr/bin/python
import sys
from Config import Config as config
import socket
import pickle 
import random
import numpy as np
from mod import Mod

class functionalities:
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
		sum1 = (a + b + mask) 
		sendlist.append(sum1)
		sum2 = send_val(sendlist)
		
		return sum1+sum2[0]

	def reconstruct(c):
		C = functionalities.send_val(c)
		return C

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
		sendlist=[]
		sendlist.append(c)
		C = reconstruct(sendlist)
		return C[0]+c

	def matrixadd(A,B,mask):
		sum1 = np.add(np.array(A),np.array(B))
		sum2 = send_val(sum1.tolist())		

		return (np.add(np.array(sum2)),sum1).tolist()

	def matrixmul(A,B,U,V,Z):
		A = np.array(A)
		B = np.array(B)
		U = np.array(U)
		V = np.array(V)
		
		E = np.subtract(A,U)
		F = np.subtract(B,V)

		sendlist = []
		sendlist.append(E.tolist())
		sendlist.append(F.tolist())
		recv_info = send_val(sendlist)

		# recv_e = send_val(E.tolist())
		# recv_f = send_val(F.tolist())
		E = E + recv_info[0]
		F = F + recv_info[1]

		c = np.add(-1 * config.partyNum * (np.multiply(E,F)),np.multiply(A*F) + np.multiply(E*B))
		C = reconstruct(c.tolist())

		C = (np.add(np.array(C),c).tolist())
		
		return C

	def matrixmul_reg(A,B,E,F,V,Z):
		A = np.array(A) #data pt
		B = np.array(B) #weights
		E = np.array(E) # datapoint - mask
		F = np.array(F) 
		V = np.array(V) #mask of weights for this batch
		Z = np.array(Z) #u[j]v[j] ->weight's mask for that batch

		# F = np.subtract(B,V)
		# recv_f = send_val(F.tolist())
		# F = F + recv_f[0]
		
		Yhat1 = np.add(-1 * config.partyNum * (np.multiply(E,F)),np.multiply(A,F))
		Yhat2 = np.add(np.multiply(E,B),Z)
		Yhat = np.add(Yhat1,Yhat2)

		return Yhat
