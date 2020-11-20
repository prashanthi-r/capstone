#!/usr/bin/python
import sys
from Config import Config as conf
import socket
import pickle 
import random
import numpy as np
import math

class functionalities:
	def floattoint64(x):
		# print(conf.converttoint64*(x))
		x = np.array(conf.converttoint64*(x), dtype = np.uint64)
		return x

	def int64tofloat(x):
		y=0
		if(x > (2**32)-1):
			x = (2**64) - x
			y = np.uint32(x)
			y = y*(-1)
		else:
			y = np.uint32(x)

		return float(y)/(1<<16)

	def send_val(send_info):
		if(conf.partyNum == 0):
			ssock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			ssock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			ssock.bind((conf.IP, conf.PORT))
			ssock.listen(1)
			while True:
				try:
					client, addr = ssock.accept()
					# print("Received connection ")
					break
				except:
					continue
			# client, addr = ssock.accept()
			# print("Size of send val: ",sys.getsizeof(send_info))
			recv_info = client.recv(4096)
			recv_info = pickle.loads(recv_info)
			client.send(pickle.dumps(send_info))
			client.close()
			ssock.close()
		else: 
			csock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			while True:
				try:
					csock.connect((conf.advIP,conf.advPORT))
					# print("Connected")
					break
				except: 
					continue
			# csock.connect((conf.advIP,conf.advPORT))
			# print("Size of send val: ",sys.getsizeof(send_info))
			csock.send(pickle.dumps(send_info))
			recv_info = pickle.loads(csock.recv(4096))
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
		c = (-1 * conf.partyNum * E * F) + (a * F) + (E * b) + z
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

		c = np.add(-1 * conf.partyNum * (np.multiply(E,F)),np.multiply(A*F) + np.multiply(E*B))
		C = reconstruct(c.tolist())

		C = (np.add(np.array(C),c).tolist())
		
		return C

	def truncate(x,scale):
		# if(conf.partyNum==0):
		# 	x = (math.floor(x)/scale)
		# else: 
		# 	x = 2**conf.l - x
		# 	x = (math.floor(x)/scale)
		# 	x = 2**conf.l - x
		# 	# x = np.uint64(-1*np.uint64(np.int64(-1*x)/scale))
		# return x
		if(conf.partyNum==0):
			x = np.uint64(x)/scale
		else: 
			# x = 2**conf.l - x
			# x = math.floor((x)/scale)
			# x = 2**conf.l - x
			x = np.uint64(-1*np.uint64(np.int64(-1*x)/scale))
		return np.uint64(x)

	def addvectors(A,B):
		m,n = A.shape
		# print(A)
		# print(B)
		C = np.array([[0]*n]*m)
		# print(C)
		for i in range(m):
			for j in range(n):
				print(i,j)
				C[i][j] = (A[i][j] + B[i][j])%(2**conf.l)
				print(C[i][j])
		# print(C)
		# print(np.array(C))
		return C

	def matrixmul_reg(A,B,E,F,V,Z):
		# A - data pt
		# B - weights
		# E = datapoint - data mask U
		# V - mask of weights for this batch
		# F = weights - weights mask V

		# print(type(A))
		# print(B.shape)
		# print(E.shape)
		# print(F.shape)
		# print(V.shape)
		# print(Z.shape)

	
		mul1 = (np.matmul(np.array(E, dtype = np.uint64),np.array(F, dtype = np.uint64)))
		mul2 = (np.matmul(np.array(A, dtype = np.uint64),np.array(F, dtype = np.uint64)))
		mul3 = (np.matmul(np.array(E, dtype = np.uint64),np.array(B, dtype = np.uint64)))
		# print("mul3: ", mul3)
		# print("Z: ", Z)

		Yhat1 = np.array(np.add(np.array(functionalities.floattoint64(-1 * conf.partyNum) * mul1, dtype = np.uint64),mul2))
		Yhat2 = np.array(np.add(mul3,Z),dtype=np.uint64)
		Yhat = np.array(np.add(Yhat1,Yhat2),dtype=np.uint64)

		return Yhat