from Config import Config as conf
import sys
import itertools
import numpy as np
from functionalities import functionalities as func
import random
import math

class linearReg:

	def readData(filename_data,filename_mask):
		# X n*d matrix
		# Y n*1 matrix
		# w d*1 matrix
		# U n*d matrix
		# V d*t matrix
		# V' 1*t matrix mask the difference Y - Y^
		# Z 1*n matrix
		# Z'd*t matrix (U_b transpose * V'[i]) 

		mask=[]
		X=[]    
		Y=[]
		U=[]
		V=[]
		V_dash=[] 
		Z=[]
		Z_dash=[]
		
		i=0
		with open(filename_data,'r+') as f:
			for line in f:
				row=line.split()
				i=i+1
				if(i<=6):
					row= [np.uint64(i) for i in row]
					X.append(row)
				else:
					Y.append(np.uint64(row[0].rstrip()))
			f.close()
		# print("Y: ",Y)

		conf.n = len(Y)
		conf.d = len(X[0])
		conf.t = conf.n
		# print("n: ",conf.n)
		# print("d: ",conf.d)
		# print("t: ",conf.t)


		with open(filename_mask,'r') as f:
			for line in f:
				row=line.split()
				row=[int(i, base=10) for i in row]
				mask.append(row)
			f.close()


		n = conf.n
		d = conf.d
		t = conf.t
		b = conf.batchsize

		U = mask[:n]
		V = mask[n:n+d]
		Vdash = mask[n+d: n+d+1]
		Z = mask[n+d+1:n+d+2]
		Zdash=mask[n+d+2:]
		
		return X,Y,U,V,Vdash,Z,Zdash

	def SGDLinear(X,Y,U,V,VDash,Z,ZDash):
		
		#ZDash = (np.matmul(np.array(U).transpose,VDash).tolist())
		# print(np.array(X))
		# print(np.array(U))
		# print(np.array(Y))
		X = (np.array(X))
		Y = (np.array(Y))

		U = np.array(U)
		V = np.array(V)
		VDash = np.array(VDash)
		Z = np.array(Z)
		ZDash = np.array(ZDash)

		E1 = np.uint64(np.subtract(X,U))
		E2 = np.uint64(func.reconstruct(E1.tolist()))
		E = np.uint64(np.add(E1,np.array(E2)))
		# randomly initialise weights vector
		weights = np.array((np.random.rand(conf.d)))
		weights = weights.reshape(conf.d,1)
		# print('Weights: ',weights)
		weights2 = func.reconstruct(weights)

		wts = weights+weights2
		print("Initial weights: ")
		print(wts)


		weights = func.floattoint64(weights)
		# print(weights)

		# print(Y)
		for e in range(conf.epochs):
			# print("e: ", e)
			loss = 0.0

			for j in range(conf.t): 
				X_B = np.array(X[j:j+conf.batchsize])
				Y_B = np.array([Y[j:j+conf.batchsize]]).transpose()
				# print(Y_B)
				E_B = np.array(E[j:j+conf.batchsize])
				V_j = np.array([V[:,j]]).transpose()	# d*1
				Z_j = np.array([Z[:,j]]).transpose()  	#|B| * 1
				Vdash_j = np.array([VDash[:,j]]).transpose()
				Zdash_j = np.array([ZDash[:,j]]).transpose()

				F1 = np.uint64(np.subtract(weights,V_j))
				F2 = func.reconstruct(F1.tolist())
				F = np.uint64(np.add(F1,np.array(F2))) #d*1 as its weights-V_j (both of dim d*1)
				# print("Vj shape: ",V_j.shape)
				# print("Zj shape: ",Z_j.shape)

				YB_dash = func.matrixmul_reg(X_B,weights,E_B,F,V_j,Z_j) #|B|*1
				# print('YB Dash shape: ',YB_dash.shape)					
				#YB_dash[0][0] = func.truncate(YB_dash[0][0],conf.converttoint64)

				D_B = np.uint64(np.subtract(YB_dash,Y_B))
				# print("Y_B: ",Y_B)
				# print("YB_dash: ",YB_dash)
				# print("D_B: ",D_B)
				yb = func.int64tofloat(Y_B[0][0])
				yb_dash = func.int64tofloat(YB_dash[0][0])
				# print(yb)
				# print(yb_dash)
				dif = (yb_dash - yb)
				# print(dif)
				loss = loss+(dif*dif)
				
				Fdash_1 = np.uint64(np.subtract(D_B,Vdash_j))
				Fdash_2 = func.reconstruct(Fdash_1)
				FDash = np.uint64(np.add(Fdash_1,np.array(Fdash_2)))

				X_BT = np.array(X_B).transpose() 
				E_BT = np.array(E_B).transpose()

				# print('EBT shape : ',E_BT.shape)
				# print('Fdash shape : ',FDash.shape)
				# print('DB shape : ',D_B.shape)
				# print("Vdashj shape: ",Vdash_j.shape)
				# print("Zdashj shape: ",Zdash_j.shape)
				# print("ZDash: ", Zdash_j)

				Del_J = func.matrixmul_reg(X_BT,D_B,E_BT,FDash,Vdash_j,Zdash_j) # the partial differentiation of the loss function output - dx1
				
				# print(Del_J)
				# print("Before Trunction")
				# DelJ2 = func.reconstruct(Del_J)

				# delta = DelJ2+Del_J
				# # print(": ")
				# print(delta)

				# for i in range(conf.d):
				# 	Del_J[i][0] = func.truncate(Del_J[i][0],conf.converttoint64)
			
				# DelJ2 = func.reconstruct(Del_J)

				# delta = DelJ2+Del_J
				# print("After truncation: ")
				# print(Del_J)
					
				# print("Alpha: ",alpha)
				# print(gradient.shape)
				# print("Trunction of Del")
				for i in range(conf.d):
					Del_J[i][0] = func.truncate(Del_J[i][0],conf.converttoint64)
					Del_J[i][0] = func.truncate(Del_J[i][0],func.floattoint64(conf.alpha_inv*(conf.batchsize)))
					

				weights = np.uint64((np.subtract(np.array(weights),Del_J)))
			
			if e == 0 or e==conf.epochs-1: 
				print("Loss: ", float(loss))

		
		# for i in range(conf.d):
			# weights[i][0] = func.int64tofloat(weights[i][0])
		
		print("\nWeights: ",weights)
		
		weights2 = func.reconstruct(weights.tolist())

		model = (np.add(np.array(weights2),np.array(weights)))
		
		return model