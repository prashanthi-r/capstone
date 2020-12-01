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
					r = []
					r.append(func.floattoint64(0.5))
					for j in row:
						r.append(np.uint64(j))
					X.append(r)
				else:
					Y.append(np.uint64(row[0].rstrip()))
			f.close()
		# print("X: ", X)
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
		X = (np.array(X, dtype = np.uint64))
		Y = (np.array(Y, dtype = np.uint64))

		# print("Y: ", Y)
		# Y2 = func.reconstruct(Y.tolist())
		# Y2 = np.array(Y2,dtype = np.uint64)
		# Y_f = np.add(Y, np.array(Y2))
		# print("Reconstructed Y: ")

		# for i in Y_f:
			# print(func.int64tofloat(i))

		U = np.array(U, dtype = np.uint64)
		V = np.array(V, dtype = np.uint64)
		VDash = np.array(VDash, dtype = np.uint64)
		Z = np.array(Z, dtype = np.uint64)
		ZDash = np.array(ZDash, dtype = np.uint64)

		E1 = np.uint64(np.subtract(X,U))
		E2 = np.uint64(func.reconstruct(E1.tolist()))
		E = np.uint64(np.add(E1,np.array(E2, dtype = np.uint64)))
		# randomly initialise weights vector
		weights = np.array((np.random.rand(conf.d)))
		weights = weights.reshape(conf.d,1)
		# print('Weights: ',weights)
		weights2 = func.reconstruct(weights)

		wts = weights+weights2
		# print("Initial weights: ")
		# print(wts)


		weights = np.array(func.floattoint64(weights), dtype = np.uint64)
		print(weights)

		# print(Y)
		for e in range(conf.epochs):
			# print("e: ", e)
			loss = 0.0

			for j in range(conf.t): 
				X_B = np.array(X[j:j+conf.batchsize], dtype = np.uint64)
				Y_B = np.array([Y[j:j+conf.batchsize]], dtype = np.uint64).transpose()
				
				# print("X_B: ", X_B)
				xb2 = func.reconstruct(X_B.tolist())
				xb2 = np.array(xb2,dtype = np.uint64)
				print("xb2: ",xb2)
				xb = np.add(X_B, np.array(xb2))
				print("xb: ", xb)
				# print("x after reconstruction:", func.int64tofloat(xb[0][0]))
				for i in xb[0]:
					print(func.int64tofloat(i)) 

				E_B = np.array(E[j:j+conf.batchsize], dtype = np.uint64)
				V_j = np.array([V[:,j]], dtype = np.uint64).transpose()	# d*1
				Z_j = np.array([Z[:,j]], dtype = np.uint64).transpose()  	#|B| * 1
				Vdash_j = np.array([VDash[:,j]], dtype = np.uint64).transpose()
				Zdash_j = np.array([ZDash[:,j]], dtype = np.uint64).transpose()

				F1 = (np.subtract(weights,V_j))
				F2 = func.reconstruct(F1.tolist())
				F = (np.add(F1,np.array(F2, dtype = np.uint64))) #d*1 as its weights-V_j (both of dim d*1)
				# print("Vj shape: ",V_j.shape)
				# print("Zj shape: ",Z_j.shape)

				YB_dash = func.matrixmul_reg(X_B,weights,E_B,F,V_j,Z_j) #|B|*1
				# print('YB Dash shape: ',YB_dash.shape)					

				D_B = (np.subtract(YB_dash,Y_B))
				# print("Y_B: ",Y_B)
				# print("Y_B shape: ", Y_B.shape)
				# print("YB_dash: ",YB_dash)
				# print("D_B: ",D_B)

				YB_dash[0][0] = func.truncate(YB_dash[0][0],conf.converttoint64)
				# computing loss
				yb2 = func.reconstruct(Y_B.tolist())
				# print("yb2: ",yb2)
				# print("yb2 shape: ", np.array(yb2, dtype = np.uint64).shape)
				y = (np.add(Y_B,np.array(yb2, dtype = np.uint64)))
				y = func.int64tofloat(y[0][0])
				# print("After int to float, y:", y)
				
				#################################################### Computing loss ###################################################################
				
				# ybdash = (np.uint64(YB_dash))
				# print(ybdash)
				ybdash2 = func.reconstruct(YB_dash.tolist())
				# print("ybdash2: ",ybdash2)
				y_hat = (np.add(YB_dash,np.array(ybdash2, dtype = np.uint64)))
				# print("y_hat: ", np.array(y_hat, dtype = np.uint64))
				# y_hat = y_hat[0][0]
				# y_hat = (np.add(YB_dash,np.array(ybdash2, dtype = np.uint64)))
				
				y_hat = (func.int64tofloat(y_hat[0][0]))
				print("y_hat: ", y_hat)
				
				dif = (y_hat - y)
				# print(dif)
				loss = loss+(dif*dif)
				print("Loss: ", loss)

				#######################################################################################################################################

				Fdash_1 = (np.subtract(D_B,Vdash_j))
				Fdash_2 = func.reconstruct(Fdash_1)
				FDash = (np.add(Fdash_1,np.array(Fdash_2, dtype = np.uint64)))

				X_BT = np.array(X_B, dtype = np.uint64).transpose() 
				E_BT = np.array(E_B, dtype = np.uint64).transpose()

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
					

				weights = ((np.subtract(np.array(weights, dtype = np.uint64),Del_J)))
				
				# for i in range(conf.d):
				# 	print(func.int64tofloat(weights[i][0]))
			
			if e == 0 or e==conf.epochs-1: 
				print("Loss: ", float(loss))

		
		# for i in range(conf.d):
			# weights[i][0] = func.int64tofloat(weights[i][0])
		
		################# Reconstructed final weights #############################################
		
		weights2 = func.reconstruct(weights.tolist())
		model = (np.add(np.array(weights2, dtype = np.uint64),np.array(weights, dtype = np.uint64)))

		###########################################################################################
		
		return model