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
				if(i<=8):
					row= [float(i) for i in row]
					X.append(row)
				if(i>506 and i<=506+8):
					Y.append(float(row[0].rstrip()))
			f.close()
		# print("i: ",i)

		conf.n = len(Y)
		conf.d = len(X[0])
		conf.t = conf.n
		print("n: ",conf.n)
		print("d: ",conf.d)
		print("t: ",conf.t)


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

		X = func.floattoint64(np.array(X))
		Y = func.floattoint64(np.array(Y))

		U = np.array(U)
		V = np.array(V)
		VDash = np.array(VDash)
		Z = np.array(Z)
		ZDash = np.array(ZDash)

		E1 = np.subtract(np.array(X),U)
		E2 = func.reconstruct(E1.tolist())
		E = np.uint64(np.add(E1,np.array(E2)))
		# randomly initialise weights vector
		weights = np.array((np.random.random(size = (conf.d,1))))
		# print('Weights: ',weights)
		# weights2 = func.reconstruct(weights)

		# wts = weights+weights2
		# print("Initial weights: ")
		# print(wts)


		weights = func.floattoint64(weights)
		print(weights)

		for j in range(conf.t): 
			X_B = np.array(X[j:j+conf.batchsize])
			Y_B= np.array([Y[j:j+conf.batchsize]]).transpose()
			E_B = np.array(E[j:j+conf.batchsize])
			V_j = np.array([V[:,j]]).transpose()	# d*1
			Z_j = np.array([Z[:,j]]).transpose()  	#|B| * 1
			Vdash_j = np.array([VDash[:,j]]).transpose()
			Zdash_j = np.array([ZDash[:,j]]).transpose()

			F1 = np.uint64(np.subtract(np.array(weights),V_j))
			F2 = func.reconstruct(F1.tolist())
			F = np.uint64(np.add(np.array(F1),np.array(F2))) #d*1 as its weights-V_j (both of dim d*1)

			# print("Vj shape: ",V_j.shape)
			# print("Zj shape: ",Z_j.shape)

			YB_dash = func.matrixmul_reg(X_B,weights,E_B,F,V_j,Z_j) #|B|*1
			#print('YB Dash shape: ',YB_dash.shape)
			D_B = np.uint64(np.subtract(YB_dash,Y_B))

			Fdash_1 = np.uint64(np.subtract(D_B,Vdash_j))
			Fdash_2 = func.reconstruct(Fdash_1)
			FDash = np.uint64(np.add(np.array(Fdash_1),np.array(Fdash_2)))

			X_BT = np.array(X_B).transpose() 
			E_BT = np.array(E_B).transpose()

			# print('EBT shape : ',E_BT.shape)
			# print('Fdash shape : ',FDash.shape)
			# print('DB shape : ',D_B.shape)
			# print("Vdashj shape: ",Vdash_j.shape)
			# print("Zdashj shape: ",Zdash_j.shape)
			# print("ZDash: ", Zdash_j)

			Del_J = func.matrixmul_reg(X_BT,D_B,E_BT,FDash,Vdash_j,Zdash_j).tolist() # the partial differentiation of the loss function output - dx1
			
			print("Before Trunction")
			DelJ2 = func.reconstruct(Del_J)

			delta = DelJ2+Del_J
			# print(": ")
			print(delta)

			for i in range(conf.d):
				Del_J[i][0] = func.truncate(Del_J[i][0])
		
			DelJ2 = func.reconstruct(Del_J)

			delta = DelJ2+Del_J
			print("After truncation: ")
			print(delta)
				
			alpha = func.floattoint64(conf.alpha*(1/conf.batchsize))
			print("Alpha: ",alpha)
			gradient = np.uint64(alpha*np.array(Del_J))
			weights = np.uint64((np.subtract(np.array(weights),gradient))).tolist()

		print("\nWeights: ",weights)

		weights2 = func.reconstruct(weights)

		model = np.uint64(np.add(np.array(weights2),np.array(weights)))
		
		return model