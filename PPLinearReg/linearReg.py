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
		with open(filename_data,'r') as f:
			for line in f:
				row=line.split()
				Y.append(float(row[-1].rstrip()))#last element
				row= [float(i) for i in row]
				X.append(row[:-1]) #all elements except the last element
			f.close()


		conf.n = len(Y)
		conf.d = len(X[0])
		conf.t = conf.n

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
		weights = func.floattoint64(np.array([[random.random() for i in range(conf.d)]]))

		for j in range(conf.t): 
			print(j)
			X_B = X[j:j+conf.batchsize]
			Y_B = Y[j:j+conf.batchsize]
			E_B = E[j:j+conf.batchsize]
			V_j = V[:,j]
			Z_j = Z[:,j]
			Vdash_j = VDash[:,j]
			Zdash_j = ZDash[:,j]

			F1 = np.uint64(np.subtract(np.array(weights),np.array(V_j)))
			F2 = func.reconstruct(F1.tolist())
			F = np.uint64(np.add(np.array(F1),np.array(F2)))

			YB_dash = func.matrixmul_reg(X_B,weights,E_B,F,V_j,Z_j)

			D_B = np.uint64(np.add(YB_dash,np.array(Y_B)))

			Fdash_1 = np.uint64(np.subtract(D_B,Vdash_j))
			Fdash_2 = func.reconstruct(Fdash_1)
			FDash = np.uint64(np.add(np.array(Fdash_1),np.array(Fdash_2))).tolist()

			X_B = np.array(X_B).transpose()
			E_B = np.array(E_B).transpose()

			Del_J = func.matrixmul_reg(X_B.tolist(),D_B,E_B.tolist(),FDash,Vdash_j,Zdash_j).tolist() # the partial differentiation of the loss function
			
			print(Del_J)

			for i in range(conf.d):
				Del_J[i] = math.floor(Del_J[i])

			weights = np.uint64(np.subtract(np.array(weights),(alpha*(1/conf.batchsize)*np.array(Del_J)))).tolist()

		weights2 = func.reconstruct(weights)

		model = np.uint64(np.add(np.array(weights2),np.array(weights)))
		
		return model