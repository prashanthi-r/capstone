from Config import Config as config
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
                Y.append(int(row[-1].rstrip())) #last element
                row= [int(i, base=10) for i in row]
                X.append(row[:-1]) #all elements except the last element
            f.close()


        config.n = len(Y)
        config.d = len(X[0])
        config.t = config.n

        with open(filename_mask,'r') as f:
            for line in f:
                row=line.split()
                row=[int(i, base=10) for i in row]
                mask.append(row)
            f.close()


        n = config.n 
        d = config.d
        t = config.t
        b = config.batchsize

        U = mask[:n]
        V = mask[n:n+d]
        Vdash = mask[n+d: n+d+1]
        Z = mask[n+d+1:n+d+2]
        Zdash=mask[n+d+2:]
        
        return X,Y,U,V,Vdash,Z,Zdash

    def LinReg(X,Y,U,V,VDash,Z,ZDash):
        
        #ZDash = (np.matmul(np.array(U).transpose,VDash).tolist())
        # print(np.array(X))
        # print(np.array(U))
        E1 = np.subtract(np.array(X),np.array(U))
        E2 = func.reconstruct(E1.tolist())
        E = (np.add(np.array(E1),np.array(E2)).tolist())

        # randomly initialise weights vector
        weights = [random.random() for i in range(config.d)]
        V = np.array(V)
        VDash = np.array(VDash)

        for j in range(config.t): 

            X_B = X[j:j+config.batchsize]
            Y_B = Y[j:j+config.batchsize]
            E_B = E[j:j+config.batchsize]
            V_j = V[:,j]
            Z_j = Z[j]
            Vdash_j = VDash[:,j]
            Zdash_j = ZDash[j]


            F1 = np.subtract(np.array(weights),np.array(V_j))
            F2 = func.reconstruct(F1.tolist())
            F = (np.add(np.array(F1),np.array(F2)).tolist())


            YB_dash = func.matrixmul_reg(X_B,weights,E_B,F,V_j,Z_j)

            D_B = np.add(YB_dash,np.array(Y_B))

            Fdash_1 = np.subtract(D_B,Vdash_j)
            Fdash_2 = func.reconstruct(Fdash_1)
            FDash = (np.add(np.array(Fdash_1),np.array(Fdash_2)).tolist())

            X_B = np.array(X_B).transpose()
            E_B = np.array(E_B).transpose()

            Del_J = func.matrixmul_reg(X_B.tolist(),D_B,E_B.tolist(),FDash,Vdash_j,Zdash_j).tolist() # the partial differentiation of the loss function
            for i in range(d):
                Del_J[i] = math.floor(Del_J[i])


            weights = np.subtract(np.array(weights),(alpha*(1/config.batchsize)*np.array(Del_J))).tolist()

        weights2 = func.reconstruct(weights)

        model = np.add(np.array(weights2),np.array(weights)).tolist()
        
        return model