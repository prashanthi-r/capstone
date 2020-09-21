from Config import Config as config
import sys
import itertools
import numpy as np
from functionalities import functionalities

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
                X.append((row[:-1])) #all elements except the last element
            f.close()

        #X = list(map(int, X))

        config.n = len(Y)
        config.d = len(X[0])
        config.t = config.n

        with open(filename_mask,'r') as f:
            for line in f:
                mask.append(line.split())
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

    def regression(X,Y,U,V,vDash,Z,zDash):
        E = np.subtract(X,U)
        send




    # def main():
    #     filename_data= sys.argv[1]
    #     filename_mask = sys.argv[2]
    #     readData(filename_data,filename_mask)
        