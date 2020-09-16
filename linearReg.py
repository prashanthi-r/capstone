from Config import Config as config
import sys
import itertools


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
            Y.append(float(row[-1].rstrip())) #last element
            X.append((row[:-1])) #all elements except the last element
        f.close()

    config.n = len(Y)
    config.d = len(X[0])
    config.t = config.n

    with open(filename_mask,'r') as f:
        for line in f:
            mask.append(line.split())

    n = config.n 
    d = config.d
    t = config.t
    b = config.batchsize

    U = mask[:n]
    print(U)
    V = mask[n+1:n+1+d]
    print(V)
    # Vdash = mask[n+1: n+1+b]
    # Z = mask[n+1+b:n+b+2] 
    # Zdash = mask[n+b+2:n+b+3]
    print('\n')
    print(mask)
   
    # print(V)
    # print('\n')
    # print(Vdash)
    # print('\n')
    # print(Z)
    # print('\n')
    # print(Zdash)



filename_data= sys.argv[1]
filename_mask = sys.argv[2]
readData(filename_data,filename_mask)