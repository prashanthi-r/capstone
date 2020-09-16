from Config import Config as config
from functionalites import functionalites as func

n = config.n 
d = config.d
t = config.t
b = config.batchsize

def readData(filename_data,filename_mask):
    # X n*d matrix
    # Y n*1 matrix
    # w d*1 matrix
    # U n*d matrix
    # V d*1 matrix
    # V'1*t matrix mask the difference Y - Y^
    # Z n*1 matrix
    # Z'd*t matrix (U_b transpose * V'[i]) 
    mask=[]
    X=[]    
    Y=[]
    U=[]
    V=[]
    V_dash=[] 
    Z=[]
    Z_dash=[]
    with open(filename_data,'r+') as f:
        for line in f:
            row=line.split()
            Y.append(float(row[-1].rstrip())) #last element
            X.append(float(row[:-1].rstrip()) #all elements except the last element

    config.n = len(Y)
    config.d = len(X[0])
    config.t = config.n

    with open(filename_mask,'r+') as f:
        for line in f:
            mask.append(float(line.split().rstrip()))

    U = mask[:n]
    V = mask[n:n+1]
    Vdash = mask[n+1: n+1+b]
    Z = mask[n+1+b:n+b+2] 
    Zdash = mask[n+b+2:n+b+3]