import random
import numpy as np
from Config import Config as conf
from functionalities import functionalities as func
#from sklearn.datasets import load_boston

n = 6
d = 2
t = n
batchsize=1

def split_shares(x,p,q):
    # print(x)
    # x = func.floattoint64(x)
    x_1 = np.uint64(np.random.uniform(0, (2**conf.l), (p,q)))
    # print("x shape ",x.shape)
    x_2 = np.array(np.subtract(x,x_1), dtype = np.uint64)
    # print("x_1 shape ", x_1.shape)
    # x_2 = (np.random.uniform(0, (2**conf.l), (p,q)))
    # x_1 = np.uint64(np.add(x,x_2))
    # x_2 = np.uint64(-1*x_2)
    # print(x_2.shape)
    return np.array(x_1, dtype = np.uint64),x_2

def generatedata():
    
    z=[]
    z_dash=[]

    # xy = np.random.randint(low = 100, size = (n, d+1)) 
    u = np.array(np.random.randint(low = 2**(conf.l-1),size = (n,d+1)), dtype = np.uint64)
    
    v = np.array(np.random.randint(low = 2**(conf.l-1),size = (d+1,t)), dtype = np.uint64)
    v_dash = np.array(np.random.randint(low = 2**(conf.l-1),size = (batchsize,t)), dtype = np.uint64)

    z = np.zeros((1,t),dtype=int)
    z_dash= np.zeros((d+1,t),dtype=int)
   
    for i in range(len(u)):
        z[:,i]= np.array((np.matmul(u[i],v[:,i])), dtype = np.uint64) #multiplying a row of u with a column of v
        u_row_tranpose = np.transpose(np.matrix(u[i]))
        #print(u_row_tranpose)
        z_dash[:,i]=np.array(np.matmul(u_row_tranpose,v_dash[:,i]), dtype = np.uint64) # print(np.transpose(np.matrix(u[0]))) 
    
    return u,v,v_dash,z,z_dash #xy
    

def saveData(serverNum,u,v,v_dash=np.array(None),z=np.array(None),z_dash=np.array(None)): #xy

    if serverNum!= 0 and serverNum!= 1:
        datafile='data.txt'
        maskfile='mask.txt'
    else:
        datafile = 'data' + str(serverNum) + '.txt' 
        maskfile = 'mask' + str(serverNum) + '.txt'

    if(v_dash.any()==None or z.any()==None or z_dash.any()==None):
        with open(datafile,'w+') as df:
            np.savetxt(df, u, delimiter=' ',fmt='%d') #x
            np.savetxt(df, v, delimiter='\n',fmt='%d') #y
    else:
        with open(maskfile,'w+') as f:
            np.savetxt(f,u,delimiter=' ',fmt='%d')
            np.savetxt(f,v,delimiter=' ',fmt='%d')
            np.savetxt(f,v_dash,delimiter=' ',fmt='%d')
            np.savetxt(f,z,delimiter=' ',fmt='%d')
            np.savetxt(f,z_dash,delimiter=' ',fmt='%d')

def check_shares(x1,x2,X):
    sx = (np.add(x1,x2))
    print("Sum:",sx)
    print("Data: ",X)


def main():
    # X, Y = load_boston(return_X_y=True)
    # X = np.uint64(conf.converttoint64*np.array(X[:8])).tolist()
    # Y = np.uint64(conf.converttoint64*np.array(Y[:8]))
    X = [[4,1],[2,8],[1,0],[3,2],[1,4],[6,7]]
    X = np.array(X, dtype = np.uint64)
    X = func.floattoint64(X)
    Y = [2,-14,1,-1,-7,-8]
    Y = np.array(Y, dtype = np.uint64)
    Y = func.floattoint64(Y)
    Y = Y.reshape(len(Y),1)
    
    X_1,X_2 = split_shares(X,len(Y),d)
    # check_shares(X_1,X_2,X)
    # print("X_1's shape: ",X_1.shape)
    # print("X_2's shape: ",X_2.shape)
    # print("Y:",Y)
    Y_1,Y_2 = split_shares(Y,len(Y),1)
    print("Y_1: ",Y_1)
    print("Y_2: ",Y_2)
    check_shares(Y_1,Y_2,Y)
    # print("Y_1's shape: ",Y_1.shape)
    # print("Y_2's shape: ",Y_2.shape)
    u,v,v_dash,z,z_dash = generatedata() #xy,
    u_1,u_2 = split_shares(u,n,d+1)
    v_1,v_2 = split_shares(v,d+1,t)
    vdash_1,vdash_2 = split_shares(v_dash,batchsize,t)
    z_1,z_2 = split_shares(z,1,t)
    zdash_1,zdash_2 = split_shares(z_dash,d+1,t) #xy_1,xy_2,
    saveData(-1,u,v,v_dash,z,z_dash) #xy
    saveData(0,u_1,v_1,vdash_1,z_1,zdash_1) #xy_1
    saveData(1,u_2,v_2,vdash_2,z_2,zdash_2) #xy_2
    # saveData(-1,X,Y) 
    saveData(0,X_1,Y_1)
    saveData(1,X_2,Y_2) 

if __name__ == '__main__':
    main()