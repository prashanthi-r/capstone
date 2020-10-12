import random
import numpy as np
from Config import Config as conf
from sklearn.datasets import load_boston

n = 8
d = 13
t = n
batchsize=1

def split_shares(x,p,q):
    # print(x.shape)
    x_1 = np.random.uniform(0, (2**conf.l)-1, (p,q))
    # print(x_1.shape)
    x_2 = np.subtract(x,x_1)
    # print(x_2.shape)
    return x_1,x_2

# def split_share(u,v,v_dash,z,z_dash): #xy
#     # xy_1 = np.random.randint(low = 100, size = (n, d+1))
#     # xy_2 = np.subtract(xy,xy_1)
#     u_1 = np.random.randint(low = (2**conf.l),size = (n,d))
#     u_2 = np.uint64(np.subtract(u,u_1))
#     v_1 = np.random.randint(low = (2**conf.l),size = (d,t))
#     v_2 = np.uint64(np.subtract(v,v_1))
#     vdash_1 = np.random.randint(low = (2**conf.l),size = (batchsize,t))
#     vdash_2 = np.uint64(np.subtract(v_dash,vdash_1))
#     z_1 = np.random.randint(low = (2**conf.l),size = (1,t))
#     z_2 = np.uint64(np.subtract(z,z_1))
#     zdash_1 = np.random.randint(low = (2**conf.l),size = (d,t))
#     zdash_2 = np.uint64(np.subtract(z_dash,zdash_1))
#     return u_1,u_2,v_1,v_2,vdash_1,vdash_2,z_1,z_2,zdash_1,zdash_2 #xy_1,xy_2,

def generatedata():
    
    z=[]
    z_dash=[]

    # xy = np.random.randint(low = 100, size = (n, d+1)) 
    u = np.random.randint(low = 2**(conf.l-1),size = (n,d))
    
    v = np.random.randint(low = 2**(conf.l-1),size = (d,t))
    v_dash = np.random.randint(low = 2**(conf.l-1),size = (batchsize,t))

    z = np.zeros((1,t),dtype=int)
    z_dash= np.zeros((d,t),dtype=int)
   
    for i in range(len(u)):
        z[:,i]= np.uint64((np.matmul(u[i],v[:,i]))) #multiplying a row of u with a column of v
        u_row_tranpose = np.transpose(np.matrix(u[i]))
        #print(u_row_tranpose)
        z_dash[:,i]=np.uint64(np.matmul(u_row_tranpose,v_dash[:,i])) # print(np.transpose(np.matrix(u[0]))) 
    
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


def main():
    X, Y = load_boston(return_X_y=True)
    print(X[:5])
    print(Y[:5])
    Y = Y.reshape(len(Y),1)
    X_1,X_2 = split_shares(X,len(Y),d)
    # print("X_1's shape: ",X_1.shape)
    # print("X_2's shape: ",X_2.shape)
    Y_1,Y_2 = split_shares(Y,len(Y),1)
    # print("Y_1's shape: ",Y_1.shape)
    # print("Y_2's shape: ",Y_2.shape)
    u,v,v_dash,z,z_dash = generatedata() #xy,
    u_1,u_2 = split_shares(u,n,d)
    v_1,v_2 = split_shares(v,d,t)
    vdash_1,vdash_2 = split_shares(v_dash,batchsize,t)
    z_1,z_2 = split_shares(z,1,t)
    zdash_1,zdash_2 = split_shares(z_dash,d,t) #xy_1,xy_2,
    saveData(-1,u,v,v_dash,z,z_dash) #xy
    saveData(0,u_1,v_1,vdash_1,z_1,zdash_1) #xy_1
    saveData(1,u_2,v_2,vdash_2,z_2,zdash_2) #xy_2
    saveData(0,X_1,Y_1)
    saveData(1,X_2,Y_2) 

if __name__ == '__main__':
    main()