import random
import numpy as np
from Config import Config as config

n = 5
d = 3
t = n
batchsize=1

def split_share(xy,u,v,v_dash,z,z_dash):
    xy_1 = np.random.randint(low = 100, size = (n, d+1))
    xy_2 = np.subtract(xy,xy_1)
    u_1 = np.random.randint(low = 100,size = (n,d))
    u_2 = np.subtract(u,u_1)
    v_1 = np.random.randint(low = 100,size = (d,t))
    v_2 = np.subtract(v,v_1)
    vdash_1 = np.random.randint(low = 100,size = (batchsize,t))
    vdash_2 = np.subtract(v_dash,vdash_1)
    z_1 = np.random.randint(low = 100,size = (1,t))
    z_2 = np.subtract(z,z_1)
    zdash_1 = np.random.randint(low = 100,size = (d,t))
    zdash_2 = np.subtract(z_dash,zdash_1)
    return xy_1,xy_2,u_1,u_2,v_1,v_2,vdash_1,vdash_2,z_1,z_2,zdash_1,zdash_2

def generatedata():
    
    z=[]
    z_dash=[]

    xy = np.random.randint(low = 100, size = (n, d+1)) 
    u = np.random.randint(low = 100,size = (n,d))
    
    v = np.random.randint(low = 100,size = (d,t))
    v_dash = np.random.randint(low = 100,size = (batchsize,t))

    z = np.zeros((1,t),dtype=int)
    z_dash= np.zeros((d,t),dtype=int)
   
    for i in range(len(u)):
        z[:,i]= (np.matmul(u[i],v[:,i])) #multiplying a row of u with a column of v
        u_row_tranpose = np.transpose(np.matrix(u[i]))
        #print(u_row_tranpose)
        z_dash[:,i]=np.matmul(u_row_tranpose,v_dash[:,i]) # print(np.transpose(np.matrix(u[0]))) 
    
    return xy,u,v,v_dash,z,z_dash
    

def saveData(serverNum,xy,u,v,v_dash,z,z_dash):

    if serverNum!= 0 and serverNum!= 1:
        datafile='data.txt'
        maskfile='mask.txt'
    else:
        datafile = 'data' + str(serverNum) + '.txt' 
        maskfile = 'mask' + str(serverNum) + '.txt'
    
    np.savetxt(datafile, xy, delimiter=' ',fmt='%d')
    
    with open(maskfile,'w+') as f:
        np.savetxt(f,u,delimiter=' ',fmt='%d')
        np.savetxt(f,v,delimiter=' ',fmt='%d')
        np.savetxt(f,v_dash,delimiter=' ',fmt='%d')
        np.savetxt(f,z,delimiter=' ',fmt='%d')
        np.savetxt(f,z_dash,delimiter=' ',fmt='%d')

def main():
    xy,u,v,v_dash,z,z_dash= generatedata()
    xy_1,xy_2,u_1,u_2,v_1,v_2,vdash_1,vdash_2,z_1,z_2,zdash_1,zdash_2 = split_share(xy,u,v,v_dash,z,z_dash)
    saveData(-1,xy,u,v,v_dash,z,z_dash)
    saveData(0,xy_1,u_1,v_1,vdash_1,z_1,zdash_1)
    saveData(1,xy_2,u_2,v_2,vdash_2,z_2,zdash_2)
    

if __name__ == '__main__':
    main()