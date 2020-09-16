import random
import numpy as np

def generatedata():
    n = 5
    d = 3
    t = n
    batchsize=1
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
    
    
    np.savetxt('data.txt', xy, delimiter=' ',fmt='%d')
    
    with open('mask.txt','a+') as f:
        np.savetxt(f,u,delimiter=' ',fmt='%d')
        np.savetxt(f,v,delimiter=' ',fmt='%d')
        np.savetxt(f,v_dash,delimiter=' ',fmt='%d')
        np.savetxt(f,z,delimiter=' ',fmt='%d')
        np.savetxt(f,z_dash,delimiter=' ',fmt='%d')

def main():
    generatedata()

if __name__ == '__main__':
    main()