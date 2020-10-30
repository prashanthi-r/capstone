from functionalities import functionalities as func
import numpy as np
import phe as paillier
from Config import Config as conf
import math

class offline:

    def lhe(U,V):
        C=[]
        keypair = paillier.generate_paillier_keypair(n_length=1024)
        for j in range(conf.t): 
            A = np.array(U[j:j+conf.batchsize])
            B = np.array([V[:,j]]).transpose()
            c_0 = np.uint64(np.matmul(A,B)) #A0B0 for S0 and A1B1 for S1
            encrypted_B = encrypt_vector(keypair.pubkey, B) #S1 encrypts B1 for A0B1 and S0 encrypts B0 A1B0
            other_B = np.array(func.reconstruct(encrypted_B.tolist()))
            other_B = other_B.reshape(conf.d,1) #B is d*1 
            c_1=1
            for i in range(conf.d): #will only work for batchsize = 1, change for any batchsize once this works
                c_1 = np.uint64(c_1 * np.uint64(math.pow(other_B[i][0],A[i]))) #not sure about mod64, not mentioned in paper
            random_num = np.array((np.random.rand(conf.batchsize)))
            encrypted_random = encrypt_vector(keypair.pubkey,random_num)
            c_1 = np.matmul(c_1,encrypted_random)
            recv = func.reconstruct(c_1.tolist())
            recv=recv.reshape(conf.batchsize,1)
            recv=decrypt_vector(keypair.privkey,recv)
            random_num = np.uint64(np.multiply(-1,random_num)) #since -r mod 2^l
            term = np.add(c_0,recv)
            term = np.add(term,random_num)
            C.append(term) #A0B0 + A0B1 + A1BO for S0, A1B1+A0B1+A1B0 for S1

                 
            






