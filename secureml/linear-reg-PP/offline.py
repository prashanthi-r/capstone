from functionalities import functionalities as func
import numpy as np
import phe as paillier
from Config import Config as conf
import math

class offline:

    def encrypt_vector(public_key, x):
        return np.array([public_key.encrypt(i) for i in x])

    def decrypt_vector(private_key, x):
        return np.array([private_key.decrypt(i) for i in x])

    def lhe(U,V):
        print('Entered')
        C=[]
        pubkey,privkey = paillier.generate_paillier_keypair(n_length=1024)
        for j in range(conf.t): 
            A = np.array(U[j:j+conf.batchsize],dtype=np.uint64)
            B = np.array(V[:,j]).transpose()
            c_0 = np.uint64(np.matmul(A,B)) #A0B0 for S0 and A1B1 for S1
            encrypted_B = offline.encrypt_vector(pubkey, B) #S1 encrypts B1 for A0B1 and S0 encrypts B0 A1B0
            other_B = np.array(func.reconstruct(encrypted_B.tolist()))
            other_B = other_B.reshape(conf.d,1) #B is d*1 
            c_1=1
            print(other_B[0]*other_B[0])
            for i in range(conf.d): #will only work for batchsize = 1, change for any batchsize once this works
                c_1 = np.uint64(c_1 * np.uint64(other_B[i][0]**A[i])) #not sure about mod64, not mentioned in paper
            
            random_num = np.array((np.random.rand(conf.batchsize)))
            encrypted_random = offline.encrypt_vector(pubkey,random_num)
            #c_1 = np.matmul(c_1,encrypted_random)
            c_1 = c_1*encrypted_random
            recv = func.reconstruct(c_1.tolist())
            recv=recv.reshape(conf.batchsize,1)
            recv=offline.decrypt_vector(privkey,recv)
            random_num = np.uint64(np.multiply(-1,random_num)) #since -r mod 2^l
            term = np.add(c_0,recv)
            term = np.add(term,random_num)
            C.append(term) #A0B0 + A0B1 + A1BO for S0, A1B1+A0B1+A1B0 for S1
            print(C)
        return C

            
            






