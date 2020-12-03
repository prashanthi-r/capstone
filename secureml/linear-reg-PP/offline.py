from functionalities import functionalities as func
import numpy as np
import phe as paillier
from Config import Config as conf
import math

class offline:

	def encrypt_vector(public_key, x):
		enc_x = []
		for i in range(len(x)):
			enc_x.append(public_key.encrypt(x[i][0]))
		return np.array(enc_x)

	def decrypt_vector(private_key, x):
		dec_x = []
		for i in range(len(x)):
			dec_x.append(private_key.decrypt(x[i][0]))
		return np.array(dec_x)

	def lhe(U,V, flag=0):
		print('Entered')
		C=[]
		pubkey,privkey = paillier.generate_paillier_keypair()
		pubkeyOther = func.reconstruct([pubkey])
		pubkeyOther=pubkeyOther[0]
		print('others pubkey ',pubkeyOther)
		for j in range(conf.t): 
			print('In loop')
			if(flag == 0):
				A = np.array(U[j:j+conf.batchsize])
			else:
				A = np.array(U[:,j])
				A= A.reshape(conf.d,1)

			B = np.array(V[:,j])
			B= B.reshape(V.shape[0],1)
			c_0 = np.matmul(A,B) #A0B0 for S0 and A1B1 for S1 // np.uint64()
			encrypted_B = offline.encrypt_vector(pubkey, B) #S1 encrypts B1 for A0B1 and S0 encrypts B0 A1B0
			other_B = np.array(func.reconstruct(encrypted_B.tolist()))
			other_B = other_B.reshape(V.shape[0],1) #B is d*1 
			c_1=0
			for i in range(V.shape[0]): 
				#will only work for batchsize = 1, change for any batchsize once this works
				c_1 = c_1 + other_B[i][0]*A[0][i] #not sure about mod64, not mentioned in paper

			c_1 = np.array(c_1)
			c_1= c_1.reshape(conf.batchsize,1)
			random_num = np.array(np.random.random())
			# random_num = func.floattoint64(random_num)
			random_num= random_num.reshape(conf.batchsize,1)
			encrypted_random = offline.encrypt_vector(pubkeyOther,random_num)
			print(encrypted_random)
			#c_1 = np.matmul(c_1,encrypted_random)
			c_1 = np.add(c_1,encrypted_random)
			recv = np.array(func.reconstruct(c_1.tolist()))
			recv=recv.reshape(conf.batchsize,1)
			recv=offline.decrypt_vector(privkey,recv)
			random_num = np.multiply(-1,random_num) #since -r mod 2^l
			term = np.add(c_0,recv)
			term = np.add(term,random_num)
			C.append(term) #A0B0 + A0B1 + A1BO for S0, A1B1+A0B1+A1B0 for S1
		
		C = np.array(C)
		C = C.reshape(1,conf.t)
		print('Final term: ',C)
		return C