from functionalities import functionalities as func
from Config import Config as conf
import math
from charm.toolbox.integergroup import IntegerGroup
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA512
from Crypto.Random import get_random_bytes
import numpy as np
import pickle

class ot_gen_offline:
	def byte_xor(ba1, ba2):
		return bytes([_a ^ _b for _a, _b in zip(ba1, ba2)])

	def KDF(k):
		password = b'' + bytes(str(k), 'utf-8') 
		salt = get_random_bytes(16)
		keys = PBKDF2(password, salt, 64, count=100)
		return keys;

	def trip_gen(U,V,flag=0):
		G = IntegerGroup() # do this in config?
		G.paramgen(1024)
		g = G.randomGen()


		# files for communication
		h_filename = str(conf.partyNum)+"_"+"h.txt"
		v_filename = str(conf.partyNum)+"_"+"v.txt"

		for j in range(conf.t):
			A = np.array(U[j:j+conf.batchsize])
			if(flag != 0):
				A = A.reshape(conf.d,1)
				# print("A.shape: ", A.shape)

			B = np.array(V[:,j])
			B = B.reshape(V.shape[0],1)
			c_0 = np.matmul(A,B) # A0xB0 for S0 and A1xB1 for S1
			c_1 = []
			c_2 = []

			# A0 x B1 and A1 x B0 // works only for batchsize = 1
			for i in range(A.shape[0]):
				for i1 in range(A.shape[1]):
					r = np.uint64(np.random.uniform(0, (2**conf.l), (conf.l,))).tolist() # in 2^l?
					# print(r)
					f_r = [np.uint64(A[i][i1]*(math.pow(2,p))+r[p]) for p in range(conf.l)]
					b = []
					for j in range(B.shape[0]):
						n = ("{0:b}".format(B[j][0])).rjust(64,'0') # bitdecompostion of B_ij
						b = [int(p) for p in n] # convert the bit string to int list

						# (h_i0,h_i1)
						alpha = [G.random(G.q) for i in range(conf.l)]
						h = []
						for l in range(conf.l):
							beta = G.random(G.q)
							h1 = (g**beta)%(G.p)
							if(b[l] == 0):
								h.append((g**alpha[l],h1))
							else:
								h.append((h1,g**alpha[l]))

						# print(h[0])
						# send h to the other party
						func.send_file(h,h_filename)

						# process received h
						
						
						m = G.random(G.q) # random element from Z_R_q
						u = (g**m)
						
						# send and receive u
						other_u = func.send_file(u)

						k = []
						v = []
						for l in range(conf.l):
							k.append((other_h[l][0]**m,other_h[l][1]**m))
							kdf_k0 = ot_gen_offline.KDF(k[l][0])
							kdf_k1 = ot_gen_offline.KDF(k[l][1])
							v.append((r[l]^kdf_k0,f_r[l]^kdf_k1))
						
						# send and receive v
						other_v = func.send_file(v)

						fin_k = []
						x = []
						for l in range(conf.l):
							fin_k.append(other_u**alpha[l])
							kdf_k = KDF(fin_k[l])
							x.append((other_v[l][b[l]])^(kdf_k))

					sum1 = 0
					for p in range(conf.l):
						sum1 = (sum1+x[p])%(2**64)

					sum2 = 0
					for p in range(conf.l):
						sum2 = (sum2 + ((-1)*r[l]))%(2**64)

				c_1.append(sum1)
				c_2.append(sum2)

		c_1 = np.array(c_1).reshape(c_0.shape[0],c_0.shape[1])
		c_2 = np.array(c_2).reshape(c_0.shape[0],c_0.shape[1])

		C = np.array(np.add(c_0,c_1))
		C = np.add(C,c_2)

		return C