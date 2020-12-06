from functionalities import functionalities as func
from Config import Config as conf
import math
from charm.toolbox.integergroup import IntegerGroup,ZR
import hashlib

class mult_triplets:
	def KDF(k):
		m = k.to_bytes(2, 'big')
		m = hashlib.sha256(m).hexdigest()
		return m

	def trip_gen(U,V):
		G = IntegerGroup() # do this in config?
		G.paramgen(1024)
		g = G.randomGen()

		for j in range(conf.t): 
			A = np.array(U[j:j+conf.batchsize])
			if(flag != 0):
				A = A.reshape(conf.d,1)
				# print("A.shape: ", A.shape)

			B = np.array(V[:,j])
			B = B.reshape(V.shape[0],1)
			c_0 = np.matmul(A,B) # A0xB0 for S0 and A1xB1 for S1

			# A0 x B1
			for i in range(len(A)):
				if(conf.partyNum==0):
					r = np.array(np.random.random(size=(conf.NUM_BITS,)))
					f_r = np.array(np.uint64(A[i][0]*(math.pow(2,p))+r[p]) for p in range(conf.NUM_BITS))
				else: 
					b = []

				for j in range(B.shape[0]):
					if(partyNum==1):
						n = ("{0:b}".format(B[j][0])).rjust(64,'0') # bitdecompostion of B_j
						b = [int(p) for p in n] # convert the bit string to int list
						# receive u

						# (h_i0,h_i1)
						alpha = [G.random() for i in range(conf.NUM_BITS)]
						h = []
						
						for l in range(conf.NUM_BITS):
							h1 = G.random()
							if(b[l] == 0):
								h.append((g**alpha[l],h1))
							else:
								h.append((h1,g**alpha[l]))

						# send h to the other party
						# receive u,v
						k = []
						x = []
						for l in range(conf.NUM_BITS):
							k.append(u**alpha[l])
							kdf_k = KDF(k[l])
							x.append((v[l][b[l]])^(kdf_k))

						sum = 0
						for p in range(conf.NUM_BITS):
							sum = (sum+x[p])%(2**64)

					else:
						m = G.random() # random element from Z_R_q
						u = (g**m)

						# receive h
						# h = [(a,b),(c,d)]
						k = []
						v = []
						for l in range(conf.NUM_BITS):
							k.append((h[l][0]**m,h[l][1]**m))
							kdf_k0 = KDF(k[l][0])
							kdf_k1 = KDF(k[l][1])
							v.append((r[l]^kdf_k0,f_r[l]^kdf_k1))

						# send u and v to the other party

						sum = 0
						for p in range(conf.NUM_BITS):
							sum = (sum + ((-1)*r[l]))%(2**64)