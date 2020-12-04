from functionalities import functionalities as func
from Config import Config as conf
import math
from charm.toolbox.integergroup import IntegerGroup


class mult_triplets:
	def ot():

	def trip_gen(U,V):
		group1 = IntegerGroup() # do this in config?
		g = group1.randomGen()

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
				for j in range(B.shape[0]):
					b = []
					for k in range(conf.NUM_BITS):
						if(conf.partyNum == 0):
							m = group1.random(ZR) # random element from Z_R_q
							u = (g**m)

						else:
							x = ("{0:b}".format(B[j][0])).rjust(64,'0') # bitdecompostion of B_j
							b = [int(p) for p in x] # convert the bit string to int list
							