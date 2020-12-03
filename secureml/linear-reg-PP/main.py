from Config import Config as conf
import sys
from functionalities import functionalities as func
from linearReg import linearReg as linearReg
from offline import offline as off
import numpy as np
from testmatmul import testmatmul as test

# command line argument - partyNum, integer input a, integer input b, mask value
# main
def main():
	conf.partyNum = int(sys.argv[1])
	#print(conf.partyNum)
	if conf.partyNum == 0:
		conf.PORT = 8004
		conf.advPORT = 8005
	else: 
		conf.PORT = 8005
		conf.advPORT = 8004

	# test.test()
	
	############# add float shares ################
	# a = float(sys.argv[2])
	# b = float(sys.argv[3])
	# u = float(sys.argv[4])
	# output = addshares(a,b,u)
	# print(output)

	############ multiply float shares ##############
	# a = float(sys.argv[2])
	# b = float(sys.argv[3])
	# u = float(sys.argv[4])
	# v = float(sys.argv[5])
	# z = float(sys.argv[6])
	# output_mul = multiplyshares(a,b,u,v,z)
	# print(output_mul)

	############ linear regression ############
	# filename_data = str(sys.argv[2])
	# filename_mask = str(sys.argv[3])

	if conf.partyNum==0:
		filename_data='data0.txt'
		filename_mask='mask0.txt'
	else:
		filename_data='data1.txt'
		filename_mask='mask1.txt'

	X,Y,U,V,Vdash,Z,Zdash = linearReg.readData(filename_data,filename_mask) #technically gen data should be just splitting data
	
	U = np.random.rand(conf.n, conf.d)
	V = np.random.rand(conf.d,conf.t)
	print("U: ",U)

	Vdash = np.random.rand(conf.batchsize,conf.t)
	print("Vdash: ",Vdash)
	U2 = np.array(func.reconstruct(U.tolist()))
	U2 = U2.reshape(conf.n,conf.d)
	u = np.add(U,U2)

	# V2 = np.array(func.reconstruct(V.tolist()))
	# V2 = V2.reshape(conf.d,conf.t)
	# v = np.add(V,V2)	

	Vdash2 = np.array(func.reconstruct(Vdash.tolist()))
	Vdash2 = Vdash2.reshape(conf.batchsize,conf.t)
	vdash = np.add(Vdash,Vdash2)	
	# print('U: ',u)
	# print('V:', v)
	# print('Vdash: ',vdash)

	# z = np.zeros((1,conf.t))
	zdash = np.zeros((conf.d,conf.t))
	# for i in range(len(u)):
	# 	z[:,i]= np.array((np.matmul(u[i],v[:,i])))
	# print("mult z: ", z)
	
	for i in range(len(u)):
		u_row_tranpose = np.transpose(np.matrix(u[i]))
		print(u_row_tranpose)
		zdash[:,i]=np.array(np.matmul(u_row_tranpose,vdash[:,i]))
	
	print("mult zdash: ", zdash)

	# flag = 0 # generate Z
	# Z=off.lhe(np.array(U),np.array(V),flag)
	flag = 1 # generate Zdash
	Zdash=off.lhe(np.array(U),np.array(Vdash),flag)

	# print('my Z: ',Z)
	# Z2 = np.array(func.reconstruct(Z.tolist()))
	# Z2 = Z2.reshape(1,conf.t)
	# Z_f = np.add(Z,Z2)

	# print("reconstructed Z: ", Z_f)

	Zdash2 = np.array(func.reconstruct(Zdash.tolist()))
	Zdash2 = Zdash2.reshape(conf.d,conf.t)
	Zdash_f = np.add(Zdash,Zdash2)

	print("reconstructed Zdash: ", Zdash_f)
	# print(Zdash)

	U = func.floattoint64(U)
	V = func.floattoint64(V)
	Vdash = func.floattoint64(Vdash)
	Z = func.floattoint64(Z)
	Zdash = func.floattoint64(Zdash)
	# model = linearReg.SGDLinear(X,Y,U,V,Vdash,Z,Zdash)

	# print(model)
	# print("\n[",end=" ")
	# for i in range(conf.d):
	# 	print(func.int64tofloat(model[i]),end=",")
	# print("]")
	# print('Before truncate Model: ',model)
	# model = func.truncate(model)
	# print('After truncate Model: ',model)


if __name__ == '__main__':
	main()