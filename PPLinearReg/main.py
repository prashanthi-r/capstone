from Config import Config as conf
import sys
from functionalities import functionalities as func
from linearReg import linearReg as linearReg

# command line argument - partyNum, integer input a, integer input b, mask value
# main
def main():
    conf.partyNum = int(sys.argv[1])
    print(conf.partyNum)
    if conf.partyNum == 0:
        conf.PORT = 8002
        conf.advPORT = 8003
    else: 
        conf.PORT = 8003
        conf.advPORT = 8002
    
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

    X,Y,U,V,Vdash,Z,Zdash = linearReg.readData(filename_data,filename_mask)
    model = linearReg.SGDLinear(X,Y,U,V,Vdash,Z,Zdash)
    model = func.truncate(model)
    print(model)



if __name__ == '__main__':
    main()