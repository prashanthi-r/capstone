from Config import Config as config
import sys
from functionalites import functionalites as func

# command line argument - partyNum, integer input a, integer input b, mask value
# main
def main():
    config.partyNum = int(sys.argv[1])
    a = float(sys.argv[2])
    b = float(sys.argv[3])

    if(config.partyNum == 0):
        config.PORT = 8002
        config.advPORT = 8003
    else: 
        config.PORT = 8003
        config.advPORT = 8002
    
    ############# add shares ################
    u = float(sys.argv[4])
    output = addshares(a,b,u)
    print(output)

    ############ multiply shares ##############

    u = float(sys.argv[4])
    v = float(sys.argv[5])
    z = float(sys.argv[6])
    output_mul = multiplyshares(a,b,u,v,z)
    print(output_mul)

if __name__ == '__main__':
    main()