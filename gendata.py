import random

def generatedata():
    n = random.randint(2,10)
    d = random.randint(2,5)
    X=[]
    U=[]
    V=[]
    Z=[]
    Vdash=[]
    Zdash=[]
    for i in range(n):
        X.append(random.sample(range(10, 100), d+1))
        U.append(random.sample(range(10, 100), d))
    V = random.sample(range(10, 100), d)
    Vdash = random.sample(range(10, 100), n)
    Z = random.sample(range(10, 100), n)
    for i in range(d):
        Zdash.append(random.sample(range(10, 100), n))
    
    print(X)
    print(V)
    with open('data.txt','w') as f:
        for i in range(len(X)):
            f.writelines("%d " % e for e in X[i])
            f.writelines("\n")
        f.close()
    
    with open('mask.txt','w') as f:
        for i in range(len(U)):
            f.writelines("%d " % e for e in U[i])
            f.writelines("\n")
        
        f.writelines("%d " % e for e in V)
        f.writelines("\n")
        f.writelines("%d " % e for e in Vdash)
        f.writelines("\n")
        f.writelines("%d " % e for e in Z)
        f.writelines("\n")

        for i in range(len(Zdash)):
            f.writelines("%d " % e for e in Zdash[i])
            f.writelines("\n")

        f.close()

generatedata()