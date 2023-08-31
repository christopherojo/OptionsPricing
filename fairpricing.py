import math

def combos(n, i):
    return math.factorial(n) / (math.factorial(n-i)*math.factorial(i))

for i in range(5):
    print("Outcomes with {}-heads = {}".format(i,combos(4,i)))

fair_value = 0 
n= 4 # number of coin flips
for k in range(n+1):
    fair_value += combos(n,k)*0.5**k*0.5**(n-k) * k
    
print("FV: {}".format(fair_value))

fair_value = 0 
n= 10 # number of coin flips
for k in range(7,n+1):
    fair_value += combos(n,k)*0.5**k*0.5**(n-k) * k
    
print("FV: {}".format(fair_value))