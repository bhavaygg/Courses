abc=input()
import operator
nuc=["A","T","C","G"]
strs=[]
for n,i in enumerate(abc):
    for j in nuc:
        if j!=i:
            if n!=len(abc)-1:
                strs.append(abc[:n]+j+abc[n+1:])
            else:
                strs.append(abc[:n]+j)
for n,i in enumerate(abc):  
    for n1,j in enumerate(abc):
        if n1>n:
            for k in nuc:
                for l in nuc:
                    if k!=i and l!=j:
                        if n1!=len(abc)-1:
                            strs.append(abc[:n]+k+abc[n+1:n1]+l+abc[n1+1:])
                        else:
                            strs.append(abc[:n]+k+abc[n+1:n1]+l)
consensus=""
for i in range(0,len(abc)):
    count={"A":0,"T":0,"C":0,"G":0}
    for n,j in enumerate(strs):
        count[strs[n][i]]+=1
    
    print(count)
    print(max(count.items(), key=operator.itemgetter(1))[0])

print(strs)
