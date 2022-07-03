import random
import operator

def consensus(count,abc,pos):
    for i in range(0,len(abc)):
        for n,j in enumerate(strs):
            count[strs[n][i]]+=1
    return count

nuc=["A","T","C","G"]
stras=[]
strs=[]
for i in range(0,100):
    temp=""
    for i in range(0,1000):
        d=random.randint(0,3)
        temp+=(nuc[d])
    stras.append(temp)

motif=""
for i in range(0,10):
    t = random.randint(0,3)
    motif+=nuc[t]
for n,i in enumerate(motif):
    for j in nuc:
        if j!=i:
            if n!=len(motif)-1:
                strs.append(motif[:n]+j+motif[n+1:])
            else:
                strs.append(motif[:n]+j)
for n,i in enumerate(motif):  
    for n1,j in enumerate(motif):
        if n1>n:
            for k in nuc:
                for l in nuc:
                    if k!=i and l!=j:
                        if n1!=len(motif)-1:
                            strs.append(motif[:n]+k+motif[n+1:n1]+l+motif[n1+1:])
                        else:
                            strs.append(motif[:n]+k+motif[n+1:n1]+l)

print("Motif:",motif)
motifs=strs
mod_strs=[]

for i in stras:
    pos=random.randint(0,len(i)-1)
    ra_motif = random.randint(0,len(motifs)-1)
    if pos != len(i)-1:
        mod_strs.append(i[:pos]+motifs[ra_motif]+i[pos:])
    else:
        mod_strs.append(i[:pos]+motifs[ra_motif])


d=1
change=[0 for x in range(100)]
rs = [random.randint(0, 99) for x in mod_strs]
while d==1:
    matrix=[]
    for n,i in enumerate(rs):
        matrix.append(mod_strs[n][i:i+10])
    seq_num = random.randint(0,99)
    seq = mod_strs[seq_num]
    con=""
    count_mat=[]
    pp_vals=[]
    for i in range(0,10):
        count={"A":0,"T":0,"C":0,"G":0}
        for n,k in enumerate(matrix):
            if n!=seq_num:
                count[matrix[n][i]]+=1
        count_mat.append(count)
        con+=max(count.items(), key=operator.itemgetter(1))[0]
    for n,i in enumerate(seq[:-10]):
        temp=seq[n:n+10]
        pp=1
        for n1,j in enumerate(temp):
            pp*= count_mat[n1][j]/sum(count_mat[n1].values())
        pp_vals.append(pp)
    ind=pp_vals.index(max(pp_vals))
    max_pp = max(pp_vals)
    if rs[seq_num]==ind:
        change[seq_num]=1
    else:
        rs[seq_num] = ind
    if 0 not in change:
        d=0

print("Consensus Motif",con)

