import json
from itertools import groupby
import pandas as pd
with open("data_banknote_authentication.txt") as fp:
    data=fp.read().splitlines()
    temp=[]
    for i in data:
        temp.append(i.split(","))
    y=[]
    for i in temp:
        y.append(i[4])
    print([len(list(group)) for key, group in groupby(y)])
    #print(temp)