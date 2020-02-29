import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "./NHS_challenge_data/"


df = pd.read_csv(path+"BNF_stems.csv")
# df = pd.read_csv(path+"01.csv")
print(df.columns)
# print(df['code_name'])
# v= df['02 Alcoholic beverages and tobacco IDEF SA 2016=100'].dropna()
# plt.plot(v)
# plt.show()
