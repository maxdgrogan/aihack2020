import numpy as np
import pandas as pd
import os
import folium
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

#scaling by number of patients in GP


path = "./NHS_challenge_data/"
#0411.csv
def parse_2(file):
	df = pd.read_csv(path + file)


	v = df.set_index(df[df.columns[0]])
	v.drop(columns=["Unnamed: 0"],axis=1,inplace=True)
	v=v.fillna(0)

	f  = pd.read_csv("./qof-1718-csv/ORGANISATION_REFERENCE.csv")
	f = f[['PRACTICE_CODE','CCG_GEOGRAPHY_CODE']].set_index('PRACTICE_CODE')

	cols = v.columns
	v["Date"]=v.index
	v=v.melt(id_vars=["Date"],value_vars=cols)
	v.columns = ["Date","PRACTICE_CODE","val"]

	df = f.merge(v,on="PRACTICE_CODE")
	df.drop("PRACTICE_CODE",axis=1,inplace=True)

	df = df.groupby(["Date","CCG_GEOGRAPHY_CODE"],as_index=False).agg({'val':'sum'})
	df["Date"] = pd.to_datetime(df.Date)

	t = df.groupby([df.Date.dt.year,"CCG_GEOGRAPHY_CODE"]).agg({'val':'mean'})
	t.reset_index(inplace=True)
	df = df.loc[(df["Date"] >= pd.to_datetime(2016,format='%Y')) & (df["Date"] <= pd.to_datetime(2020,format='%Y'))]
	k = []
	k_sj = ["E38000227"]

	# k_sj = ["E38000104","E38000210","E38000214","E38000227"]
	for cat in t["CCG_GEOGRAPHY_CODE"]:
		temp = df.loc[df["CCG_GEOGRAPHY_CODE"] == cat] 
		temp = (temp['val'] - temp['val'].mean())/temp['val'].std()
		# temp = temp['val']/temp['val'].std()
		c,d=scipy.stats.kstest(temp,'norm')
		# if d<0.1 and cat in k_sj:
		# 	flag=True
		# elif d<0.1 and cat not in k_sj:
		# 	flag=False
		# 	break
		if cat in k_sj:
			return(temp.values)
			# k.extend(temp)
			# plt.show()
		# k.extend(temp.values)
	# if flag:
	# 	print("YES")
	# max_val = t["CCG_GEOGRAPHY_CODE"][t['val'].argmax()]
	# df = df.loc[df["CCG_GEOGRAPHY_CODE"] == max_val].reset_index()
	return(np.array(k))

# for file in os.listdir(path):
# 	try:
v = parse_2("1914.csv")
v2 = parse_2("0411.csv")

# plt.plot(v)
# # plt.plot(v2)
# plt.show()
# plt.plot(v2)
# plt.show()
# 		print("File: " ,file)
# 	except:
# 		print("fuck")
# print(len(v))
# print(len(v))
# np.random()
# print(len(v))
# # print(v)
# print(len(v))
# plt.plot(v)
# plt.show()
# v = (v-v.mean())/v.std()
# print(scipy.stats.kstest(v,'norm'))
plt.plot(v)
plt.plot(v2)
plt.show()
# sns.kdeplot(v,shade=True,label="0302")
# sns.kdeplot(v2,shade=True,label="0411")

# plt.xlabel("Quantity")
# plt.ylabel("Distribution")
# plt.show()

