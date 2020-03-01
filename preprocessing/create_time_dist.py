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
	# k_sj = ["E38000227"]

	# k_sj = ["E38000104","E38000210","E38000214","E38000227"]
	temp = df.groupby("CCG_GEOGRAPHY_CODE")['val'].transform(sum).drop_duplicates()
	# print(temp)
	for cat in t["CCG_GEOGRAPHY_CODE"]:
		temp = df.loc[df["CCG_GEOGRAPHY_CODE"] == cat] 
		temp = (temp['val'] - temp['val'].mean())/temp['val'].std()
		# temp = temp['val']/temp['val'].std()
		c,d=scipy.stats.kstest(temp,'norm')
		if d>0.1:
			k.append(temp)
		# 	flag=True
		# elif d<0.1 and cat not in k_sj:
		# 	flag=False
		# 	break
		# if cat in k_sj:
			# return(temp.values)
			# k.extend(temp)
			# plt.show()
		# k.extend(temp.values)
	# if flag:
	# 	print("YES")
	# max_val = t["CCG_GEOGRAPHY_CODE"][t['val'].argmax()]
	# df = df.loc[df["CCG_GEOGRAPHY_CODE"] == max_val].reset_index()
	return(np.array(k.values))

# for file in os.listdir(path):
# 	try:
def parse(file):
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

	# df = df.groupby(["CCG_GEOGRAPHY_CODE"],as_index=False).agg({'val':'sum'})
	# df["Date"] = pd.to_datetime(df.Date)

	t = df.groupby(["CCG_GEOGRAPHY_CODE"]).agg({'val':'mean'})
	# t.reset_index(inplace=True)
	# df = df.loc[(df["Date"] >= pd.to_datetime(2016,format='%Y')) & (df["Date"] <= pd.to_datetime(2020,format='%Y'))]
	# k = []
	# k_sj = ["E38000227"]

	# k_sj = ["E38000104","E38000210","E38000214","E38000227"]
	# temp = df.groupby("CCG_GEOGRAPHY_CODE")['val'].agg(sum)
	# print(temp)
	# for cat in t["CCG_GEOGRAPHY_CODE"]:
	# 	temp = df.loc[df["CCG_GEOGRAPHY_CODE"] == cat] 
	# 	temp = (temp['val'] - temp['val'].mean())/temp['val'].std()
	# 	# temp = temp['val']/temp['val'].std()
	# 	c,d=scipy.stats.kstest(temp,'norm')
	# 	if d>0.1:
	# 		k.append(temp)
	# 	# 	flag=True
	# 	# elif d<0.1 and cat not in k_sj:
	# 	# 	flag=False
	# 	# 	break
	# 	# if cat in k_sj:
	# 		# return(temp.values)
	# 		# k.extend(temp)
	# 		# plt.show()
	# 	# k.extend(temp.values)
	# # if flag:
	# # 	print("YES")
	# # max_val = t["CCG_GEOGRAPHY_CODE"][t['val'].argmax()]
	# # df = df.loc[df["CCG_GEOGRAPHY_CODE"] == max_val].reset_index()
	return(t.values)

stor=[]
v = parse("0411.csv") 
v2 = parse("060602.csv") 
v3 = parse("010501.csv") 
v4 = parse("040101.csv") 
# v5 = parse("0209.csv")
# v6 = parse("0604.csv")

v = (v.flatten())
# v2 = (v2.flatten()- np.mean(v2))/np.std(v)
# v3 = (v3.flatten()- np.mean(v3))/np.std(v)
# v4 = (v4.flatten()- np.mean(v4))/np.std(v)
# v5 = v5.flatten()
# v6 = v6.flatten()


# x = np.stack([v,v2,v3,v4])
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.api import VAR
import statsmodels.tsa.stattools as st
print(st.adfuller(v))
# plot_acf(v)
# plt.show()
# # plot_acf(v2)
# plot_acf(v3)

# plt.show()
#all stationary

# v = np.stack([v,v2,v3])
# print(np.shape(x))
# model = VAR(x.transpose())
# results = model.fit(2)
# print(results.summary())
# for i,file in enumerate(os.listdir(path)):
# 	try:
# 		v2 = parse(file)
# 		c = max(len(v),len(v2))
# 		v = v[:c]
# 		v2 = v2[:c]
# 		stor.append("File: " + file + str(scipy.stats.pearsonr(v2,v)))
# 	except:
# 		print("fuck")

# np.savetxt("obv.txt",np.array(stor),fmt='%s')
# print(v)
# plt.plot(v)
# # # plt.plot(v2)
# # plt.show()
# # plt.plot(v2)
# plt.show()
# # 		print("File: " ,file)
# # 	except:
# # 		print("fuck")
# # print(len(v))
# # print(len(v))
# # np.random()
# # print(len(v))
# # # print(v)
# # print(len(v))
# # plt.plot(v)
# # plt.show()
# # v = (v-v.mean())/v.std()
# # print(scipy.stats.kstest(v,'norm'))
# # plt.plot(v)
# # plt.plot(v2)
# # plt.show()
# print(v)
# sns.kdeplot(v,shade=True,label="0411")
# sns.kdeplot(v2,shade=True,label="060602")
# sns.kdeplot(v3,shade=True,label="010501")

# plt.xlabel("Quantity")
# plt.ylabel("Distribution")
# plt.show()

