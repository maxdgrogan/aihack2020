import numpy as np
import pandas as pd
import os
path = "./NHS_challenge_data/"

f  = pd.read_csv("./qof-1718-csv/ORGANISATION_REFERENCE.csv")
f = f[['PRACTICE_CODE','CCG_GEOGRAPHY_CODE']].set_index('PRACTICE_CODE')


def parse_f(filename):
	df = pd.read_csv(path + filename)
	v = df.set_index(df[df.columns[0]])
	v.drop(columns=["Unnamed: 0"],axis=1,inplace=True)		
	v=v.fillna(0)
	cols = v.columns
	v["Date"]=v.index
	v=v.melt(id_vars=["Date"],value_vars=cols)
	v.columns = ["Date","PRACTICE_CODE","val"]

	df = f.merge(v,on="PRACTICE_CODE")
	df.drop("PRACTICE_CODE",axis=1,inplace=True)
	df = df.groupby(["Date","CCG_GEOGRAPHY_CODE"],as_index=False).agg({'val':'sum'})
	df["Date"] = pd.to_datetime(df.Date)
	df = df.groupby([df.Date.dt.year,"CCG_GEOGRAPHY_CODE"]).agg({'val':'mean'})
	df.reset_index(inplace=True)
	v = df.loc[df['Date'] == 2016]
	v['val'] = v['val']/v['val'].mean()
	v = v.drop("Date",axis=1)
	return(v)


v = parse_f("2144.csv")
# print(v)
corr_score = []
for n,file in enumerate(os.listdir(path)):
	try:
		v2 = parse_f(file)
		temp = v2.merge(v,on="CCG_GEOGRAPHY_CODE")
		vu = ((temp['val_y']-temp['val_x'])**2).mean()/float(temp['val_y'].size)
		corr_score.append(file + "   :" + str(vu))
	except:
		print(file + " skipped")
np.savetxt("ok.txt", corr_score,fmt='%s')
# 	val = parse_f(file)
# 	corr_score.append(val-v)
# print(v2)
# print(v)
# print(v2-v)




# import statsmodels.api as sm
# from statsmodels.api import OLS
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# import statsmodels.api as sm
# import matplotlib.pyplot as plt





# print(v)
# v=v.apply(np.floor)
# print(v)
# v['idx'] = v.index
# print(v)
# print(df.columns)



# # idx = 24
# res = sm.tsa.seasonal_decompose(tot.interpolate(),
#                                 freq=2,
#                                 model='additive')
# # resplot = res.plot()
# t = (res.trend + res.resid).dropna()
# # print(t)
# # plot_acf(t)
# # plt.plot(t)
# # plot_acf(tot)
# # plt.show()
# # # plot_acf(tot)
# # plot_acf(v[v.columns[idx]])
# # # plot_pacf(v[v.columns[idx]])
# # plt.show()
# date = "2019-01-01"
# v2 = df.loc[df['Date'] == date]

# # v = df.groupby("[[0]]")
# v['val']=(v['val']/10000.0)
# v2['val']=(v2['val']/10000.0)

# print(v)
# print(v2)




# # # t = (v[v.columns[idx]] - v[v.columns[idx]].shift(-2))
# # # t=t[~np.isnan(t)]

# # # t2 = tot.rolling(window=2).mean().dropna()
# # # de_trended = tot-t2
# # # plt.plot(de_trended)
# # t2 = tot-tot.shift(2)
# # t2 = t2.dropna()
# # # # plt.show()
# # # plt.plot(t2)
# # # plot_pacf(t2)
# # # plt.plot(tot)
# # # plt.plot(t2)
# # # plot_acf(tot)
# # # plot_acf(t2)

# # # plt.show()

# # # v['rolling'] = (v['C83037'] - v['C83037'].rolling(window=12).mean()) / v['C83037'].rolling(window=12).std()
# # # v['rolling_z'] = v['rolling'] - v['rolling'].shift(12)

# # from statsmodels.tsa.stattools import adfuller
# # dftest = adfuller(t, autolag='AIC')
# # print(" > Is the data stationary ?")
# # print("Test statistic = {:.3f}".format(dftest[0]))
# # print("P-value = {:.3f}".format(dftest[1]))
# # print("Critical values :")
# # for k, v in dftest[4].items():
# #     print("\t{}: {} - The data is {} stationary with {}% confidence".format(k, v, "not" if v<dftest[0] else "", 100-int(k[:-1])))
    
# # # print("\n > Is the de-trended data stationary ?")
# # # dftest = adfuller(t, autolag='AIC')
# # # print("Test statistic = {:.3f}".format(dftest[0]))
# # # print("P-value = {:.3f}".format(dftest[1]))
# # # print("Critical values :")
# # # for k, v in dftest[4].items():
# # #     print("\t{}: {} - The data is {} stationary with {}% confidence".format(k, v, "not" if v<dftest[0] else "", 100-int(k[:-1])))
    
# # # print("\n > Is the 12-lag differenced de-trended data stationary ?")
# # # dftest = adfuller(v['rolling_z'].dropna(), autolag='AIC')
# # # print("Test statistic = {:.3f}".format(dftest[0]))
# # # print("P-value = {:.3f}".format(dftest[1]))
# # # print("Critical values :")
# # # for k, v in dftest[4].items():
# # #     print("\t{}: {} - The data is {} stationary with {}% confidence".format(k, v, "not" if v<dftest[0] else "", 100-int(k[:-1])))



# # # # plt.plot(v['C83037'].rolling(window=12).std())
# # # # plt.show()
# # # # model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=3)
# # # # fit = model.fit()

# # # # print(df)

# # # # # df = pd.read_csv(path+"BNF_stems.csv")
# # # # for t,file in enumerate(os.listdir(path)):
# # # # 	if t==3:
# # # # 		break
# # # # 	df = pd.read_csv(path+file)
# # # # 	for val in df.columns:
# # # # 		gp_dat = [df[df.columns[0]],df[val]]
# # # # 		print(gp_dat)


# # # # df = pd.read_csv(path+"01.csv")
# # # # print(df.head())
# # # # # print(df['code_stem'])
# # # # # print()
# # # # # print(df['code_name'])
# # # # # v= df['02 Alcoholic beverages and tobacco IDEF SA 2016=100'].dropna()
# # # # # plt.plot(v)
# # # # # plt.show()
