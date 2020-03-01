import numpy as np
import pandas as pd
import os
import folium

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
	print(df)
	df.drop("PRACTICE_CODE",axis=1,inplace=True)

	df = df.groupby(["Date","CCG_GEOGRAPHY_CODE"],as_index=False).agg({'val':'sum'})
	df["Date"] = pd.to_datetime(df.Date)


	df = df.groupby([df.Date.dt.year,"CCG_GEOGRAPHY_CODE"]).agg({'val':'mean'})
	df.reset_index(inplace=True)


	v = df.loc[df['Date'] == 2016]


	return v

map = folium.Map(location=[54.2, -2.45], zoom_start=5)


# v = parse_2('0411.csv')
# folium.Choropleth(geo_data="ccgs.json",data=v,columns =['CCG_GEOGRAPHY_CODE','val'],data_out='data1.json',key_on='feature.properties.CCG13CD',
#       fill_color='PuBu', fill_opacity=0.7,line_opacity=0.3).add_to(map)
# v = parse_2('0605.csv')
# folium.Choropleth(geo_data="ccgs.json",data=v,columns =['CCG_GEOGRAPHY_CODE','val'],data_out='data1.json',key_on='feature.properties.CCG13CD',
#       fill_color='PuBu', fill_opacity=0.7,line_opacity=0.3).add_to(map)

# v = parse_2('0105.csv')
# folium.Choropleth(geo_data="ccgs.json",data=v,columns =['CCG_GEOGRAPHY_CODE','val'],data_out='data1.json',key_on='feature.properties.CCG13CD',
#       fill_color='PuBu', fill_opacity=0.7,line_opacity=0.3).add_to(map)

# v = parse_2('0403.csv')
# folium.Choropleth(geo_data="ccgs.json",data=v,columns =['CCG_GEOGRAPHY_CODE','val'],data_out='data1.json',key_on='feature.properties.CCG13CD',
#       fill_color='PuBu', fill_opacity=0.7,line_opacity=0.3).add_to(map)




# # for date in range(2017,2020):
# # 	v = df.loc[df['Date'] == date]
# # 	map2 = folium.Map(location=[54.2, -2.45], zoom_start=5)
# # 	folium.Choropleth(geo_data="ccgs.json",data=v,columns =['CCG_GEOGRAPHY_CODE','val'],data_out='data1.json',key_on='feature.properties.CCG13CD',
# #       fill_color='PuBu', fill_opacity=0.7,line_opacity=0.3).add_to(map)

# folium.LayerControl(collapsed=False).add_to(map)

# map.save("fuck2.html")




d=np.array(["E38000104","E38000210","E38000214","E38000227","E38000104","E38000210","E38000214","E38000227","E38000104","E38000210","E38000214","E38000227","E38000104","E38000210","E38000214","E38000227"])
e=np.ones(len(d))
v = np.array([d,e])
v = pd.DataFrame({'CCG_GEOGRAPHY_CODE':d[0:],'val':e[0:]})
folium.Choropleth(geo_data="ccgs.json",data=v,columns =['CCG_GEOGRAPHY_CODE','val'],key_on='feature.properties.CCG13CD',
      fill_color='YlGn', fill_opacity=0.7,line_opacity=0.3).add_to(map)
map.save("fuck3.html")

