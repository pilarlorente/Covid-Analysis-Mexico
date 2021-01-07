#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### Import packages
# Basic packages
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json
import geopandas as gpd

# Data Visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import FeatureGroup, LayerControl, Map


# To avoid warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


##### Import data
df_st= pd.read_csv("CoordEstados.csv", encoding = "ISO-8859-1")
df_cov=pd.read_csv("201114COVID19MEXICO.csv", encoding = "ISO-8859-1")


# In[3]:


def autolabel(plot):
    for p in plot.patches:
        plot.annotate(format(p.get_height(), '.0f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points') 

# Plot a pie chart    
def graph_pie(dictionary,title):
    colors = ['#DAF7A6','#FFC300','#FF5733','#C70039','#581845','#ff9999','#66b3ff','#99ff99','#ffcc99']
    dictionary = dict(sorted(dictionary.items(), key = lambda item: item[1], reverse=True))
    plt.figure(figsize=(10, 10))
    plt.pie(dictionary.values(), labels = dictionary.keys(), explode = [0.1 for i in range(len(dictionary.values()))],
            autopct='%.1f%%', shadow = True, labeldistance = 1.07, startangle = 45, colors=colors)
    plt.title(title, {'fontsize':25} )
    centre_circle = plt.Circle((0,0),0.80,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# In[4]:


####Statistics

# States info
print(' **Mexican states data** '.center(50))
df_st.info()
df_st.describe()
print(' ')

# Covid-19 data
print(' **Covid-19 data** '.center(50))
df_cov.info()
df_cov.describe()


# In[5]:


##### Number of missing values
print('**MEXICAN STATES**'.center(50))
print('Number of NaNs:',df_st.isna().sum().sum())
print('**COVID DATA**'.center(50))
print('Number of NaNs:', df_cov.isna().sum().sum())


# In[6]:


##### Change Mexican states from name to number
#dictionary with name and number
dict_st=pd.Series(df_st['Estado'].values,index=df_st['Clave Estado']).to_dict()

#apply dict to columns 'ENTIDAD'
df_cov['ENTIDAD_RES']=df_cov['ENTIDAD_RES'].map(dict_st)
df_cov['ENTIDAD_UM']=df_cov['ENTIDAD_UM'].map(dict_st)
df_cov['ENTIDAD_NAC']=df_cov['ENTIDAD_NAC'].map(dict_st)


# In[7]:


##### Transform all the mexican National Health System institutions from number to name

dict_sector = {1:'CRUZ ROJA', 2:'DIF',3:'ESTATAL',4:'IMSS',5:'IMSS-BIENESTAR',6:'ISSSTE',7:'MUNICIPAL',8:'PEMEX',
               9:'PRIVADA',10:'SEDENA',11:'SEMAR',12:'SSA',13:'UNIVERSITARIO',99:'NO ESPECIFICADO'}

df_cov['SECTOR'] = df_cov['SECTOR'].map(dict_sector)


# In[8]:


####New column with the period between being diagnosed covid positive and die.
    # if the person didn't die, the value is 0
    
df_cov['FECHA_SINTOMAS']=df_cov['FECHA_SINTOMAS'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df_cov['FECHA_DEF']=df_cov['FECHA_DEF'].replace('9999-99-99', '2001-01-01')

df_cov['FECHA_DEF']=df_cov['FECHA_DEF'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df_cov['DIFERENCIA'] = df_cov['FECHA_DEF'].sub(df_cov['FECHA_SINTOMAS'], axis=0)

df_cov['DIFERENCIA'] = df_cov['DIFERENCIA'] / np.timedelta64(1, 'D')
df_cov.loc[df_cov['DIFERENCIA']<0,'DIFERENCIA'] = 0


# ## POSITIVE COVID-19
# 
# 

# In[9]:


##### Positive cases by gender, age and sector
#The meaning of each feauture could be found at Catalago and Descriptores excels.
# Positive cases are marked as 1,2 or 3 in the CLASIFICACION_FINAL column

df_pos=df_cov[df_cov['CLASIFICACION_FINAL'].isin([1,2,3])]


# In[10]:


fig, axes = plt.subplots(1, 3, figsize=(40, 10))

#Gender
axes[0].set_title('Gender of positive cases', {'fontsize':40})
plot = sns.barplot(x = ['Women','Men'],y = df_pos['SEXO'].value_counts().values, palette = 'pastel', ax = axes[0])
plot.set(ylim = (450000,550000), ylabel = None, yticklabels = [])
plot.tick_params(left=False)
autolabel(plot)

#Age distribution of positives cases
axes[1].set_title('Age distribution of positive cases', {'fontsize':40})
plot = sns.distplot(df_pos['EDAD'], ax = axes[1])
plot.set(ylabel = None, yticklabels = [])
plot.tick_params(left=False)

#Sector 
axes[2].set_title('National Health System institution \n that provided the care', {'fontsize':40})
data = df_pos['SECTOR'].value_counts()
plot = sns.barplot(x = data.index[:5], y = data.values[:5],
                   palette = 'pastel', ax = axes[2])
plot.set(ylim = (10000,650000), ylabel = None, yticklabels = [])
plot.tick_params(left=False)
autolabel(plot)


# In[11]:


##### Positive cases by state
data=df_pos.groupby('ENTIDAD_RES').count()['CLASIFICACION_FINAL'].sort_values(ascending=False)

print('Total number of positive cases = {} \n '.format(df_pos.shape[0]).center(85))
plt.figure(figsize=(35, 10))
plot = sns.barplot(x = data.index, y = data.values, palette="pastel")
plt.title('Positive cases by state', {'fontsize':60})
plt.xticks(rotation=75)
autolabel(plot)
plt.show()


# In[12]:


##### Mean age of positive cases by state
data=df_pos.groupby('ENTIDAD_RES')['EDAD'].mean().sort_values(ascending=False)
print('Mean age between positive cases = {} \n '.format(df_pos.EDAD.mean()).center(85))
plt.figure(figsize=(35, 10))
plot = sns.barplot(x = data.index, y = data.values, palette="pastel")
plt.title('Mean age between positive cases by state', {'fontsize':60})
plt.xticks(rotation=75)
autolabel(plot)
plt.show()


# In[13]:


##### Common illness on total positive cases 

names= ['DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA']
dict_ill = dict()
for name in names:
    dict_ill[name] = df_pos.query(f'{name} == 1').shape[0]
print('Most common illness on total positive cases  = {}'.format(max(dict_ill, key=dict_ill.get)).center(75))


graph_pie(dict_ill,'Common illness on total positive cases')


# In[14]:


names = ['DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA']
dict_ill=dict()
for state in sorted(df_cov['ENTIDAD_RES'].unique()):
    print('{}'.format(state).center(75))
    for name in names:
        dict_ill[name]= df_pos.query(f'{name}== 1 & ENTIDAD_RES== "{state}"').shape[0]
    print(f'Common illness = {max(dict_ill, key=dict_ill.get)} \n'.center(75))
    graph_pie(dict_ill,f'Common illness on total positive cases at {state}')
        


# ## DECEASED CASES

# In[15]:



# FECHA_DEF is 2001-01-01 when covid positive but not die.

df_dec =  df_cov[df_cov['FECHA_DEF'] !='2001-01-01']

fig, axes = plt.subplots(1, 2, figsize=(40, 10))

#Gender 
axes[0].set_title('Gender of deceased cases',  {'fontsize':40})
plot = sns.barplot(x = ['Women','Men'],y = df_dec['SEXO'].value_counts().values, palette = 'pastel', ax = axes[0])
#plot.set(ylim = (1200000,1400000), ylabel = None, yticklabels = [])
plot.tick_params(left=False)
autolabel(plot)

#Age distribution 
axes[1].set_title('Age distribution of deceased cases',  {'fontsize':40})
plot = sns.distplot(df_dec.EDAD, ax = axes[1])
plot.set(ylabel = None, yticklabels = [])
plot.tick_params(left=False)

plt.tight_layout()
plt.show()


# In[16]:


##### Deceased cases by state
data=df_dec.groupby('ENTIDAD_RES').count()['CLASIFICACION_FINAL'].sort_values(ascending=False)

print('Total number of deceased cases = {} | Represents {}% of the total \n '.format(df_dec.shape[0], df_dec.shape[0]/df_cov.shape[0]).center(85))
plt.figure(figsize=(35, 10))
plot = sns.barplot(x = data.index, y = data.values, palette="rocket")
plt.title('Deceased cases by state', {'fontsize':60})
plt.xticks(rotation=75)
autolabel(plot)
plt.show()


# In[17]:


##### Mean age of deceased by state
data=df_dec.groupby('ENTIDAD_RES')['EDAD'].mean().sort_values(ascending=False)

print('Mean age of deceased = {} \n '.format(df_dec.EDAD.mean()).center(85))
plt.figure(figsize=(35, 10))
plot = sns.barplot(x = data.index, y = data.values, palette="rocket")
plt.title('Mean age of deceased by state', {'fontsize':60})
plt.xticks(rotation=75)
autolabel(plot)
plt.show()


# In[18]:


##### Period(days) between date when a person has been diagnosed positive and die.

data = df_dec['DIFERENCIA'].value_counts()

print('Mean period between get positive and die = {} \n '.format(df_dec.DIFERENCIA.mean()).center(85))
plt.figure(figsize=(35, 10))
plot = sns.barplot(x = data.index[1:15], y = data.values[1:15], palette="rocket")
plt.title('Days between get positive and die ', {'fontsize':60})
plt.xticks(rotation=75)
autolabel(plot)
plt.show()


# In[19]:


##### Common illness on total deceased cases 

names= ['DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA']
dict_ill = dict()
for name in names:
    dict_ill[name] = df_dec.query(f'{name} == 1').shape[0]
print('Most common illness on deceased cases  = {}'.format(max(dict_ill, key=dict_ill.get)).center(75))


graph_pie(dict_ill,'Common illness on deceased cases')


# In[20]:


#common illnes of deceased by state
names = ['DIABETES','EPOC','ASMA','INMUSUPR','HIPERTENSION','OTRA_COM','CARDIOVASCULAR','OBESIDAD','RENAL_CRONICA']
dict_ill=dict()
for state in sorted(df_cov['ENTIDAD_RES'].unique()):
    print('{}'.format(state).center(75))
    for name in names:
        dict_ill[name]= df_dec.query(f'{name}== 1 & ENTIDAD_RES== "{state}"').shape[0]
    print(f'Common illness = {max(dict_ill, key=dict_ill.get)} \n'.center(75))
    graph_pie(dict_ill,f'Common illness on deceased cases at {state}')


# In[21]:


gasta aqui


# ## MAP

# In[22]:


# Replace the wrongs names of the states in the DataFrame with the json 

with open("mexico22.json") as f:
    data = json.load(f)
    
states_json = list()
for i in range(32):
    states_json.append(data['features'][i]['properties']['name'])
    
states_json = sorted(states_json)
states_df = sorted(df_cov.ENTIDAD_RES.unique())

dict_states = dict()
for json,df in zip(sorted(set(states_json) - set(states_df)),sorted(set(states_df) - set(states_json))):
    dict_states[df] = json
    
df_cov['ENTIDAD_RES'] = df_cov['ENTIDAD_RES'].replace(dict_states)


# In[59]:


##### Number of confirmed cases and deceased cases on Mexico (dataset)

data = gpd.read_file("mexico22.json").sort_values('name',ascending = True).reset_index(drop = True)
data.rename(columns = {'name': "States"}, inplace=True)

data['Positive']=df_cov[(df_cov['CLASIFICACION_FINAL'].isin([1,2,3])) & (~(df_cov['ENTIDAD_NAC'].isin([97,98,99])))].groupby('ENTIDAD_RES').count()['CLASIFICACION_FINAL'].values
data['Death']=df_cov[(~(df_cov['FECHA_DEF'] =='2001-01-01')) & (~(df_cov['ENTIDAD_NAC'].isin([97,98,99])))].groupby('ENTIDAD_RES').count()['CLASIFICACION_FINAL'].values


# In[60]:


# Creation of the map
mexico_map = folium.Map(location=[23.634501, -102.552784], zoom_start=5.5, tiles = None)
folium.TileLayer('http://tile.stamen.com/watercolor/{z}/{x}/{y}.png', name = "Watercolor map", control = False, attr = "toner-bcg").add_to(mexico_map)
mexico_geo = r"mexico22.json"


# In[61]:


# Adding confirmed cases layer
choropleth = folium.Choropleth(
    name='Confirmed cases',
    geo_data = mexico_geo,
    data = data,
    columns = ['States','Positive'],
    key_on = 'feature.properties.name',
    fill_color = 'YlOrRd', 
    fill_opacity = 0.65, 
    line_opacity = 0.5,
    threshold_scale = list(data['Positive'].quantile([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1])),
    overlay= False)

for key in choropleth._children:
    if key.startswith('color_map'):
        del(choropleth._children[key])
choropleth.add_to(mexico_map)


# In[62]:


# Adding deceased cases layer
choropleth = folium.Choropleth(
    name='Deceased cases',
    geo_data = mexico_geo,
    data = data,
    columns = ['States','Death'],
    key_on = 'feature.properties.name',
    fill_color = 'YlOrBr', 
    fill_opacity = 0.65, 
    line_opacity = 0.5,
    threshold_scale = list(data['Death'].quantile([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1])),
    overlay= False)

for key in choropleth._children:
    if key.startswith('color_map'):
        del(choropleth._children[key])
choropleth.add_to(mexico_map)


# In[63]:


# Adding pop-up tooltips
style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}

highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}


# In[64]:


data_Geo = gpd.GeoDataFrame(data , geometry = data.geometry)


# In[65]:


pop_up = folium.features.GeoJson(
    data_Geo,
    style_function = style_function, 
    control = False,
    highlight_function = highlight_function, 
    tooltip = folium.features.GeoJsonTooltip(
        fields=['States','Positive', 'Death'],
        aliases=['State: ','Number of COVID-19 confirmed cases: ', 'Number of COVID-19 deceased cases: '],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px;")))
mexico_map.add_child(pop_up)
mexico_map.keep_in_front(pop_up)


# In[68]:


# To control the layers
folium.LayerControl(collapsed=False).add_to(mexico_map)


# In[69]:



mexico_map


# In[ ]:




