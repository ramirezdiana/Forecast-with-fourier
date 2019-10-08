import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import LinearRegression
general = pd.read_excel (r'C:\Users\Diana\PAP\Data\Data1.xlsx')
special_days= pd.read_excel (r'C:\Users\Diana\PAP\Data\Christmas.xlsx')
#season= pd.read_excel (r'C:\Users\Diana\PAP\Data\Special Dates.xlsx')
special_days.NewYear = pd.to_datetime(special_days.NewYear)
special_days.Christmas = pd.to_datetime(special_days.Christmas)
special_days.Grito = pd.to_datetime(special_days.Grito)
special_days.Santo = pd.to_datetime(special_days.Santo)
#%%

general_series=general.MWh
onlyMWh=pd.DataFrame(general_series)

general = general.set_index('fecha')
general['Month'] = general.index.month
general['Weekday_Name'] = general.index.weekday_name
dummies = pd.get_dummies(general['Weekday_Name']).astype(int)
dummies2 = pd.get_dummies(general['Month']).astype(int)
Dum=pd.DataFrame(dummies.join(dummies2))
t=np.arange(0,len(onlyMWh))
Dum["t"]= np.arange(0,len(onlyMWh))
Dum["tiempo"]= np.arange(1,len(onlyMWh)+1)
Dum["ones"]=np.ones(len(t))
Dum= Dum.set_index('t')
#%%
Dum["Christmas"]=0
Dum["NewYear"]=0
Dum["Grito"]=0
Dum["Santo"]=0
ind=0 
for date in general.index:
    for date2 in special_days["Christmas"]:
        if date ==date2:
            Dum.iloc[ind,21]=1
    for date2 in special_days["NewYear"]:
        if date ==date2:
            Dum.iloc[ind,22]=1
    for date2 in special_days["Grito"]:
        if date ==date2:
            Dum.iloc[ind,23]=1
    for date2 in special_days["Santo"]:
        if date ==date2:
            Dum.iloc[ind,24]=1
    ind+=1
del Dum["Friday"]
Dum.drop(Dum.columns[[15]], axis=1,inplace=True)
#%%
#Dum["verano"]=season["VERANO"]
#Dum["otoño"]=season["OTOÑO"]
#Dum["invierno"]=season["INVIERNO"]

#%%con producto kroneker
t=np.arange(1,len(onlyMWh)+1)
Tiempo=pd.DataFrame(t)
Tiempo["one"]=np.ones(len(onlyMWh))
x=0
Dum_kron=Dum[x:x+1]
t_kron=Tiempo[x:x+1]
Combinacion=np.kron(Dum_kron,t_kron)
Combinacion=pd.DataFrame(Combinacion)
for x in range(1,len(Dum)):
    Dum_kron=Dum[x:x+1]
    t_kron=Tiempo[x:x+1]
    kron=np.kron(Dum_kron,t_kron)
    Kron=pd.DataFrame(kron)
    Combinacion=Combinacion.append(Kron)
#%%
X =Combinacion
y = general.MWh.values
model = LinearRegression()
model.fit(X, y)
coefficients=model.coef_
prediction= model.predict(X)
#%%
plt.plot(y)
plt.plot(prediction)
#plt.axis([1630,1640,120000,200000])
#plt.plot(t,ynew)
plt.show()
#%%
Tabla=pd.DataFrame(columns=['regresion','datos','resta'])
Tabla["regresion"]=prediction
Tabla["datos"]=onlyMWh
Tabla["resta"]=Tabla.datos-Tabla.regresion
#plt.plot(general.MWh.values)
plt.plot(Tabla.resta)
plt.show()
#%%
fs = 1
f, Pxx_den = signal.periodogram(Tabla.resta, fs)
plt.plot(1/f, Pxx_den)
plt.xlabel('periodo')
plt.ylabel('PSD')
#plt.axis([0,1000,0,10000000000])
#plt.axis([2,2.5,0,200000000000])
#plt.axis([6,8,0,200000000000])
plt.show()
#%%
top_3_periods = {}
# get indices for 3 highest Pxx values
top5_freq_indices = np.flip(np.argsort(Pxx_den), 0)[0:50]

freqs = f[top5_freq_indices]
power = Pxx_den[top5_freq_indices]
periods = 1 / np.array(freqs)
matrix=pd.DataFrame(columns=["power","periods"])
matrix.power=power
matrix.periods=periods
