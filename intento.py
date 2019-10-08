import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal
#leer excel de datos y de dias especiales
general = pd.read_excel (r'C:\Users\Diana\PAP\Data\Data1.xlsx')
special_days= pd.read_excel (r'C:\Users\Diana\PAP\Data\Christmas.xlsx')
#convertir dias especiales a fechas en python
special_days.NewYear = pd.to_datetime(special_days.NewYear)
special_days.Christmas = pd.to_datetime(special_days.Christmas)
special_days.Grito = pd.to_datetime(special_days.Grito)
special_days.Santo = pd.to_datetime(special_days.Santo)
#%%establecer días a pronosticar anteriores
n=-6
general_series=general.MWh
final=general.MWh.tail(n*-1)

#%%hacer varaibles sen y cos para las frecuencias sin efecto dummies
onlyMWh=pd.DataFrame(general_series)
t=np.arange(1,len(onlyMWh)+1)
sencos=pd.DataFrame(columns=['t','one','sen1','cos1','sen2','cos2','sen3','cos3','sen4','cos4','sen5','cos5','sen7','cos7','sen6','cos6','sen8','cos8'])
sencos.t=np.arange(1,len(onlyMWh)+1)
sencos.one=np.ones(len(onlyMWh))
sencos.sen1=np.sin(((2*np.pi)/2872)*t)
sencos.cos1=np.cos(((2*np.pi)/2872)*t)
sencos.sen2=np.sin(((2*np.pi)/1436)*t)
sencos.cos2=np.cos(((2*np.pi)/1436)*t)
sencos.sen3=np.sin(((2*np.pi)/1914.67)*t)
sencos.cos3=np.cos(((2*np.pi)/1914.67)*t)
sencos.sen4=np.sin(((2*np.pi)/45.58)*t)
sencos.cos4=np.cos(((2*np.pi)/45.58)*t)
sencos.sen5=np.sin(((2*np.pi)/820.57)*t)
sencos.cos5=np.cos(((2*np.pi)/820.57)*t)
sencos.sen7=np.sin(((2*np.pi)/1148.8)*t)
sencos.cos7=np.cos(((2*np.pi)/1148.8)*t)
sencos.sen6=np.sin(((2*np.pi)/36.58)*t)
sencos.cos6=np.cos(((2*np.pi)/36.58)*t)
sencos.sen8=np.sin(((2*np.pi)/91.17)*t)
sencos.cos8=np.cos(((2*np.pi)/91.17)*t)

#%%Hacer variables dummies
general = general.set_index('fecha')
general['Month'] = general.index.month
general['Weekday_Name'] = general.index.weekday_name
dummies = pd.get_dummies(general['Weekday_Name']).astype(int)
dummies2 = pd.get_dummies(general['Month']).astype(int)
Dum=pd.DataFrame(dummies.join(dummies2))
Dum["t"]= np.arange(0,len(onlyMWh))
Dum["tiempo"]= np.arange(1,len(onlyMWh)+1)
Dum["ones"]=np.ones(len(t))
Dum= Dum.set_index('t')

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
#%% separar en datos para pronóstico
del dummies,dummies2,date2
del Dum["Friday"]
Dum.drop(Dum.columns[[15]], axis=1,inplace=True)

#%%
sencos_test=sencos[n:]
sencos_train=sencos[0:n]
Dum_test=Dum[n:]
Dum_train=Dum[0:n]
#%%hacer producto kronequer
x=0
Dum_kron=Dum_train[x:x+1]
sencos_kron=sencos_train[x:x+1]
Combinacion=np.kron(Dum_kron,sencos_kron)
Combinacion=pd.DataFrame(Combinacion)
for x in range(1,len(Dum_train)):
    Dum_kron=Dum_train[x:x+1]
    sencos_kron=sencos_train[x:x+1]
    kron=np.kron(Dum_kron,sencos_kron)
    Kron=pd.DataFrame(kron)
    Combinacion=Combinacion.append(Kron)
    

#%%regresión lineal de datos contra porducto kronequer (hecho simple) de dummies y sencos
from sklearn.linear_model import LinearRegression
X =Combinacion
y = general.MWh[:n].values
model = LinearRegression()
model.fit(X, y)
coefficients=model.coef_
prediction= model.predict(X)

#%% graficar
plt.plot(y)
plt.plot(prediction)
#plt.axis([1630,1640,120000,200000])
plt.show()
#%%obtener mape de regresión
comparacion=pd.DataFrame(columns=['real','prediccion','error'])
comparacion.real=y
comparacion.prediccion=prediction
comparacion.error=np.abs((y-prediction)/y)
MAPE=comparacion.error.mean()*100
print(MAPE)

#%% producto kronequer para datos a pronosticar
x=0
Dum_kron=Dum_test[x:x+1]
sencos_kron=sencos_test[x:x+1]
combinaciontest=np.kron(Dum_kron,sencos_kron)
Combinaciontest=pd.DataFrame(combinaciontest)
for x in range(1,len(Dum_test)):
    Dum_kron=Dum_test[x:x+1]
    sencos_kron=sencos_test[x:x+1]
    kron=np.kron(Dum_kron,sencos_kron)
    Kron=pd.DataFrame(kron)
    Combinaciontest=Combinaciontest.append(Kron)
#%% mape de pronostico
ynew = model.predict(Combinaciontest)
comp_pronostico=pd.DataFrame(columns=['real','prediccion','error'])
comp_pronostico.real=final
comp_pronostico.prediccion=ynew
comp_pronostico.error=np.abs((final-ynew)/final)
MAPE=comp_pronostico.error.mean()*100
print(MAPE)
#%%
error=pd.DataFrame(np.abs((y-prediction)/y))
#%%
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(error, lags=400)
plt.axis([360,370,0,.5])
plt.show()
##1,2,3,7,365
#%%
d365=pd.DataFrame(np.zeros(365),columns=["MWh"])
d365=d365.append(onlyMWh[0:-365],ignore_index=True)
#Combinacion=Combinacion.append(d365.MWh)
