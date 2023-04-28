# -*- coding: utf-8 -*-
"""
Projeto Mestrado Computação Aplicada - IFES
Título: Uso de PINNs associadas a Balanço de Materiais de Reservatórios
Autor: Marcelo Garret de Melo Filho
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from sklearn.metrics import r2_score
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.preprocessing import StandardScaler
# import copy
# import seaborn as sns
# import tensorflow as tf
# from sklearn.linear_model import LinearRegression
# import xgboost as xgb
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import TimeSeriesSplit

df=pd.read_excel('unisim_hist.xlsx')
# df=df.drop(["Press_b", "Np_b", "Gp_b", "Wp_b", "Winj_b"], axis=1)
# print(df.head())

# print(df.dtypes)

## Parametros escalares (MODSI)
phi = 0.13
k = 77
m = 0.0
N = 130286000
Rsi = 113.06
Bw = 1.0210
uw = 0.3
cf = 5.3E-05
pb = 210.3
bob = 1.414
co = 0.000162
cw = 47.6E-06
Swi = 0.35

df["Rp"] = df["Gp"]/df["Np"]
df["t"] = (df["Date"]-df["Date"].iloc[0]).astype("int64")/10**9/60/60/24
df["dt"]=df["t"].diff()
df["p"] = df["Press"].iloc[0]-df["Press"]
df["dp"]=df["p"].diff()
# print(df.head())

## PVT functions

def func_bo(p, a, b):
    bo1 = a*p[p<pb]+b 
    bo2 = bob+co*bob*(pb-p[p>=pb])
    return np.append(bo1, bo2)

def func_bg(p, a, b):
    return a*p**(b)

def func_rs(p, a, b):
    rs1 = a*p[p<pb]+b 
    rs2 = Rsi*np.ones(len(p[p>=pb]))
    return np.append(rs1, rs2)

p = df["Press"].values

bo = func_bo(p, 0.0012, 1.1538)
df["Bo"]=bo

bg = func_bg(p, 1.40676, -1.04229)
df["Bg"]=bg

rs = func_rs(p, 0.4655, 15.0114)
df["Rs"]=rs

df["Bt"]=df["Bo"]+(Rsi-df["Rs"])*df["Bg"]
df["F"] = df["Np"]*(df["Bt"]+(df["Rp"]-Rsi)*df["Bg"])+(df["Wp"]-df["Winj"])*Bw
df["Eo"] = df["Bt"]-df["Bt"].iloc[0]
df["Eg"] = df["Bt"].iloc[0]*(df["Bg"]/df["Bg"].iloc[0]-1)
df["Efw"] = df["Bt"].iloc[0]*((cf+cw*Swi)/(1-Swi))*df["p"]
df["We"] = df["F"]-N*(df["Eo"]+m*df["Eg"]+(1+m)*df["Efw"])
df["x"] = df["We"]/(df["Eo"]+m*df["Eg"]+(1+m)*df["Efw"])/1E6
df["y"] = df["F"]/(df["Eo"]+m*df["Eg"]+(1+m)*df["Efw"])/1E6

# df = df.set_index("Date")
# df=df.dropna()

#print(df)

#plt.scatter(df["x"], df["y"])
#plt.title("EBM linear")
#plt.xlabel("We/(Eo+mEg+(1+m)Efw)")
#plt.ylabel("F/(Eo+mEg+(1+m)Efw)")
#plt.show()

train = df.copy()
train = train.drop(["Gp", "Bt", "Bo", "Bg", "Rs", "F", "Eo", "Eg", "Efw", "x", "y", "p", "dt", "dp"], axis=1)

p=train["Press"].values
we=train["We"].values
t=train["t"].values
npp=train["Np"].values
rp=train["Rp"].values
wp=train["Wp"].values
winj=train["Winj"].values
we[0]=0
we[1]=0
rp[0]=0
rp[1]=0

##Aquífero Schilthuis
def func_we1(t, J):
    dt=t-np.roll(t,1)
    p_med=(p+np.roll(p,1))/2
    pmt = ((p[0]-p_med[2:])*dt[2:]).cumsum()
    return np.append(np.zeros(2), J*pmt[0:len(t)])

##Aquífero Hurst Modificado
def func_we2(t, C, a):
    dt=t-np.roll(t,1)
    p_med=(p+np.roll(p,1))/2
    pmt = (((p[0]-p_med[1:])*dt[1:])/np.log(a*t[1:])).cumsum()
    return np.append(np.zeros(1), C*pmt[0:len(t)])

##Aquífero Fetkovich
def func_we3(t, Wei, J):
    dt=t-np.roll(t,1)
    p_med=(p+np.roll(p,1))/2
    we3=np.zeros(len(p))
    pa_med=np.zeros(len(p))
    for i in range(len(p)):
        if p[i] == p[0]:
            dt[i] = 0
            p_med[i] = p[0]
            we3[i] = 0
            pa_med[i] = p[0]
        else:
            dt[i]=t[i]-t[i-1]
            p_med[i]=(p[i]+p[i-1])/2
            we3[i]=we3[i-1]+(Wei/p[0])*(pa_med[i-1]-p_med[i])*(1-np.exp(-J*p[0]*dt[i]/Wei))
            pa_med[i]=p[0]*(1-we3[i]/Wei)
    return we3

print(train.head())

initialGuess1=[10000]
initialGuess2=[15,0.01]
initialGuess3=[1E6,15]
popt1,pcov1 = curve_fit(func_we1, t, we, initialGuess1)
popt2,pcov2 = curve_fit(func_we2, t, we, initialGuess2)
popt3,pcov3 = curve_fit(func_we3, t, we, initialGuess3)
print(popt1, popt2, popt3)

fittedData1=func_we1(t, *popt1)
fittedData2=func_we2(t, *popt2)
fittedData3=func_we3(t, *popt3)
r2_1 = r2_score(we, fittedData1)
r2_2 = r2_score(we, fittedData2)
r2_3 = r2_score(we, fittedData3)
print(r2_1, r2_2, r2_3)

train["We_pred1"]=fittedData1
train["We_pred2"]=fittedData2
train["We_pred3"]=fittedData3

plt.scatter(t, we, label="Data", color="blue")
plt.plot(t, fittedData1, label=f"Schilthuis: J = {popt1[0]:0.2f} | R\N{SUPERSCRIPT TWO} = {r2_1:.2f}", color="red", linewidth=3)
plt.plot(t, fittedData2, label=f"Hurst Mod.: C = {popt2[0]:0.2f} ; a = {popt2[1]:0.3f} | R\N{SUPERSCRIPT TWO} = {r2_2:.2f}", color="green", linewidth=3)
plt.plot(t, fittedData3, label=f"Fetkovich: Wei = {popt3[0]:0.2f} ; J = {popt3[1]:0.3f} | R\N{SUPERSCRIPT TWO} = {r2_3:.2f}", color="orange", linewidth=3)
plt.legend(fontsize='small')
plt.xlabel("t")
plt.ylabel("We")
plt.show()

#plt.scatter(p, we, label="Data", color="blue")
#plt.plot(p, fittedData1, label=f"Fit: J = {popt1[0]:0.2f} | R\N{SUPERSCRIPT TWO} = {r2_1:.2f}", color="red", linewidth=3)
#plt.plot(p, fittedData2, label=f"Fit: C = {popt2[0]:0.2f} ; a = {popt2[1]:0.3f} | R\N{SUPERSCRIPT TWO} = {r2_2:.2f}", color="green", linewidth=3)
#plt.plot(p, fittedData3, label=f"Fit: Wei = {popt3[0]:0.2f} ; J = {popt3[1]:0.3f} | R\N{SUPERSCRIPT TWO} = {r2_3:.2f}", color="orange", linewidth=3)
#plt.legend()
#plt.xlabel("p")
#plt.ylabel("We")
#plt.show()

# syntax for 3-D projection
#ax = plt.axes(projection ='3d')
# 
## plotting
#ax.scatter(df["t"], df["p"], df["We"])
#ax.plot3D(df["t"], df["p"], train["We_pred"], 'green')
#ax.set_title('3D line plot')
#plt.show()
#

#Otimização

p_prev=np.zeros(len(t))
we_prev=np.zeros(len(t))
pimt=np.zeros(len(t))

def f_sch(pn, i):
    pmed=(p[i-1]+pn)/2
    pimt[i]=pimt[i-1]+(p[0]-pmed)*(t[i]-t[i-1])
    Sch=popt1[0]*pimt[i]
    return Sch

def f_ebm(pn, i):
    Bo=bob+co*bob*(pb-pn)
    Bg=1.40676*pn**(-1.04229)
    F=npp[i]*(Bo+(rp[i]-Rsi)*Bg)+(wp[i]-winj[i])*Bw
    Eo=Bo-bo[0]
    Efw=bo[0]*((cf+cw*Swi)/(1-Swi))*(p[0]-pn)
    EBM=F-N*(Eo+Efw)
    return EBM

for i in range(len(t)):
    if p[i] == p[0]:
        p_prev[i]=p[0]
        we_prev[i]=0
        pimt[i]=0
    else:
        pn=p[i-1]
        def f_obj(pn):
            Aquif=f_sch(pn, i)
            EBM=f_ebm(pn, i)
            return(EBM-Aquif)
        result=f_obj(pn)
        const = {'type':'eq', 'fun': f_obj}
        result=minimize(f_obj, pn, constraints=const)
        p_prev[i]=result.x[0]
    
plt.scatter(t, p, label="Data", color="blue")
plt.plot(t, p_prev, label="Fit", color="red", linewidth=3)
plt.plot(t, df["Press_b"], label="Benchmark", color="black", linewidth=2, linestyle='--')
plt.legend(fontsize='small')
plt.xlabel("t")
plt.ylabel("p")
plt.show()

train["P_pred1"]=p_prev
train.to_excel('unisim_hist_match.xlsx', index=False)