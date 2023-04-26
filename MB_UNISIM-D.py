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
df=df.drop(["Press_b", "Np_b", "Gp_b", "Wp_b", "Winj_b"], axis=1)
# print(df.head())

# print(df.dtypes)

# Parametros escalares (MODSI)
phi = 0.136
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
cw = 48E-6
Swi = 0.17

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
df["Efw"] = df["Bt"].iloc[0]*((cf+cw*Swi)/(1-Swi))*df["dp"]
df["We"] = df["F"]-N*(df["Eo"]+m*df["Eg"]+(1+m)*df["Efw"])
df["x"] = df["We"]/(df["Eo"]+m*df["Eg"]+(1+m)*df["Efw"])/1E6
df["y"] = df["F"]/(df["Eo"]+m*df["Eg"]+(1+m)*df["Efw"])/1E6

# df = df.set_index("Date")
# df=df.dropna()

print(df.head())

plt.scatter(df["x"], df["y"])
plt.title("EBM linear")
plt.xlabel("We/(Eo+mEg)")
plt.ylabel("F/(Eo+mEg)")
plt.show()

plt.scatter(df["t"], df["We"])
plt.title("We x t")
plt.xlabel("t")
plt.ylabel("We")
plt.show()

plt.scatter(df["p"], df["We"])
plt.title("We x p")
plt.xlabel("p")
plt.ylabel("We")
plt.show()

# syntax for 3-D projection
ax = plt.axes(projection ='3d')
 
# plotting
ax.scatter(df["t"], df["p"], df["We"])
# ax.plot3D(x, y, z, 'green')
ax.set_title('3D line plot')
plt.show()

train = df.copy()
train = train.drop(["Np", "Gp", "Rp", "Wp", "Winj", "Bt", "Bo", "Bg", "Rs", "F", "Eo", "Eg", "Efw", "x", "y", "p", "dt"], axis=1)

p=train["Press"].values
we=train["We"].values
t=train["t"].values
we[0]=0
we[1]=0

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

print(train.head())

# initialGuess=[1]
initialGuess=[1,1]
# popt,pcov = curve_fit(func_we1, t, we, initialGuess)
popt,pcov = curve_fit(func_we2, t, we, initialGuess)
print(popt)

# fittedData=func_we1(t, *popt)
fittedData=func_we2(t, *popt)
r2 = r2_score(we, fittedData)
print(r2)

train["We_pred"]=fittedData

plt.scatter(t, we, label="Data", color="blue")
# plt.plot(t, fittedData, label=f"Fit: J = {popt[0]:0.2f} | R\N{SUPERSCRIPT TWO} = {r2:.2f}", color="red", linewidth=3)
plt.plot(t, fittedData, label=f"Fit: C = {popt[0]:0.2f} ; a = {popt[1]:0.3f} | R\N{SUPERSCRIPT TWO} = {r2:.2f}", color="red", linewidth=3)
plt.legend()
plt.xlabel("t")
plt.ylabel("We")
plt.show()

plt.scatter(p, we, label="Data", color="blue")
# plt.plot(p, fittedData, label=f"Fit: J = {popt[0]:0.2f} | R\N{SUPERSCRIPT TWO} = {r2:.2f}", color="red", linewidth=3)
plt.plot(p, fittedData, label=f"Fit: C = {popt[0]:0.2f} ; a = {popt[1]:0.3f} | R\N{SUPERSCRIPT TWO} = {r2:.2f}", color="red", linewidth=3)
plt.legend()
plt.xlabel("p")
plt.ylabel("We")
plt.show()

# syntax for 3-D projection
ax = plt.axes(projection ='3d')
 
# plotting
ax.scatter(df["t"], df["p"], df["We"])
ax.plot3D(df["t"], df["p"], train["We_pred"], 'green')
ax.set_title('3D line plot')
plt.show()

train.to_excel('unisim_hist_match.xlsx', index=True)
