# -*- coding: utf-8 -*-
"""
Projeto Mestrado Computação Aplicada - IFES
Título: Uso de PINNs associadas a Balanço de Materiais de Reservatórios
Autor: Marcelo Garret de Melo Filho
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import copy
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

df=pd.read_excel('unisim_hist.xlsx')
df=df.drop(["Press_b", "Np_b", "Gp_b", "Wp_b", "Winj_b"], axis=1)
df.head()

print(df.dtypes)

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

df["t"] = (df["Date"]-df["Date"].iloc[0]).astype("int64")/10**9/60/60/24
df["dt"]=df["t"].diff()
df["p"] = df["Press"].iloc[0]-df["Press"]
df["dp"]=df["p"].diff()

print(df.head())

df["Rp"] = df["Gp"]/df["Np"]

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
bo =func_bo(p, 0.0012, 1.1538)
df["Bo"]=bo

bg=func_bg(p, 1.40676, -1.04229)
df["Bg"]=bg

rs=func_rs(p, 0.4655, 15.0114)
df["Rs"]=rs

df["Bt"]=df["Bo"]+(Rsi-df["Rs"])*df["Bg"]
df["F"] = df["Np"]*(df["Bt"]+(df["Rp"]-Rsi)*df["Bg"])+(df["Wp"]-df["Winj"])*Bw
df["Eo"] = df["Bt"]-df["Bt"].iloc[0]
df["Eg"] = df["Bt"].iloc[0]*(df["Bg"]/df["Bg"].iloc[0]-1)
df["We"] = df["F"]-N*(df["Eo"]+m*df["Eg"])
df["x"] = df["We"]/(df["Eo"]+m*df["Eg"])/1E6
df["y"] = df["F"]/(df["Eo"]+m*df["Eg"])/1E6

# df = df.set_index("Date")

df=df.dropna()
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

FEATURES_we = ["t", "p"]
TARGET_we = "We"

X_train_we = train[FEATURES_we]
y_train_we = train[TARGET_we]

reg_we = xgb.XGBRegressor()
reg_we.fit(X_train_we, y_train_we)

print(reg_we.score(X_train_we, y_train_we))
print(reg_we.predict(X_train_we))

df["We_Pred"] = reg_we.predict(X_train_we)

plt.scatter(df["t"], df["We"], label="Hist", color="blue")
plt.plot(df["t"], df["We_Pred"], label="Fit", color="red", linewidth=3)
plt.title("We x t")
plt.legend()
plt.xlabel("dt")
plt.ylabel("We")
plt.show()

plt.scatter(df["p"], df["We"], label="Hist", color="blue")
plt.plot(df["p"], df["We_Pred"], label="Fit", color="red", linewidth=3)
plt.title("We x p")
plt.legend()
plt.xlabel("dp")
plt.ylabel("We")
plt.show()

# syntax for 3-D projection
ax = plt.axes(projection ='3d')
 
# plotting
ax.scatter(df["t"], df["p"], df["We"])
ax.plot3D(df["t"], df["p"], df["We_Pred"], 'green')
ax.set_title('3D line plot')
plt.show()

train=df.copy()

FEATURES_press= ["F", "Eo", "Eg", "We_Pred"]
TARGET_press = "Press"

X_train_press = train[FEATURES_press]
y_train_press = train[TARGET_press]

reg_press = xgb.XGBRegressor()
reg_press.fit(X_train_press, y_train_press)

print(reg_press.score(X_train_press, y_train_press))
print(reg_press.predict(X_train_press))

df["Press_Pred"] = reg_press.predict(X_train_press)

plt.scatter(df["t"], df["Press"], label="Hist", color="blue")
plt.plot(df["t"], df["Press_Pred"], label="Fit", color="red", linewidth=3)
plt.title("Pressure x t")
plt.legend()
plt.xlabel("t")
plt.ylabel("Pressure")
plt.show()

reg_we.save_model('unisim-d_we.json')
reg_press.save_model('unisim-d_press.json')
df.to_excel('unisim_hist_match.xlsx', index=True)
