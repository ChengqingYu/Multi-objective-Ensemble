import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd


def init_data():
    #导入数据
    data = pd.read_excel('数据.xls')
    y0 = data.iloc[:, 0].values
    x1 = data.iloc[:, 1].values
    x2 = data.iloc[:, 2].values
    x3 = data.iloc[:, 3].values
    
    dt = np.arange(0, 1.001, 0.001)
    w1, w2 = np.meshgrid(dt, dt)
    w3 = 1 - w1 - w2
    
    Lt = len(w1)
    MaxX = Lt # X坐标取值
    MaxY = Lt # Y坐标取值
    
    return y0, x1, x2, x3, w1, w2, w3, MaxX, MaxY

y0, x1, x2, x3, w1, w2, w3, MaxX, MaxY = init_data()
kr = 0

def ndi2lin(a, b):
    x = a
    y = b
    # qindex = (x[0]-1)*y[1]*y[2] + (x[1]-1)*y[2] + x[2] # 17-04-13-10:00
    qindex = ((x[1]-1)*y[1]+x[2]-1)*y[0]+a[0] # 17-04-13-16:10
    return qindex



# def MovAction(state, action, actoffsets, x1, x2, x3, y0, w1, w2, w3, MaxX, MaxY, kr):
def MovAction(state, action, actoffsets):
    global kr
    # global x1, x2, x3, y0, w1, w2, w3, MaxX, MaxY, kr
    # 更新机器人位置并保存
    Prestate = state
    y = w1[Prestate[0], Prestate[1]] * x1 + w2[Prestate[0], Prestate[1]] * x2 + w3[Prestate[0], Prestate[1]] * x3
    rmse_old = mean_squared_error(y, y0, squared=False)  # 均方根误差
    state = state + actoffsets[action, :]
    if state[0] < 1:
        state[0] = 1
    if state[1] < 1:    
        state[1] = 1
    if state[0] > MaxX:
        state[0] = MaxX
    if state[1] > MaxY:
        state[1] = MaxY
    y = w1[state[0], state[1]] * x1 + w2[state[0], state[1]] * x2 + w3[state[0], state[1]] * x3
    rmse_new = mean_squared_error(y, y0, squared=False)  # 均方根误差
    if rmse_new < rmse_old:
        reward = 1 + abs(rmse_new - rmse_old)
        state = Prestate
        kr = kr + 1
    else:
        reward = -0.1 - abs(rmse_new - rmse_old)
    return state, reward