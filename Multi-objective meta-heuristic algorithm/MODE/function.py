"""
优化的目标函数，返回多个目标函数值 
"""
# import numpy as np
#
# def function(X):
#     y1 = 1 - np.exp(-np.sum((X-1/np.sqrt(3))**2))
#     y2 = 1 - np.exp(-np.sum((X+1/np.sqrt(3))**2))
#     return y1, y2
#
# if __name__ == "__main__":
#     tX = np.array([-0.57735, -0.57735, -0.57735])
#     print(function(tX))
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
data=np.array(pd.read_excel('ETTH.xlsx'))
ztz=data[:,1:]
ztr=data[:,0]
x1,x2,x3,x4=ztz[:,5],ztz[:,6],ztz[:,8],ztz[:,12]

def function(X):
    pred = x1 * X[0] + x2 * X[1] + x3 * X[2] + x4 * X[3]
    RMSE = np.sqrt(mean_squared_error(ztr, pred))  # 均方根误差
    MAPE = mean_absolute_percentage_error(ztr, pred)  # 平均绝对误差百分比
    MAE = mean_absolute_error(ztr, pred)
    # y1 = 1 - np.exp(-np.sum((X-1/np.sqrt(3))**2))
    # y2 = 1 - np.exp(-np.sum((X+1/np.sqrt(3))**2))
    # return y1, y2
    return RMSE, MAPE, MAE

if __name__ == "__main__":
    # tX = np.array([-0.57735, -0.57735, -0.57735])
    tX = np.array([0.25, 0.25, 0.25, 0.25])
    print(function(tX))