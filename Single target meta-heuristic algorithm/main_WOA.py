import numpy as np
import random
import pandas as pd
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error


class woa():
    # 初始化
    def __init__(self, X_train, Y_train, LB=np.array([0, 0, 0, 0]), \
                 UB=np.array([1, 1, 1, 1]), dim=4, b=1, whale_num=20, max_iter=500):
        self.X_train = X_train
        self.Y_train = Y_train
        self.LB = LB
        self.UB = UB
        self.dim = dim
        self.whale_num = whale_num
        self.max_iter = max_iter
        self.b = b
        # Initialize the locations of whale
        self.X = np.random.uniform(0, 1, (whale_num, dim)) * (UB - LB) + LB
        self.gBest_score = np.inf
        self.gBest_curve = np.zeros(max_iter)
        self.gBest_X = np.zeros(dim)

        # 适应度函数

    def fitFunc(self, input):
        a = input[0];
        b = input[1];
        c = input[2];
        d = input[3]
        Y_Hat = a * self.X_train[:,0] + b * self.X_train[:,1] + c * self.X_train[:,2] + d * self.X_train[:,3]
        rmse_value = np.sqrt(mean_squared_error(self.Y_train, Y_Hat))
        return rmse_value

        # 优化模块

    def opt(self):
        t = 0
        while t < self.max_iter:
            for i in range(self.whale_num):
                self.X[i, :] = np.clip(self.X[i, :], self.LB, self.UB)  # Check boundries
                fitness = self.fitFunc(self.X[i, :])
                # Update the gBest_score and gBest_X
                if fitness < self.gBest_score:
                    self.gBest_score = fitness
                    self.gBest_X = self.X[i, :].copy()

            a = 2 * (self.max_iter - t) / self.max_iter
            # Update the location of whales
            for i in range(self.whale_num):
                p = np.random.uniform()
                R1 = np.random.uniform()
                R2 = np.random.uniform()
                A = 2 * a * R1 - a
                C = 2 * R2
                l = 2 * np.random.uniform() - 1

                if p >= 0.5:
                    D = abs(self.gBest_X - self.X[i, :])
                    self.X[i, :] = D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + self.gBest_X
                else:
                    if abs(A) < 1:
                        D = abs(C * self.gBest_X - self.X[i, :])
                        self.X[i, :] = self.gBest_X - A * D
                    else:
                        rand_index = np.random.randint(low=0, high=self.whale_num)
                        X_rand = self.X[rand_index, :]
                        D = abs(C * X_rand - self.X[i, :])
                        self.X[i, :] = X_rand - A * D

            self.gBest_curve[t] = self.gBest_score
            if (t % 100 == 0):
                print('At iteration: ' + str(t))
            t += 1
        return self.gBest_curve, self.gBest_X

data=np.array(pd.read_excel('ETTH.xlsx'))
pre_result = data[:,[6,7,9,13]]
target = data[:,0]
x1,x2,x3,x4=pre_result[:,0],pre_result[:,1],pre_result[:,2],pre_result[:,3]


fitnessCurve, para = woa(pre_result, target, dim=4, whale_num=100, max_iter=1000).opt()

pred = x1 * para[0] + x2 * para[1] + x3 * para[2] +x4 * para[3]
R2=r2_score(target,pred)
RMSE=np.sqrt(mean_squared_error(target,pred))#均方根误差
MAPE=mean_absolute_percentage_error(target,pred)#平均绝对误差百分比
MAE=mean_absolute_error(target,pred)
print('验证集的\nR2: {},RMSE: {},MAPE: {},MAE: {}'.format(R2,RMSE,MAPE,MAE))
print(para)

plt.figure()
plt.plot(fitnessCurve, linewidth='0.5')
plt.show()

output = np.stack((target.ravel(),pred.ravel()))
finall_output = output.T
m=pd.DataFrame(finall_output)
m.to_csv('result_ETTH.csv')