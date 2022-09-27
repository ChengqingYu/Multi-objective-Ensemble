import numpy as np
import random
import pandas as pd
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error

data=np.array(pd.read_excel('ETTH.xlsx'))
pre_result = data[:,1:]
target = data[:,0]
x1,x2,x3,x4=pre_result[:,5],pre_result[:,6],pre_result[:,8],pre_result[:,12]

'''优化函数'''
# y = x^2, 用户可以自己定义其他函数
def fun(x):
    b1,b2,b3,b4=x[0],x[1],x[2],x[3]
    pred_value=x1*b1+x2*b2+x3*b3+x4*b4
    rmse_value=np.sqrt(mean_squared_error(target,pred_value))
    return rmse_value

''' 种群初始化函数 '''
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]

    return X, lb, ub

'''边界检查函数'''
def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X

'''计算适应度函数'''
def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness

'''适应度排序'''
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index

'''根据适应度对位置进行排序'''
def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew

'''灰狼算法'''
def GWO(pop, dim, lb, ub, MaxIter, fun):
    Alpha_pos = np.zeros([1, dim])
    Alpha_score = float("inf")
    Beta_pos = np.ones([1, dim])
    Beta_score = float("inf")
    Delta_pos = np.ones([1, dim])
    Delta_score = float("inf")

    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = fitness[0]
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = X[0, :]
    Curve = np.zeros([MaxIter, 1])
    for t in range(MaxIter):

        for i in range(pop):
            fitValue = fun(X[i, :])
            if fitValue < Alpha_score:
                Alpha_score = fitValue
                Alpha_pos[0, :] = X[i, :]

            if fitValue > Alpha_score and fitValue < Beta_score:
                Beta_score = fitValue
                Beta_pos[0, :] = X[i, :]

            if fitValue > Alpha_score and fitValue > Beta_score and fitValue < Delta_score:
                Delta_score = fitValue
                Delta_pos[0, :] = X[i, :]

        a = 2 - t * (2 / MaxIter)
        for i in range(pop):
            for j in range(dim):
                r1 = random.random()
                r2 = random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_alpha = np.abs(C1 * Alpha_pos[0, j] - X[i, j])
                X1 = Alpha_pos[0, j] - A1 * D_alpha

                r1 = random.random()
                r2 = random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = np.abs(C2 * Beta_pos[0, j] - X[i, j])
                X2 = Beta_pos[0, j] - A2 * D_beta

                r1 = random.random()
                r2 = random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_beta = np.abs(C3 * Delta_pos[0, j] - X[i, j])
                X3 = Delta_pos[0, j] - A3 * D_beta

                X[i, j] = (X1 + X2 + X3) / 3

        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if fitness[0] <= GbestScore:  # 更新全局最优
            GbestScore = fitness[0]
            GbestPositon[0, :] = X[0, :]
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve

'''主函数 '''
# 设置参数
pop = 1000  # 种群数量
MaxIter = 200  # 最大迭代次数
dim = 4  # 维度
lb = 0.01 * np.ones([dim, 1])  # 下边界
ub = 1 * np.ones([dim, 1])  # 上边界

GbestScore, GbestPositon, Curve = GWO(pop, dim, lb, ub, MaxIter, fun)
print('最优适应度值：', GbestScore)
print('最优解：', GbestPositon)

# 绘制适应度曲线
plt.figure(1)
plt.plot(Curve, 'r-', linewidth=2)
plt.xlabel('Iteration', fontsize='medium')
plt.ylabel("Fitness", fontsize='medium')
plt.grid()
plt.title('GWO', fontsize='large')

# 绘制搜索空间
fig = plt.figure(2)
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
Z = (X) ** 2 + (Y) ** 2
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()

###基于优化的权重的得到集成结果
b1,b2,b3,b4=GbestPositon[0][0],GbestPositon[0][1],GbestPositon[0][2],GbestPositon[0][3]
pred=x1*b1+x2*b2+x3*b3+x4*b4
R2=r2_score(target,pred)
RMSE=np.sqrt(mean_squared_error(target,pred))#均方根误差
MAPE=mean_absolute_percentage_error(target,pred)#平均绝对误差百分比
MAE=mean_absolute_error(target,pred)
print('验证集的\nR2: {},RMSE: {},MAPE: {},MAE: {}'.format(R2,RMSE,MAPE,MAE))


output = np.stack((target.ravel(),pred.ravel()))
finall_output = output.T
m=pd.DataFrame(finall_output)
m.to_csv('result_ETTH.csv')