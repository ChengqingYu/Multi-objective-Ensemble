import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error

#获取原始数据
data=np.array(pd.read_excel('ETTH.xlsx'))
pre_result = data[:,1:]
target = data[:,0]
x1,x2,x3,x4=pre_result[:,5],pre_result[:,6],pre_result[:,8],pre_result[:,12]

#设置权重初始值
# b1,b2,b3,b4=0.549137,0.06559116,0.17129299,0.22646807
b1,b2,b3,b4=0.25,0.25,0.25,0.25
# b1=np.arange(0,1,0.1)
# b2=np.arange(0,1,0.1)
# b3=np.arange(0,1,0.1)
# b4=np.arange(0,1,0.1)

#设置目标优化对象为RMSE
#先利用粒子群算法PSO
def mubiao1(b1,b2,b3,b4):
    pred_value=x1*b1+x2*b2+x3*b3+x4*b4
    rmse_value=np.sqrt(mean_squared_error(target, pred_value))
    return rmse_value
def mubiao2(b1,b2,b3,b4):
    pred_value = x1*b1+x2*b2+x3*b3+x4*b4
    mae_value = mean_absolute_error(target, pred_value)
    return mae_value

# 3.初始化参数
W = 0.6                                 # 惯性因子
c1 = 1.5                                # 学习因子
c2 = 1.5                                # 学习因子
n_iterations =200                      # 迭代次数
n_particles = 100                        # 种群规模
def fitness_function(position):
    score1 = mubiao1(b1=position[0],b2=position[1],b3=position[2],b4=position[3])                                                           #R2
    score2 = mubiao2(b1=position[0], b2=position[1], b3=position[2], b4=position[3])
    return np.array([score1,score2])

# 初始化粒子位置，进行迭代
particle_position_vector = np.array([np.array([abs(0.01+random.random()) , abs(0.01+random.random()), abs(0.01+random.random()), abs(0.01+random.random())]) for _ in range(n_particles)])
pbest_position = particle_position_vector#50个1到10向量
pbest_fitness_value = np.array([float('inf') for _ in range(n_particles)])#n_particles种群规模,返回50个inf
gbest_fitness_value = np.array([float('inf'), float('inf'), float('inf'), float('inf')])#返回1个inf向量
gbest_position = np.array([float('inf'), float('inf'), float('inf'), float('inf')])#返回1个inf向量
velocity_vector = ([np.array([0,0,0,0]) for _ in range(n_particles)])#返回50个0向量
iteration = 0

#小于迭代次数
while iteration < n_iterations:
    for i in range(n_particles):
        fitness_cadidate = fitness_function(particle_position_vector[i])#返回RMSE_value
        print("error of particle-", i, "is (rmse, mae)", fitness_cadidate, " At (b1,b2,b3,b4): ",
              particle_position_vector[i])#打印此时的模型参数

        if (pbest_fitness_value[i] >fitness_cadidate[1]):
            pbest_fitness_value[i] = fitness_cadidate[1]
            pbest_position[i] = particle_position_vector[i]

        if (gbest_fitness_value[1] > fitness_cadidate[1]):
            gbest_fitness_value = fitness_cadidate
            gbest_position = particle_position_vector[i]

        elif (gbest_fitness_value[1] == fitness_cadidate[1] and gbest_fitness_value[0] >fitness_cadidate[0]):
            gbest_fitness_value = fitness_cadidate
            gbest_position = particle_position_vector[i]

    for i in range(n_particles):
        new_velocity = (W * velocity_vector[i]) + (c1 * random.random()) * (
                    pbest_position[i] - particle_position_vector[i]) + (c2 * random.random()) * (
                                   gbest_position - particle_position_vector[i])
        new_position = new_velocity + particle_position_vector[i]
        particle_position_vector[i] = new_position

    iteration = iteration + 1
print("The best position is ", gbest_position, "in iteration number", iteration, "with error (rmse, mae):",
      fitness_function(gbest_position))


pred=x1*gbest_position[0]+x2*gbest_position[1]+x3*gbest_position[2]+x4*gbest_position[3]
R2=r2_score(target, pred)
RMSE=np.sqrt(mean_squared_error(target, pred))#均方根误差
MAPE=mean_absolute_percentage_error(target, pred)#平均绝对误差百分比
MAE=mean_absolute_error(target, pred)
print('验证集的\nR2: {},RMSE: {},MAPE: {},MAE: {}'.format(R2,RMSE,MAPE,MAE))

output = np.stack((target.ravel(),pred.ravel()))
finall_output = output.T
m=pd.DataFrame(finall_output)
m.to_csv('result_ETTH.csv')