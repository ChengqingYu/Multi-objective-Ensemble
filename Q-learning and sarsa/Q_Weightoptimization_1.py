import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from base_func import init_data, ndi2lin, MovAction, y0, x1, x2, x3, w1, w2, w3, MaxX, MaxY


Lt = len(w1)
# rmse = np.ones((Lt, Lt)) * 100

#全局搜索,寻找最优
# for i in range(Lt):
#     for j in range(Lt):
#         if w3[i, j] > 0:
#             y = w1[i, j] * x1 + w2[i, j] * x2 + w3[i, j] * x3
#             rmse[i, j] = np.sqrt(np.mean((y - y0)**2))
# r_m = np.min(rmse)
# fi, fj = np.where(rmse == r_m)

#全局搜索,寻找最优，矩阵化
rmse = np.ones((Lt * Lt, )) * 100
w_temp = np.vstack((w1.ravel(), w2.ravel(), w3.ravel()))
y = np.vstack((x1, x2, x3)).T @  w_temp
rmse = np.sqrt(np.mean(y-y0.reshape(-1, 1), axis=0)**2)
r_m_idx = np.argmin(rmse)
r_m = rmse[r_m_idx]

#RL 算法
MaxX = Lt # X坐标取值
MaxY = Lt # Y坐标取值
ActionNum = 9 # 动作数：up,down,left,right(1 2 3 4 5)
actoffsets = np.array([[-1, 0], [-1, 1], [-1, -1], [1, 0], [1, 1], [1, -1], [0, 1], [0, 0], [0, -1]])
alpha = 0.3
gamma = 0.95
lambda_ = 0.5
epsilon = 0.5 # 学习参数
trials = 100 # 设置最大尝试次数
maxiter = 100 # 每次尝试最大迭代次数
convgoal = 0.25 # 收敛目标
avgtrials = 10 # 收敛时平均迭代次数
k = 1 # 收敛性数组下标初始化
stats = []
err_con = []
#Q值的初始化
Q = np.zeros((ActionNum * MaxX * MaxY, 1))
Q_s_all = []


for i in tqdm(range(trials)):
    action = 0  # 动作初始化
    E = np.zeros((ActionNum * MaxX * MaxY, 1))
    state = np.random.randint(1, Lt + 1, size=(2, ))  # 初始状态在开始点
    while w3[state[0] - 1, state[1] - 1] < 0:
        state = np.random.randint(1, Lt + 1, size=(2, ))
    state_chushi = state
    
    exploring = False
    reward = 0
    Prestate = 0
    for j in (range(1, maxiter+1)):
        # Calculate Q values
        if j > 1:  # The first step of each exploration does not calculate Q values
            ix = ndi2lin([1, state[0], state[1]], [ActionNum, MaxX, MaxY])
            qix = ndi2lin([action, Prestate[0], Prestate[1]], [ActionNum, MaxX, MaxY])
            delta = reward + gamma * max(Q[ix:ix+ActionNum-1]) - Q[qix]
            E[qix] = 1
            Q = Q + alpha * delta * E
            Q_qix_temp = Q[qix][0]
            # print(Q_qix_temp.shape)
            E = gamma * lambda_ * E * (not exploring)
            Q_s = [Q_qix_temp , state[0], state[1]]
            Q_s_all.append(Q_s)
        
        # Choose action
        ix = ndi2lin([1, state[0], state[1]], [ActionNum, MaxX, MaxY])
        ix = slice(ix, ix+ActionNum-1)
        topactions = np.flatnonzero(Q[ix] == max(Q[ix]))  
        # print(topactions, topactions.shape)
        # action = topactions[int(np.ceil( np.random.rand() * len(topactions) ))]
        action = np.random.choice(topactions)
        # Exploration strategy
        if np.random.rand() < epsilon:
            action = np.random.randint(1, ActionNum+1)
            exploring = True
        else:
            exploring = False
        epsilon = epsilon / trials
        
        Prestate = state
        state, reward = MovAction(state, action, actoffsets)

    stats.append(j)
    if k > avgtrials:
        err_con.append(np.std(stats[-avgtrials:]) ) # 根据不同情况修改,共5处之2
        # err_initi[k] = np.std(stats[-avgtrials:])
        # err_poe[k] = np.std(stats[-avgtrials:])
    if k > avgtrials and np.std(stats[-avgtrials:]) < convgoal and j < 300:
        break
    k += 1

Q_s_all_ = np.array(Q_s_all)
mq, mi = np.max(Q_s_all_[:, 0]), np.argmax(Q_s_all_[:, 0])
Max_state = Q_s_all_[mi, 1:3].astype(int)
yS1 = w1[state[0], state[1]] * x1 + w2[state[0], state[1]] * x2 + w3[state[0], state[1]] * x3
yS2 = w1[Max_state[0], Max_state[1]] * x1 + w2[Max_state[0], Max_state[1]] * x2 + w3[Max_state[0], Max_state[1]] * x3
rmse1 = np.sqrt(np.mean((yS1 - y0) ** 2))
rmse2 = np.sqrt(np.mean((yS2 - y0) ** 2))
if rmse1 > rmse2:
    rmse_jieguo = rmse2
    W_jieguo = [w1[Max_state[0], Max_state[1]], w2[Max_state[0], Max_state[1]], w3[Max_state[0], Max_state[1]]]
    W_chushi = [w1[state[0], state[1]], w2[state[0], state[1]], w3[state[0], state[1]]]
    W1 = w1[Max_state[0], Max_state[1]]
    W2 = w2[Max_state[0], Max_state[1]]
    W3 = w3[Max_state[0], Max_state[1]]
else:
    rmse_jieguo = rmse1
    W_jieguo = [w1[state[0], state[1]], w2[state[0], state[1]], w3[state[0], state[1]]]
    W_chushi = [w1[Max_state[0], Max_state[1]], w2[Max_state[0], Max_state[1]], w3[Max_state[0], Max_state[1]]]
    W1 = w1[state[0], state[1]]
    W2 = w2[state[0], state[1]]
    W3 = w3[state[0], state[1]]

B = W1*x1 + W2*x2 + W3*x3 
A = y0
Net_mae = np.mean(np.abs(A - B))
Net_mape = np.mean(np.abs((A - B) / A)) * 100
errors = B - A
MSE = np.mean(errors**2)
RMSE = np.sqrt(MSE)
ERR = np.mean(A - B)
C = A - B - ERR
D = np.mean(C**2)
SDE = np.sqrt(D)
print('验证集的\nRMSE: {},SDE: {},MAPE: {},MAE: {}'.format(RMSE,SDE,Net_mape,Net_mae))