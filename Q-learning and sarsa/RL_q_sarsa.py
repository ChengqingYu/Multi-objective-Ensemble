import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

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

def sarsa_func():
    #根据参数条件，初始化
    Lt = len(w1)
    # rmse = np.ones((Lt, Lt)) * 100

    # #全局搜索,寻找最优
    # for i in range(Lt):
    #     for j in range(Lt):
    #         if w3[i, j] > 0:
    #             y = w1[i, j] * x1 + w2[i, j] * x2 + w3[i, j] * x3
    #             rmse[i, j] = np.sqrt(np.mean((y - y0) ** 2))

    # r_m = np.min(rmse)
    # fi, fj = np.where(rmse == r_m)
    
    #全局搜索,寻找最优，矩阵化
    rmse = np.ones((Lt * Lt, )) * 100
    w_temp = np.vstack((w1.ravel(), w2.ravel(), w3.ravel()))
    y = np.vstack((x1, x2, x3)).T @  w_temp
    rmse = np.sqrt(np.mean(y-y0.reshape(-1, 1), axis=0)**2)
    r_m_idx = np.argmin(rmse)
    r_m = rmse[r_m_idx]

    #RL算法
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
    Q = np.zeros((ActionNum * MaxX * MaxY, 1)) # 初始化Q值
    Q_s_all = []

    for i in tqdm(range(trials)):
        state = np.random.randint(low=1, high=Lt+1, size=(2,))   # 初始状态在开始点
        action = 0   # 动作初始化
        E = np.zeros_like(Q)
    
        while w3[state[0]-1, state[1]-1] < 0:
            state = np.random.randint(low=1, high=Lt+1, size=(2,))
        
        state_chushi = state
        
        exploring = False
        reward = 0
        Prestate = 0
        for j in range(1, maxiter+1):
            # 计算Q值
            if j > 1:   # 每次探索的第一步不计算Q值
                ix = ndi2lin([1, state[0], state[1]], [ActionNum, MaxX, MaxY])
                qix = ndi2lin([action, Prestate[0], Prestate[1]], [ActionNum, MaxX, MaxY])
                delta = reward + gamma * np.max(Q[ix:ix+ActionNum-1]) - Q[qix]
                E[qix] = 1
                Q = Q + alpha * delta * E
                print(E.shape)
                E = gamma * lambda_ * E * (not exploring)
                Q_s = [Q[qix], state[0], state[1]]
                Q_s_all.append(Q_s)
                
            # 选择动作
            ix = ndi2lin([1, state[0], state[1]], [ActionNum, MaxX, MaxY])   # 当前状态对应第一个动作的Q值第索引
            ix = slice(ix, ix+ActionNum)   # 当前状态对应的所有动作的Q值索引
            topactions = np.argwhere(Q[ix] == np.max(Q[ix])).flatten()   # 最大Q值动作的索引：1-5
            action = np.random.choice(topactions)   # 选择的动作
            
            # 探索策略
            if np.random.rand() < epsilon:
                action = np.random.randint(1, ActionNum+1)   # 进行新的探索
                exploring = True
            else:
                exploring = False
            
            epsilon = epsilon / trials
            
            # 机器人移动并计算reward值
            Prestate = state
            state, reward = MovAction(state, action, actoffsets)
    #         if (state == EP).all():   # 是否到达目标
    #             break
            
        # stats[k] = j # stats 没找到初始化
        stats.append(j)
    
        if k > avgtrials:
            err_con[k].append(np.std(stats[-avgtrials:])) # 根据不同情况修改，共5处之2
            # err_con[k] = np.std(stats[length(stats)-avgtrials:length(stats)])   # 根据不同情况修改，共5处之2
            #err_initi[k] = np.std(stats[length(stats)-avgtrials:length(stats)])
            #err_poe[k] = np.std(stats[length(stats)-avgtrials:length(stats)])
            
        # if k > avgtrials and np.std(stats[length(stats)-avgtrials:length(stats)]) < convgoal and j < 300:
        #     break
        if k > avgtrials and np.std(stats[-avgtrials:]) < convgoal and j < 300:
            break
        
        k = k + 1

    mq, mi = np.max(Q_s_all[:, 0]), np.argmax(Q_s_all[:, 0])
    Max_state = Q_s_all[mi, 1:3]
    yS1 = w1[state[0], state[1]]*x1 + w2[state[0], state[1]]*x2 + w3[state[0], state[1]]*x3
    yS2 = w1[Max_state[0], Max_state[1]]*x1 + w2[Max_state[0], Max_state[1]]*x2 + w3[Max_state[0], Max_state[1]]*x3
    rmse1 = np.sqrt(np.mean(np.square(yS1-y0)))
    rmse2 = np.sqrt(np.mean(np.square(yS2-y0)))
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

    B = W1*x1+W2*x2+W3*x3 
    A = y0
    Net_mae = np.mean(np.abs(A - B))
    Net_mape = np.mean(np.abs((A - B)/A))*100
    errors=B-A
    MSE=np.mean(errors**2)
    RMSE=np.sqrt(MSE)
    ERR= np.mean(A-B)
    C = A-B-ERR
    D = np.mean(C**2)
    SDE = np.sqrt(np.mean(D))
    ## w1+w2+w3=1 0<w<1
    # y=w1*x1+w2*x2+w3*x3;

    # rmse=rms(y-y0);% 均方根误差

def Q_learning_func():
    
    #根据参数条件，初始化
    #根据参数条件，初始化
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
                E = gamma * lambda_ * E * (not exploring)
                Q_s = [Q[qix], state[0], state[1]]
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
            err_con[k].append(np.std(stats[-avgtrials:]) ) # 根据不同情况修改,共5处之2
            # err_initi[k] = np.std(stats[-avgtrials:])
            # err_poe[k] = np.std(stats[-avgtrials:])
        if k > avgtrials and np.std(stats[-avgtrials:]) < convgoal and j < 300:
            break
        k += 1

    mq, mi = np.max(Q_s_all[:, 0]), np.argmax(Q_s_all[:, 0])
    Max_state = Q_s_all[mi, 1:3]
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
    

# Q_learning_func()
sarsa_func()

