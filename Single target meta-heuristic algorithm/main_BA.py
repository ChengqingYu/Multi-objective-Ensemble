# =========导入相关库===============
import numpy as np
from numpy.random import random as rand
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error

# ========参数设置==============
# objfun:目标函数
# N_pop: 种群规模，通常为10到40
# N_gen: 迭代数
# A: 响度（恒定或降低）
# r: 脉冲率（恒定或减小）
# 此频率范围决定范围
# 如有必要，应更改这些值
# Qmin: 频率最小值
# Qmax: 频率最大值
# d: 维度
# lower: 下界
# upper: 上界

def bat_algorithm(objfun, N_pop=20, N_gen=1000, A=0.5, r=0.5,
                  Qmin=0, Qmax=2, d=4, lower=0, upper=1):
    N_iter = 0  # Total number of function evaluations

    # =====速度上下限================
    Lower_bound = lower * np.ones((1, d))
    Upper_bound = upper * np.ones((1, d))

    Q = np.zeros((N_pop, 1))  # 频率
    v = np.zeros((N_pop, d))  # 速度
    S = np.zeros((N_pop, d))

    # =====初始化种群、初始解=======
    # Sol = np.random.uniform(Lower_bound, Upper_bound, (N_pop, d))
    # Fitness = objfun(Sol)
    Sol = np.zeros((N_pop, d))
    Fitness = np.zeros((N_pop, 1))
    for i in range(N_pop):
        Sol[i] = np.random.uniform(Lower_bound, Upper_bound, (1, d))
        Fitness[i] = objfun(Sol[i])

    # ====找出初始最优解===========
    fmin = min(Fitness)
    Index = list(Fitness).index(fmin)
    best = Sol[Index]

    # ======开始迭代=======
    for t in range(N_gen):

        # ====对所有蝙蝠/解决方案进行循环 ======
        for i in range(N_pop):
            # Q[i] = Qmin + (Qmin - Qmax) * np.random.rand
            Q[i] = np.random.uniform(Qmin, Qmax)
            v[i] = v[i] + (Sol[i] - best) * Q[i]
            S[i] = Sol[i] + v[i]

            # ===应用简单的界限/限制====
            Sol[i] = simplebounds(Sol[i], Lower_bound, Upper_bound)
            # Pulse rate
            if rand() > r:
                # The factor 0.001 limits the step sizes of random walks
                S[i] = best + 0.001 * np.random.randn(1, d)

            # ====评估新的解决方案 ===========
            # print(i)
            Fnew = objfun(S[i])
            # ====如果解决方案有所改进，或者声音不太大，请更新====
            if (Fnew <= Fitness[i]) and (rand() < A):
                Sol[i] = S[i]
                Fitness[i] = Fnew

            # ====更新当前的最佳解决方案======
            if Fnew <= fmin:
                best = S[i]
                fmin = Fnew

        N_iter = N_iter + N_pop

    print('Number of evaluations: ', N_iter)
    print("Best = ", best, '\n fmin = ', fmin)

    return best


def simplebounds(s, Lower_bound, Upper_bound):
    Index = s > Lower_bound
    s = Index * s + ~Index * Lower_bound
    Index = s < Upper_bound
    s = Index * s + ~Index * Upper_bound

    return s


# ====目标函数=============
def fitness_func(u):
    a = u[0] * x1 + u[1] * x2 + u[2] * x3 + u[3] * x4
    rmse_value = np.sqrt(mean_squared_error(target, a))
    return rmse_value


if __name__ == '__main__':
    # print(bat_algorithm(test_function))
    data = np.array(pd.read_excel('ETTH.xlsx'))
    pre_result = data[:, 1:]
    target = data[:, 0]
    x1, x2, x3, x4 = pre_result[:, 5], pre_result[:, 6], pre_result[:, 8], pre_result[:, 12]
    Best = bat_algorithm(fitness_func)

    b1, b2, b3, b4 = Best[0], Best[1], Best[2], Best[3]
    pred = x1 * b1 + x2 * b2 + x3 * b3 + x4 * b4
    R2 = r2_score(target, pred)
    RMSE = np.sqrt(mean_squared_error(target, pred))  # 均方根误差
    MAPE = mean_absolute_percentage_error(target, pred)  # 平均绝对误差百分比
    MAE = mean_absolute_error(target, pred)
    print('验证集的\nR2: {},RMSE: {},MAPE: {},MAE: {}'.format(R2, RMSE, MAPE, MAE))
    output = np.stack((target.ravel(), pred.ravel()))
    finall_output = output.T
    m = pd.DataFrame(finall_output)
    m.to_csv('result_ETTH.csv')
