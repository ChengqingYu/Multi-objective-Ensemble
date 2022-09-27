"""
测试NSGA2算法 
"""
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis 
from NSGA2 import * 
from function import * 
from fitness import * 
import math

def MaxMinNormalization(data,Max,Min):
    result = []
    for i in range(len(data)):
        x = (data[i] - Min) / (Max - Min)
        result.append(x)
    return result

def Euclidean_distance(data1,data2,data3,len_data):
    result = []
    for i in range(len_data):
        distance_data = math.sqrt(data1[i] * data1[i] + data2[i] * data2[i] + data3[i] * data3[i])
        result.append(distance_data)
    return result

def main(): 
    nIter = 200
    nChr = 4
    nPop = 100 
    pc = 0.6  
    pm = 0.1 
    etaC = 1 
    etaM = 1 
    func = function 
    lb = 0
    rb = 1
    patuojie, patuojie_shiyingdu = NSGA2(nIter, nChr, nPop, pc, pm, etaC, etaM, func, lb, rb)
    #print(patuojie_shiyingdu)
    print(f"paretoFront: {patuojie_shiyingdu.shape}")

    # 理论最优解集合 
    x = np.linspace(0,1, 100).reshape(100,1)
    X = np.concatenate((x,x,x,x), axis=1)
    thFits = fitness(X, function) 

    plt.rcParams['font.sans-serif'] = 'KaiTi'  # 设置显示中文 
    fig = plt.figure(dpi=400) 
    ax = fig.add_subplot(111) 
    ax.plot(thFits[:,0], thFits[:,1], color='green', label='理论帕累托前沿') 
    ax.scatter(patuojie_shiyingdu[:,0], patuojie_shiyingdu[:,1], color='red', label='实际解集')
    ax.legend() 
    fig.savefig('test.png', dpi=400) 

    #print(patuojie)

    max_data1 = np.max(patuojie_shiyingdu[:, 0])
    min_data1 = np.min(patuojie_shiyingdu[:, 0])

    max_data2 = np.max(patuojie_shiyingdu[:, 1])
    min_data2 = np.min(patuojie_shiyingdu[:, 1])

    max_data3 = np.max(patuojie_shiyingdu[:, 2])
    min_data3 = np.min(patuojie_shiyingdu[:, 2])

    index_1 = MaxMinNormalization(patuojie_shiyingdu[:, 0], max_data1, min_data1)
    index_2 = MaxMinNormalization(patuojie_shiyingdu[:, 1], max_data2, min_data2)
    index_3 = MaxMinNormalization(patuojie_shiyingdu[:, 2], max_data3, min_data3)
    pareto_len = patuojie_shiyingdu.shape[0]

    Euclidean_distance_result = Euclidean_distance(index_1, index_2, index_3, pareto_len)
    index_result = Euclidean_distance_result.index(min(Euclidean_distance_result))

    data = np.array(pd.read_excel('wind.xlsx'))
    pre_result = data[:, 1:]
    target = data[:, 0]
    x1, x2, x3, x4 = pre_result[:, 5], pre_result[:, 6], pre_result[:, 7], pre_result[:, 12]

    para = patuojie[index_result]
    b1, b2, b3, b4 = para[0], para[1], para[2], para[3]
    pred = x1 * b1 + x2 * b2 + x3 * b3 + x4 * b4
    R2 = r2_score(target, pred)
    RMSE = np.sqrt(mean_squared_error(target, pred))  # 均方根误差
    MAPE = mean_absolute_percentage_error(target, pred)  # 平均绝对误差百分比
    MAE = mean_absolute_error(target, pred)
    print('验证集的\nR2: {},RMSE: {},MAPE: {},MAE: {}'.format(R2, RMSE, MAPE, MAE))

    output = np.stack((target.ravel(), pred.ravel()))
    finall_output = output.T
    m = pd.DataFrame(finall_output)
    m.to_csv('result_wind.csv')

if __name__ == "__main__": 
    main()
#%%

#%%
