"""
随机初始化种群 
"""
import numpy as np 

def initPops(nPop, nChr, lb, rb, Vmax, Vmin):
    pops = np.random.rand(nPop, nChr)*(rb-lb) + lb 
    VPops = np.random.rand(nPop, nChr)*(Vmax-Vmin) + Vmin
    return pops, VPops
 # """多目标粒子群算法
 #    Params:
 #        nIter: 迭代次数
 #        nPOp: 粒子群规模
 #        nAr: archive集合的最大规模
 #        nChr: 粒子大小
 #        func: 优化的函数
 #        c1、c2: 速度更新参数
 #        lb: 解下界
 #        rb：解上界
 #        Vmax: 速度最大值
 #        Vmin：速度最小值
 #        M: 划分的栅格的个数为M*M个
 #    Return:
 #        paretoPops: 帕累托解集
 #        paretoPops：对应的适应度
 #    """