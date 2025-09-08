import numpy as np
import random
import copy
import math
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
import threading

class PSO_Algorithm:

    def __init__(self, w, c1, c2, var_max, var_min, dim, pool, thread, r):
        self.w = w
        self.c1 = c1 
        self.c2 = c2
        #位置極值 list
        self.var_max = var_max 
        self.var_min = var_min
        self.dim = dim #維度
        self.P_list = [] #粒子清單
        self.count = 0 #粒子數量
        self.gfit = float('inf') #最佳適應值
        self.gbest = np.zeros(2) #最佳座標位置
        self.pool = mp.Pool(pool) #進程數量
        self.thread = thread #線城數量
        self.r = r
    
    #創建粒子
    def createParticals(self, count):
        self.count = count
        for _ in range(self.count):
            self.P_list.append(Partical(self.var_max, self.var_min, self.dim, self.w, self.c1, self.c2, self.r))
        self.calgbest()

    #尋找全域最佳值
    def calgbest(self):
        all_pfit = []
        all_pbest = []
        for p in self.P_list:
            all_pfit.append(p.fit)
            all_pbest.append(p.pbest)
        all_pfit = np.array(all_pfit)
        arg = np.argmin(all_pfit)
        self.gbest = all_pbest[arg]
        for p1 in self.P_list:
            p1.gbest = self.gbest

    #搜尋
    def search(self):
        for i in self.P_list:
            i.updatev()
        self.calgbest()
        p_group = []
        group = []
        for p in self.P_list:
            if len(group) == self.thread:
                p_group.append(group)
            group.append(p)
        result = self.pool.map(self.job, p_group)
        full_result = []
        for g in result:
            full_result += g
        for k in range(self.count):
            self.P_list[k].updateall(full_result[k][0], full_result[k][1])
    
    @staticmethod
    def job(partical_list):
        thread = []
        result = []
        for p in partical_list:
            a = threading.Thread(target=p.serching)
            a.start()
            thread.append(a)

        for p1 in thread:
            p1.join()
        for  i in partical_list:
            result.append([i.pbest, i.fit])
        return result        

    #獲取最佳值
    def getBestParameter(self):
        self.calgbest()
        return self.gbest



class Partical:

    def __init__(self, var_max, var_min, dim, w, c1, c2, r):
        self.r = r 
        self.dim = dim #維度
        #位置極值
        self.var_max = np.array(var_max)
        self.var_min = np.array(var_min)
        #座標
        self.x = np.random.uniform(self.var_min, self.var_max,self.dim)
        #向量速度
        self.v = np.random.uniform(size=self.dim) * r
        #個體最佳值
        self.pbest = self.x.copy()
        #個體最佳適應值
        self.fit = float('inf')
        self.fit = self.function()
        #全域最佳值
        self.gbest = np.zeros(2)
        self.w = w
        self.c1 = c1
        self.c2 = c2       

    #更新速度
    def updatev(self):
        g1 = self.pbest - self.x
        g2 = self.gbest - self.x
        g1 = g1 / (np.sum((g1 ** 2)) ** 0.5) if np.sum((g1 ** 2)) ** 0.5 != 0 else g1
        g2 = g2 / (np.sum((g2 ** 2)) ** 0.5) if np.sum((g2 ** 2)) ** 0.5 != 0 else g2
        self.v = self.w * self.v + self.c1 * random.random() * g1 + self.c2 * random.random() * g2
        self.v = self.v / (np.sum((self.v ** 2) ** 0.5))
        self.v = self.v * self.r
        self.x = self.x + self.v
        self.x = np.clip(self.x, self.var_min, self.var_max)

    def serching(self):
        self.updatev()
        fit = self.function()
        if fit < self.fit:
            self.fit = fit
            self.pbest = self.x.copy()
    
    #更新參數
    def updateall(self, pbest, fit):
        self.pbest = pbest
        self.fit = fit

    #適應值函數
    def function(self):
        x1 = self.x[0]
        x2 = self.x[1]
        x3 = x1 ** 2 + (x2 - 0.5) ** 2
        if self.fit > x3:
            self.fit = x3
            self.pbest = self.x
        return x3

def PSO_test_run():
    """
    Please make your pso algorithm can be excute by following scripts.
    Feel free to add arguments when initializing algorithm if it is needed.
    	For example: PSO_Algorithm(dimention_of_variable, variable_range)
    """
    w = 0.5
    c1 = 2
    c2 = 2
    var_max = [10, 10]
    var_min = [-10, -10]
    dim = 2
    pool = mp.cpu_count()
    thread = 4
    r = 1
    total_epoch = 1000
    pso = PSO_Algorithm(w, c1, c2, var_max, var_min, dim, pool, thread, r)

    pso.createParticals(10)
    for epoch in range(total_epoch):
        pso.search()

    print(pso.getBestParameter())

if __name__ == "__main__":
    PSO_test_run()