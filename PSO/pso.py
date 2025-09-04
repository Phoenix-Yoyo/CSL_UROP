import numpy as np
import random
import copy
import math
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
import threading

class PSO_Algorithm:

    def __init__(self, w, c1, c2, var_max, var_min, dim):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.var_max = var_max
        self.var_min = var_min
        self.dim = dim
        self.P_list = []
        self.count = 0
        self.gfit = float('inf')
        self.gbest = []
    def createParticals(self, count):
        self.count = count
        for r in range(count):
            self.P_list.append(Partical(self.var_max, self.var_min, self.dim))
    def search(self):
        for g in range(self.count):
            fit = self.P_list[g].funtion()
            if self.gfit > fit:
                self.gfit = fit
                self.gbest = copy.deepcopy(self.P_list[g].x)
            self.P_list[g].update(self.w, self.c1, self.c2, self.gbest)
    def getBestParameter(self):
        return self.gbest



class Partical:

    def __init__(self, var_max, var_min, dim):
        self.dim = dim
        self.var_max = var_max
        self.var_min = var_min
        self.v_max = 0.2 * (var_max - var_min)
        self.v_min = -self.v_max
        self.x = [random.uniform(var_min, var_max) for _ in range(self.dim)]
        self.v = [random.uniform(self.v_min, self.v_max) for _ in range(self.dim)]
        self.pbest = self.x
        self.fit = float('inf')
    def update(self, w, c1, c2, gbest):
        for i in range(self.dim):
            self.v[i] = w * self.v[i] + c1 * random.random() * (self.pbest[i] - self.x[i]) + c2 * random.random() * (gbest[i] - self.x[i])
            if self.v[i] > self.v_max:
                self.v[i] = self.v_max
            elif self.v[i] < self.v_min:
                self.v[i] = self.v_min
            self.x[i] = self.x[i] + self.v[i]
            if self.x[i] > self.var_max:
                self.x[i] = self.var_max
            elif self.x[i] < self.var_min:
                self.x[i] = self.var_min
    def funtion(self):
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
    w = 0.7
    c1 = 2
    c2 = 2
    var_max = 10
    var_min = -10
    dim = 2
    pso = PSO_Algorithm(w, c1, c2, var_max, var_min, dim)
    pso.createParticals(10)

    total_epoch = 1000
    for epoch in range(total_epoch):
        pso.search()

    print(pso.getBestParameter())

if __name__ == "__main__":
    PSO_test_run()