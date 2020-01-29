#!/usr/bin/env python
# -*- coding:utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

def ratelist(filename):
    data = ""
    with open("./"+filename, "r") as f:
        data = f.read()
    data = data.split("\n")[:-1]
    data = map(lambda x: x.split(" ")[8], data)
    data = list(map(lambda x: 1-float(x), data))
    return data

def get_mean(filename_syntax):
    data = [ratelist(i) for i in [filename_syntax%(n) for n in range(1,20+1)]]
    data = np.mean(np.array(data).T, axis=1)
    return data

def main():
    train_2 = get_mean("cr-train-2-%d.txt")
    train_inf = get_mean("cr-train-inf-%d.txt")
    test_2 = get_mean("cr-test-2-%d.txt")
    test_inf = get_mean("cr-test-inf-%d.txt")

    means = [[train_2.tolist(), train_inf.tolist()], [test_2.tolist(), test_inf.tolist()]]
    count = range(0, len(means[0][0])*100, 100)

    drbm_2 = np.array(test_2) - np.array(train_2)
    drbm_inf = np.array(test_inf) - np.array(train_inf)
    datas = np.vstack((drbm_2, drbm_inf)).T
    np.savetxt("MNIST-noise-d0.4-adam-general.csv", datas, delimiter=",", header="DRBM(2)-generalization-error, DRBM(inf)-generalization-error")
    print(datas.shape)
    
    # mpl.rcParams["font.family"] = "IPAPGothic"
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,7))

    # for ax, datas, ylabel in zip(axes, means, ["訓練誤差", "テスト誤差"]):
    #     #ax.set_xlim([0,1000])
    #     ax.set_ylim([0,0.6])
    #     ax.set_title("学習回数に対する%sの推移\n(20回の試行の平均,パラメータ固定,訓練データ数1,000,\nテストデータのみガウスノイズ加算(μ=0,σ=0.4)),\nmomentum"%ylabel)
    #     ax.set_xlabel("学習回数")
    #     ax.set_ylabel(ylabel)
    #     for d, t in zip(datas, ["DRBM(2)", "DRBM(inf)"]):
    #         ax.plot(count, d, label=t)
    #     ax.legend()
    #     ax.grid(True)
    # plt.show()

if __name__=="__main__":
    if not len(sys.argv)==2:
        print("directory is not set")
        exit()
    os.chdir(sys.argv[1])
    main()