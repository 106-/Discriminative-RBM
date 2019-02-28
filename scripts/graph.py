#!/usr/bin/env python
# -*- coding:utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser(description="make average graph from log file")
parser.add_argument("directory", action="store", type=str, help="set directory of log file")
parser.add_argument("setting_file", action="store", type=str, help="specify setting file")
args = parser.parse_args()

class DataSeries:
    def __init__(self, logs):
        self.logs = logs

    def _get_values_mean(self, value_name):
        log_matrix = []
        for log_onetime in self.logs:
            log_column = log_onetime["log"][value_name]
            log_vector = list(map(lambda x: x[1], log_column))
            log_matrix.append(log_vector)
        m = np.mean(np.array(log_matrix), axis=0)
        return m

    def kl_divergence(self):
        return self._get_values_mean("KL-Divergence")
    def test_correct_rate(self):
        return self._get_values_mean("test-correct-rate")
    def train_correct_rate(self):
        return self._get_values_mean("train-correct-rate")
    def test_error_rate(self):
        return 1 - self._get_values_mean("test-correct-rate")
    def train_error_rate(self):
        return 1 - self._get_values_mean("train-correct-rate")
    def generalization_error_rate(self):
        return self.test_error_rate() - self.train_error_rate()
    
    def train_count_range(self):
        first_log = self.logs[0]
        return range(0, first_log["learning_num"])
    def train_epoch_range(self):
        first_log = self.logs[0]
        epoch_num = first_log["learning_num"] * first_log["batch_size"] / first_log["train_data_length"]
        test_interval_epoch = first_log["test_interval"] / (first_log["train_data_length"] / first_log["batch_size"])
        return np.arange(0, epoch_num+test_interval_epoch, test_interval_epoch)

def main():
    settings = json.load(open(args.setting_file, "r"))
    filelist = os.listdir(args.directory)

    data_series_objects = [] 
    for t in settings["data-types"]:
        datas = []
        for f in filelist:
            if t["filename_includes"] in f and "log" in f:
                datas.append(json.load( open(os.path.join(args.directory,f),"r")))
        data_series_objects.append( DataSeries(datas) )

    plot_datas = []
    plot_counts = []
    for p in settings["plots"]:
        values = []
        for d in data_series_objects:
            if p["type"] == "KL-Divergence":
                values.append(d.kl_divergence())
            elif p["type"] == "test-error-rate":
                values.append(d.test_error_rate())
            elif p["type"] == "train-error-rate":
                values.append(d.train_error_rate())
            elif p["type"] == "generalization-error-rate":
                values.append(d.generalization_error_rate())
        if p["xtype"] == "epoch":
            plot_counts.append(data_series_objects[0].train_epoch_range())
        elif p["xtype"] == "count":
            plot_counts.append(data_series_objects[0].train_count_range())
        plot_datas.append(values)

    mpl.rcParams["font.family"] = "IPAPGothic"
    mpl.rcParams["font.size"] = 20
    fig, axes = plt.subplots(nrows=1, ncols=len(plot_datas), figsize=(8,4))

    fig.subplots_adjust(wspace=0.3)

    if len(plot_datas) == 1:
        axes = np.array(axes)

    for ax, plot, plot_data, plot_count in zip(axes.reshape(-1), settings["plots"], plot_datas, plot_counts):
        # ax.set_xlim([0,1000])
        # ax.set_ylim([0,1.0])
        ax.set_title(plot["title"])
        ax.set_xlabel(plot["xlabel"])
        ax.set_ylabel(plot["ylabel"])
        for data_type, data in zip(settings["data-types"], plot_data):
            ax.plot(plot_count, data, label=data_type["name"], linewidth=4.0)
        ax.legend()
        ax.grid(True)
    plt.show()

if __name__=="__main__":
    main()