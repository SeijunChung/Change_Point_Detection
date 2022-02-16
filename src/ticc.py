from TICC.TICC_solver import TICC

import os
import sys
import glob
import time
import copy

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

cmaps= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#3182BD', '#6BAED6', '#9ECAE1', '#E6550D', '#FD8D3C',
        '#FDAE6B', '#31A354', '#74C476', '#A1D99B', '#756BB1',
        '#9E9AC8', '#BCBDDC', '#636363', '#969696', '#BDBDBD']

file_list = sorted(glob.glob("../data/CPD/csv/*.csv"))
print(file_list)

# load data
dataset = 'dataset1'
cases = [3, 8]
resample_interval = None  # '60T' # T=min

data_sensor = []
data_bay = []
data_pmmode = []
sensor_cols_merge = []

for case in cases:
    case = case - 1

    if dataset == 'dataset2':
        fname = file_list[case + 24]
    else:
        fname = file_list[case]

    df = pd.read_csv(fname, low_memory=False)

    # extract time index
    df.set_index(['Date'], inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # sensors = df.loc[:, df.columns.str.contains('Bay_NH3')]
    pm_mode_cols = list(df.columns[df.columns.str.contains('PM Mode')])
    pm_mode = df[pm_mode_cols]
    pm_mode = pm_mode.replace({False: 0, True: 1, np.nan: 0})
    # resampling
    # if resample_interval != None:
    #     df = df.resample(resample_interval).median()
    #     pm_mode = pm_mode.resample(resample_interval).median()

    # forwardfill Nan with linear interpolation
    bay_cols = list(df.columns[df.columns.str.contains('Bay_NH3')])  # .pop(list(df.filter(regex='Filter')))
    bay_cols = [x for x in bay_cols if "Filter" not in x]
    # bays = df.loc[:, df.columns.str.contains('Bay_NH3')].columns.drop(list(df.filter(regex='Filter')))

    sensor_cols = list(df.columns[df.columns.str.contains('ACNTS')])
    print(sensor_cols)

    pm_mode_temp = pm_mode.iloc[:, 0]

    # concatenate data
    data_sensor.append(df[sensor_cols])
    data_bay.append(df[bay_cols])
    data_pmmode.append(pm_mode)
quit()
df_sensor = pd.concat(data_sensor)  # overlap with each same port
df_bay = pd.concat(data_bay)
df_pmmode = pd.concat(data_pmmode)

# # including bay NH3 into sensor data ///////////////////////////////////
# df_sensor = pd.concat([df_bay, df_sensor], axis=1)

# reset column names to integer
df_sensor.columns = range(df_sensor.shape[1])
df_sensor = df_sensor.sort_index().reset_index(drop=True)
df_pmmode = df_pmmode.sort_index().reset_index(drop=True)

df_sensor = df_sensor.interpolate(method='linear', limit_direction='forward')
print('before dropna:', df_sensor.shape, df_pmmode.shape)
# df_sensor = df_sensor.dropna()
# df_pmmode = df_pmmode.dropna()
df_sensor = df_sensor.fillna(0)
df_pmmode = df_pmmode.fillna(0)
print('after dropna:', df_sensor.shape, df_pmmode.shape)

# # Normalization with Standard scaler
scaler = StandardScaler()
scaler.fit(df_sensor)
df_sensor = scaler.transform(df_sensor)

# plot data samples
# fig1, ax1 = plt.subplots(figsize=(20, 5))
# trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
# ax1.fill_between(df_pmmode.index.values, 0, 1, where=df_pmmode.sum(axis=1) != 0,
#                 facecolor='r', alpha=0.3, transform=trans)
# ax1.plot(df_sensor, alpha=0.8, linewidth=0.5)
# plt.show()

# save txt
# np.savetxt('output_folder/Data_sensor_{}_{}case_{}.txt'.format(dataset,case,resample_interval), df_sensor, fmt='%d', delimiter=',')
# np.savetxt('output_folder/Data_pmmode_{}_{}case_{}.txt'.format(dataset,case,resample_interval), df_pmmode, fmt='%d', delimiter=',')
# print('txt file is saved: output_folder/Data_sensor_{}_{}case_{}_includingbay.txt'.format(dataset,case,resample_interval))
# print('txt file is saved: output_folder/Data_pmmode_{}_{}case_{}.txt'.format(dataset,case,resample_interval))

# find proper window_size after resampling
intervals = 3800
w = 10

# plot data samples
# fig2, ax2 = plt.subplots(figsize=(10,5))
# trans = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
# ax2.fill_between(df_pmmode.index.values[intervals:intervals+w], 0, 1, where=df_pmmode[intervals:intervals+w].sum(axis=1) !=0,
#                 facecolor='r', alpha=0.3, transform=trans)
# ax2.plot(df_sensor[intervals:intervals+w], alpha=0.8, linewidth=0.5)
# fig2.suptitle("resample=60min, window size=10 steps")
# plt.show()

BOCD = {'case': [1,8,9,11,14,4],
        'dataset' : ['dataset1','dataset1','dataset1','dataset1','dataset1','dataset2'],
        'CP': [[41265, 41288, 41860, 43020],
               [3431,5357,6731,9842,19411],
               [44455,45539,45555,49322,61652],
               [62842,63489,64707,65332,65966,66781,66925,67761,68287,69546,69766,70475,71436,71801,72608,73202,73798,74095,74614,74681,75017,76064,76342,77369,77840,78349,79500,80095,81113,81622,81870,82379,83042,83176,83627],
               [63238,66706,70176,71507,80543,82629],
               [34959,38177,38771,41019,41074,41603]],
        'start': [40000,0,41897,62842,62842,31704],
        'end': [45000,20948,62846,83789,83789,42272]}

max_clusters = [3, 4, 5]
window_size = 5  # 1
maxIters = 8  # 100      ## number of Iterations of the smoothening
beta = 400  # 600                 ## smoothness parameter
lambda_parameter = 11e-2  # 5e-3 #11e-2  ## regularization parameter, sparsity of MRF for each clusters
threshold = 2e-5  # 2e-5

for itt in range(6):
    #### Load data
    dataset = BOCD['dataset'][itt]
    case = list(map(int, str(BOCD['case'][itt])))
    resample_interval = None  # '60T' # T=min

    data_sensor = []
    data_bay = []
    data_pmmode = []
    sensor_cols_merge = []
    for num_case in case:
        num_case = num_case - 1

        if dataset == 'dataset2':
            fname = file_list[num_case + 24]
        else:
            fname = file_list[num_case]

        print(fname)
        df = pd.read_csv(fname, low_memory=False)

        # extract time index
        df.set_index(['Date'], inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # sensors = df.loc[:, df.columns.str.contains('Bay_NH3')]
        pm_mode_cols = list(df.columns[df.columns.str.contains('PM Mode')])
        pm_mode = df[pm_mode_cols]
        pm_mode = pm_mode.replace({False: 0,
                                   True: 1,
                                   np.nan: 0})
        # resampling
        if resample_interval != None:
            df = df.resample(resample_interval).median()
            pm_mode = pm_mode.resample(resample_interval).median()

        # forwardfill Nan with linear interpolation
        bay_cols = list(df.columns[df.columns.str.contains('Bay_NH3')])  # .pop(list(df.filter(regex='Filter')))
        bay_cols = [x for x in bay_cols if "Filter" not in x]
        # bays = df.loc[:, df.columns.str.contains('Bay_NH3')].columns.drop(list(df.filter(regex='Filter')))

        sensor_cols = list(df.columns[df.columns.str.contains('ACNTS')])

        pm_mode_temp = pm_mode.iloc[:, 0]

        # plot data samples
        fig, ax = plt.subplots(figsize=(20, 5))
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(pm_mode.index.values, 0, 1, where=pm_mode.sum(axis=1) != 0,
                        facecolor='r', alpha=0.3, transform=trans)
        ax.plot(df, alpha=0.8, linewidth=0.5)
        plt.show()

        # concatenate data
        data_sensor.append(df[sensor_cols])
        data_bay.append(df[bay_cols])
        data_pmmode.append(pm_mode)
    df_sensor = pd.concat(data_sensor)  # overlap with each same port
    df_bay = pd.concat(data_bay)
    df_pmmode = pd.concat(data_pmmode)

    # reset column names to integer
    df_sensor.columns = range(df_sensor.shape[1])
    df_sensor = df_sensor.sort_index().reset_index(drop=True)
    df_pmmode = df_pmmode.sort_index().reset_index(drop=True)

    # df.loc[:, sensor_cols] = df[sensor_cols].interpolate(method ='linear', limit_direction ='forward')
    df_sensor = df_sensor.interpolate(method='linear', limit_direction='forward')
    df_sensor = df_sensor.fillna(0)
    df_pmmode = df_pmmode.fillna(0)

    #### TICC training
    fname = df_sensor[BOCD['start'][itt]:BOCD['end'][itt]].reset_index(drop=True)
    df_pmmode = df_pmmode[BOCD['start'][itt]:BOCD['end'][itt]].reset_index(drop=True)

    # xtick options
    #     idx_orig = []
    #     for tpoint in range(0, BOCD['end'][itt]-BOCD['start'][itt], 1000):
    #         idx_orig.append(tpoint)
    #     idx_val = []
    #     for tpoint in range(BOCD['start'][itt], BOCD['end'][itt], 1000):
    #         idx_val.append(tpoint)

    t = 0
    bic = []
    fig, axs = plt.subplots(len(max_clusters), 1, figsize=(10, 4 * 3))
    for num_clusters in max_clusters:
        print('num_clusters:', num_clusters)
        ticc = TICC(window_size=window_size, number_of_clusters=num_clusters, lambda_parameter=lambda_parameter,
                    beta=beta, maxIters=maxIters, threshold=threshold,
                    write_out_file=False, prefix_string="output_folder/", num_proc=1,
                    compute_BIC=True)
        (cluster_assignment, cluster_MRFs, bic) = ticc.fit(input_file=fname)

        np.savetxt(
            '../results/models/ticc_results/Results_{}_{}case_w={}_{}clusters_beta{}_lambda{}_{}_example_{}th_specific_region_.txt'.format(dataset, case, window_size, num_clusters, beta, lambda_parameter, resample_interval, itt),
            cluster_assignment, fmt='%d', delimiter=',')
        print("bic/{}cluster: {}".format(num_clusters, bic))

        # plot
        #         trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        #         axs[t].fill_between(df_pmmode.index.values, 0, 1, where=df_pmmode.sum(axis=1) !=0,
        #                         facecolor='r', alpha=0.2, transform=trans)
        # BOCD CP results
        for j in range(len(BOCD['CP'][itt])):
            axs[t].axvline((BOCD['CP'][itt][j] - BOCD['start'][itt]), color='k', linestyle='-', linewidth=0.6)

        for k in range(num_clusters):
            df_sensor_temp = fname.copy()
            df_sensor_temp = df_sensor_temp  # pd.DataFrame(np.nan_to_num(df_sensor_temp))
            df_sensor_temp.loc[np.where(cluster_assignment != k)[0]] = np.nan
            axs[t].plot(df_sensor_temp.fillna(0), c=cmaps[k], alpha=0.5, linewidth=1.)
            axs[t].set_title('# of clusters = {}'.format(num_clusters))
        t += 1

    fig.tight_layout()
    plt.savefig(
        '../results/plots/ticc/Results_{}_{}case_w={}_beta{}_lambda{}_{}_example_{}th_specific_region_.png'.format(dataset,
                                                                                                           case,
                                                                                                           window_size,
                                                                                                           beta,
                                                                                                           lambda_parameter,
                                                                                                           resample_interval,
                                                                                                           itt))
    plt.show()

# load results and plot each number of clusters
dataset = 'dataset1'
case = [8]
window_size, max_clusters, beta, lambda_parameter = 5, [3, 4, 5], 400, 11e-2
resample_interval = None  # '60T' # T=min
BOCD = [41265, 41288, 41860, 43020]

t = 0
fig, axs = plt.subplots(len(max_clusters), 1, figsize=(10, 3 * 3))
for num_clusters in max_clusters:
    input_file = 'results/models/ticc_results/Results_{}_{}case_w={}_{}clusters_beta{}_lambda{}_{}_example_{}th_specific_region_.txt'.format(
        dataset, case, window_size, num_clusters, beta, lambda_parameter, resample_interval, itt)
    #     df_sensor_file = 'output_folder/Data_sensor_{}_{}case_{}_includingbay.txt'.format(dataset,case,resample_interval) #_includingbay_scaler
    #     df_pmmode_file = 'output_folder/Data_pmmode_{}_{}case_{}.txt'.format(dataset,case,resample_interval)

    cluster_assignment = np.loadtxt(input_file, delimiter=",")
    #     df_sensor = pd.read_csv(df_sensor_file, delimiter=",")
    #     df_pmmode = pd.read_csv(df_pmmode_file, delimiter=",")

    # plot
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    axs[t].fill_between(df_pmmode[40000:45000].index.values, 0, 1, where=df_pmmode[40000:45000].sum(axis=1) != 0,
                        facecolor='r', alpha=0.2, transform=trans)
    # BOCD results
    for j in range(len(BOCD)):
        axs[t].axvline(BOCD[j], color='k', linestyle='-', linewidth=0.6)

    # plot
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    axs[t].fill_between(df_pmmode[40000:45000].index.values, 0, 1, where=df_pmmode[40000:45000].sum(axis=1) != 0,
                        facecolor='r', alpha=0.2, transform=trans)
    for k in range(num_clusters):
        df_sensor_temp = df_sensor.copy()
        df_sensor_temp.loc[np.where(cluster_assignment != k)[0]] = np.nan
        axs[t].plot(df_sensor_temp[40000:45000], c=cmaps[k], alpha=0.5, linewidth=1.)
        axs[t].set_title('# of clusters = {}'.format(num_clusters))
    t += 1
plt.tight_layout()
plt.savefig('results/plots/ticc/Results_{}_{}case_w={}_beta{}_lambda{}_{}_example_{}th_specific_region.png'.format(dataset, case, window_size, beta,
                                                                                 lambda_parameter, resample_interval, itt))
plt.show()