from TICC.TICC_solver import TICC
import glob
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

cmaps= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#3182BD', '#6BAED6', '#9ECAE1', '#E6550D', '#FD8D3C',
        '#FDAE6B', '#31A354', '#74C476', '#A1D99B', '#756BB1',
        '#9E9AC8', '#BCBDDC', '#636363', '#969696', '#BDBDBD']

file_list = sorted(glob.glob("../data/CPD/csv/*.csv"))
print(file_list)


BOCD = {'case': [1,8,9,14,4],
        'dataset' : ['dataset1','dataset1','dataset1','dataset1','dataset2'],
        'CP': [[41265, 41288, 41860, 43020],
               [3431, 5357, 6731, 9842, 19411],
               [44455,45539,45555,49322,61652],
               [63238,66706,70176,71507,80543,82629],
               [34959,38177,38771,41019,41074,41603]],
        'start': [40000,0,41897,62842,62842,31704],
        'end': [45000,20948,62846,83789,83789,42272]}

max_clusters = [3, 4, 5]
window_size = 5  # 1
maxIters = 8  # 100      ## number of Iterations of the smoothening
beta = 400  # 600                 ## smoothness parameter
lambda_parameter = 11e-2  # 5e-3  ## regularization parameter, sparsity of MRF for each clusters
threshold = 2e-5  # 2e-5

for itt in range(1):
    #### Load data
    dataset = BOCD['dataset'][itt]
    case = list(map(int, str(BOCD['case'][itt])))

    resample_interval = None  # '60T' # T=min

    data_sensor = []
    sensor_cols_merge = []
    for num_case in case:
        num_case = num_case - 1

        if dataset == 'dataset2':
            fname = file_list[num_case + 24]
        else:
            fname = file_list[num_case]

        df = pd.read_csv(fname, low_memory=False)

        # extract time index
        df.set_index(['Date'], inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        sensor_cols = list(df.columns[df.columns.str.contains('ACNTS')])

        # concatenate data
        data_sensor.append(df[sensor_cols])
    df_sensor = pd.concat(data_sensor)

    # reset column names to integer
    df_sensor.columns = range(df_sensor.shape[1])
    df_sensor = df_sensor.sort_index().reset_index(drop=True)

    # scaler = StandardScaler()
    # df_sensor = scaler.fit_transform(df_sensor)

    #### TICC training
    fname = df_sensor[BOCD['start'][itt]:BOCD['end'][itt]].reset_index(drop=True)

    t = 0
    bic = []
    fig, axs = plt.subplots(len(max_clusters), 1, figsize=(10, 4 * 3))
    for num_clusters in max_clusters:
        print('num_clusters:', num_clusters)
        ticc = TICC(window_size=window_size, number_of_clusters=num_clusters, lambda_parameter=lambda_parameter,
                    beta=beta, maxIters=maxIters, threshold=threshold,
                    write_out_file=False, prefix_string="results/", num_proc=1,
                    compute_BIC=True)
        (cluster_assignment, cluster_MRFs, bic) = ticc.fit(input_file=fname)

        np.savetxt(
            '../results/models/ticc_results/Results_{}_case_{}_winsize={}_{}clusters_beta{}_lambda{}_example_{}th_specific_region.txt'.format(dataset, case, window_size, num_clusters, beta, lambda_parameter, itt),
            cluster_assignment, fmt='%d', delimiter=',')
        print("bic/{}cluster: {}".format(num_clusters, bic))

        for j in range(len(BOCD['CP'][itt])):
            axs[t].axvline((BOCD['CP'][itt][j] - BOCD['start'][itt]), color='k', linestyle='-', linewidth=0.6)

        for k in range(num_clusters):
            df_sensor_temp = fname.copy()
            df_sensor_temp.loc[np.where(cluster_assignment != k)[0]] = np.nan
            axs[t].plot(df_sensor_temp.fillna(0), c=cmaps[k], alpha=0.5, linewidth=1.)
            axs[t].set_title('# of clusters = {}'.format(num_clusters))
        t += 1

    plt.tight_layout()
    plt.savefig(
        '../results/plots/ticc/Results_{}_{}case_w={}_beta{}_lambda{}_example_{}th_specific_region.png'.format(dataset,
                                                                                                           case,
                                                                                                           window_size,
                                                                                                           beta,
                                                                                                           lambda_parameter,
                                                                                                           itt))

# # load results and plot each number of clusters
# dataset = 'dataset1'
# case = [1]
# # window_size, max_clusters, beta, lambda_parameter = 5, [3, 4, 5], 400, 11e-2
# # resample_interval = None  # '60T' # T=min
# BOCD = [41265, 41288, 41860, 43020]
#
# t = 0
# fig, axs = plt.subplots(len(max_clusters), 1, figsize=(10, 3 * 3))
# for num_clusters in max_clusters:
#     input_file = '../results/models/ticc_results/Results_{}_case_{}_winsize={}_{}clusters_beta{}_lambda{}_example_{}th_specific_region.txt'.format(
#         dataset, case, window_size, num_clusters, beta, lambda_parameter, itt)
#
#     cluster_assignment = np.loadtxt(input_file, delimiter=",")
#
#     # plot
#     # trans = mtransforms.blended_transform_factory(axs.transData, axs.transAxes)
#     # axs[t].fill_between(df_pmmode[40000:45000].index.values, 0, 1, where=df_pmmode[40000:45000].sum(axis=1) != 0,
#     #                     facecolor='r', alpha=0.2, transform=trans)
#     # BOCD results
#     for j in range(len(BOCD)):
#         axs[t].axvline(BOCD[j], color='k', linestyle='-', linewidth=0.6)
#
#     # plot
#     # trans = mtransforms.blended_transform_factory(ax.transData, axs.transAxes)
#     # axs[t].fill_between(df_pmmode[40000:45000].index.values, 0, 1, where=df_pmmode[40000:45000].sum(axis=1) != 0,
#     #                     facecolor='r', alpha=0.2, transform=trans)
#     for k in range(num_clusters):
#         df_sensor_temp = df_sensor.copy()
#         df_sensor_temp.loc[np.where(cluster_assignment != k)[0]] = np.nan
#         axs[t].plot(df_sensor_temp[40000:45000], c=cmaps[k], alpha=0.5, linewidth=1.)
#         axs[t].set_title('# of clusters = {}'.format(num_clusters))
#     t += 1
# plt.tight_layout()
# plt.savefig('../results/plots/ticc/Results_{}_{}case_w={}_beta{}_lambda{}_example_{}th_specific_region.png'.format(dataset, case, window_size, beta,
#                                                                                  lambda_parameter, itt))
# plt.show()