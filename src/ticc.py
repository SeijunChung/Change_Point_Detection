from TICC.TICC_solver import TICC
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cmaps= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#3182BD', '#6BAED6', '#9ECAE1', '#E6550D', '#FD8D3C',
        '#FDAE6B', '#31A354', '#74C476', '#A1D99B', '#756BB1',
        '#9E9AC8', '#BCBDDC', '#636363', '#969696', '#BDBDBD']

file_list = sorted(glob.glob("../data/CPD/csv/*.csv"))

BOCD = {'case': [1],
        'dataset' : ['dataset1'],
        'CP': [[41265, 41288, 41860, 43020]],
        'start': [40000],
        'end': [45000]}

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
            if j == 0:
                axs[t].axvline((BOCD['CP'][itt][j] - BOCD['start'][itt]), color='red', linestyle='--', linewidth=0.7,
                               label="Result of BOCPD")
            else:
                axs[t].axvline((BOCD['CP'][itt][j] - BOCD['start'][itt]), color='red', linestyle='--', linewidth=0.7)
        axs[t].legend(fontsize=13)
        for k in range(num_clusters):
            df_sensor_temp = fname.copy()
            df_sensor_temp.loc[np.where(cluster_assignment != k)[0]] = np.nan
            axs[t].plot(df_sensor_temp, c=cmaps[k], alpha=0.5, linewidth=1.7)
            axs[t].set_title('# of clusters = {}'.format(num_clusters), fontsize=12, weight="bold")
        t += 1

    plt.tight_layout()
    plt.savefig(
        '../results/plots/ticc/Results_{}_{}case_w={}_beta{}_lambda{}_example_{}th_specific_region.png'.format(dataset,
                                                                                                       case,
                                                                                                           window_size,
                                                                                                           beta,
                                                                                                           lambda_parameter,
                                                                                                           itt))
    plt.show()
