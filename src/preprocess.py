import math

import copy
import glob
import datetime
import numpy as np
import pandas as pd
import gpytorch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import torch
from torch.utils.data import DataLoader
from scipy.signal import find_peaks
from scipy.interpolate import Rbf
from utils import make_dirs, file_names, cases, plot_interpolants, plot_gps, plot_timeseries, get_irregularly_sampled_data, mean_square_error, mean_absolute_percentage_error
from models import MultitaskGPModel, ExactGPModel, GPRegressionModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
pd.set_option("display.max_rows", 700)
pd.set_option("display.max_columns", 10)
device = 'cuda:7' if torch.cuda.is_available() else 'cpu'

def z_score_normalization(observation, axis):
    mean = np.outer(np.mean(observation, axis=axis), np.ones(observation.shape[axis])) if axis == 1 else np.mean(observation, axis=axis)
    std = np.outer(np.std(observation, axis=axis), np.ones(observation.shape[axis])) if axis == 1 else np.std(observation, axis=axis)
    return np.divide((observation - mean), std)


def format_time(date_time: str):
    t = datetime.datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S.%f")
    if t.microsecond / 1e6 >= 0.5:
        t = t + datetime.timedelta(seconds=1)
    return t.strftime('%Y-%m-%d %H:%M:%S')


def get_rt_box(df_, va: str):
    dic = df_.loc["dtDate"].to_dict()
    if va in dic.values():
        for k, v in dic.items():
            if v == va:
                return df_[k]["sRT"]


def get_area_in_RTbox(f, rt_box_start_point: int, rt_box_end_point: int):
    return np.sum(f[rt_box_start_point:rt_box_end_point])


def get_differential(observations: list):
    return [observations[i + 1] - observations[i] for i in range(len(observations) - 1)]


def get_index_int_starts(df_):
    for i in df_.index:  # find row index that type 'int' starts
        if isinstance(df_[1][i], int):
            break
    return i


def get_base_line(observations: np.array, period: np.array, n: np.array, end: bool):
    if end:
        return [observations.transpose()[i, k-j:k].mean() for i, (j, k) in enumerate(zip((5 / (period * 1e-3)).astype(np.int16), n))]
    else:
        return [observations.transpose()[i, :j].mean() for i, j in enumerate((5 / (period * 1e-3)).astype(np.int16))]


def get_difference_max_in_rt_and_peak_before_rt(rt_box_start_point, peak_y, observations):
    return np.max(peak_y) - np.max(observations[rt_box_start_point-20: rt_box_start_point])


def peaks_in_rt(observations: list, rt_box_start_point: int, rt_box_end_point: int, ind):
    signal = observations[rt_box_start_point:rt_box_end_point]
    max_in_rt_box = np.max(signal)
    peak_x, _ = find_peaks(signal, distance=20, width=7, prominence=max_in_rt_box*0.04, height=max_in_rt_box*0.2)
    if len(peak_x) == 0:
        peak_x = np.append(peak_x, np.argmax(signal))

    peak_y = signal[peak_x]
    if max_in_rt_box != np.max(peak_y):
        peak_y = [max_in_rt_box]
    peak_x = [x + rt_box_start_point for x in peak_x]
    peaks = [(x, y) for x, y in zip(peak_x, peak_y)]

    difference_max_in_rt_and_peak_before_rt = max_in_rt_box - np.max(observations[rt_box_start_point-20: rt_box_start_point])
    return peaks, difference_max_in_rt_and_peak_before_rt


def get_rt_info(rt_info: pd.core.frame.DataFrame):
    rt_info.drop([0, 2], inplace=True)  # get rid of 2 rows not necessary
    rt_info.index = rt_info["A"].map(lambda x: x.split("=")[0]).tolist()  # get indices

    for i in rt_info.columns:
        if rt_info[i]["sRtOffset2"] is np.nan:
            rt_info[i]["sRtOffset2"] = rt_info[i]["sRtOffset1"]
        rt_info[i] = rt_info[i].map(lambda x: x.split("=")[1])
        rt_info[i]["dtDate"] = "".join(rt_info[i]["dtDate"].split("-")[:3])[2:] + "-" + "".join(rt_info[i]["dtDate"].split("-")[-1].split(":")[:2])
        rt_info[i]["sRT"] = (round(float(rt_info[i]["sRT"])-float(rt_info[i]["sRtOffset1"]), 2),
                             round(float(rt_info[i]["sRT"])+float(rt_info[i]["sRtOffset2"]), 2))
    rt_info.drop(["sRtOffset1", "sRtOffset2"], axis=0, inplace=True)
    rt_info = rt_info.loc[:, ~rt_info.loc["dtDate"].duplicated()]

    assert rt_info.isnull().any().any() == False
    return rt_info


def reshaping_df(df_original: pd.core.frame.DataFrame):
    int_col = [col for col in df_original.columns if isinstance(col, int)]
    str_col_index = [df_original.columns.get_loc(col) for col in df_original.columns if not isinstance(col, int) and
                     df_original.columns.get_loc(col) in np.array(int_col)*2-1]

    int_starts = get_index_int_starts(df_original)

    cols = df_original[int_col[0]][:int_starts].values

    column_names = {"Baseline(정상0, 비정상1)": "Baseline",
                    "Chrome 점검(정상:0, 비정상:1)": "Chrome",
                    "진/가성(진:0, 가:1)": "label",
                    "크롬버전": "Spectrum Version",
                    "RT  파일": "RT  File",
                    "측정시작": "Analysis Start",
                    "측정종료": "Analysis End",
                    "측정포트": "Analysis Port",
                    "측정주기": "Analysis Period",
                    "측정개수": "Analysis Number"}

    cols = [column_names.get(col, col) for col in cols.copy()]

    df_ = pd.DataFrame(data=df_original.values[:int_starts, str_col_index].transpose(),
                       columns=cols,
                       index=range(1, int_col[-1] + 1))

    observation = df_original.values[int_starts:, str_col_index].astype(np.float32)

    df_["Baseline_start"] = get_base_line(observation, df_['Analysis Period'].values, df_['Analysis Number'].values, False)
    df_["Baseline_end"] = get_base_line(observation, df_['Analysis Period'].values, df_['Analysis Number'].values, True)
    df_["RT  File"] = df_["RT  File"].apply(lambda x: x[2:].split(".")[0])

    return df_, observation


def get_data(task, path, sheet_name, fn, how_to_sample=None, how_to_interpolate=None):
    if task == "task1":
        return get_data_task_1(path)
    elif task == "task2":
        return get_data_task_2(path, fn, sheet_name)
    elif task == "task3":
        return get_data_task_3(path, fn, sheet_name, how_to_sample, how_to_interpolate)
    else:
        raise NotImplementedError


def get_data_task_1(path):
    fp_fab_amc = [fp for fp in glob.glob(path+"/*.xlsx") if "FAB-AMC" in fp]
    for fp in fp_fab_amc:
        if "1st" in fp:
            df_1 = pd.read_excel(open(fp, "rb"), sheet_name="진가성", engine='openpyxl')
            df_2 = pd.read_excel(open(fp, "rb"), sheet_name="Baseline", engine='openpyxl')
            df_1_rt_info = pd.read_excel(open(fp, 'rb'), sheet_name="RT_info", engine='openpyxl')
            df_1_rt_info = df_1_rt_info[0:6]
        elif "2nd" in fp:
            df_3 = pd.read_excel(open(fp, 'rb'), sheet_name="진성", engine='openpyxl')
            df_2_rt_info = pd.read_excel(open(fp, 'rb'), sheet_name="RT_info", engine='openpyxl')

    df1, df1_observations = reshaping_df(df_1)
    df2, df2_observations = reshaping_df(df_2)
    df3, df3_observations = reshaping_df(df_3)
    df3["label"] = 0

    df = pd.concat([df1, df2, df3], ignore_index=True, join="outer")

    no_samples = df1_observations.shape[1] + df2_observations.shape[1] + df3_observations.shape[1]
    max_analysis_number = max(df1_observations.shape[0], df2_observations.shape[0], df3_observations.shape[0])

    temp = np.empty((max_analysis_number - df1_observations.shape[0], df1_observations.shape[1]))
    temp[:] = np.NaN
    df1_observations = np.concatenate([df1_observations, temp], axis=0)
    temp = np.empty((max_analysis_number - df3_observations.shape[0], df3_observations.shape[1]))
    temp[:] = np.NaN
    df3_observations = np.concatenate([df3_observations, temp], axis=0)

    observations = np.concatenate([df1_observations, df2_observations, df3_observations], axis=1).transpose().astype(np.float32)

    for i in range(no_samples):
        ts = observations[i, :].copy()
        mask = np.isnan(ts)
        if len(ts[mask]) > 0:
            observations[i, :][mask] = round(df["Baseline_end"][i], 5)

    assert (len(np.argwhere(np.isnan(observations))) == 0)

    df.loc[(df["Chrome"] == 1), "label"] = 1
    df.loc[(df["Chrome"] == 0), "label"] = 0
    df.loc[(df["Baseline"] == 0) & (df["Chrome"] == 0), "label"] = 0

    df1_rt_info = get_rt_info(df_1_rt_info)
    df2_rt_info = get_rt_info(df_2_rt_info)
    df_rt_info = pd.concat([df1_rt_info, df2_rt_info], ignore_index=True, join="outer", axis=1)
    df_rt_info = df_rt_info.loc[:, ~df_rt_info.loc["dtDate"].duplicated()]
    df["RT_box"] = df["RT  File"].map(lambda x: get_rt_box(df_rt_info, x))

    np.save("../data/210422_Samsung_FAB-AMC_observations.npy", observations)

    return df, observations


def get_new_features(df, observations):
    features = ["rt_area", "rt_area_norm", "peaks", "multi_peaks", "increase_in_rt", "max_value", "min_value"]
    values = []

    for i in df.index:
        val = df.loc[i, ["RT_box", "Analysis Number", "Baseline_start", "Baseline_end", "label"]]
        max_val = max(observations[i, :])
        min_val = min(observations[i, :])
        s = int(val["RT_box"][0] * 250)
        e = int(val["RT_box"][1] * 250)
        area = get_area_in_RTbox(observations[i, :], s, e)
        area_norm = get_area_in_RTbox(observations[i, :] / max_val, s, e)
        peaks, increase_in_rt = peaks_in_rt(observations[i, :], s, e, i)
        multi_peaks = 1 if len(peaks) > 1 else 0
        # if len(peaks) == 0:
        #     nan_peaks.append(i)
        values.append([area, area_norm, peaks, multi_peaks, increase_in_rt, max_val, min_val])
        # plot_peaks_in_rt_box(observations[i, :], i, val["Analysis Number"], val["Baseline_start"], val["Baseline_end"],
        #                      val["label"], [tup[0] for tup in peaks], area, s, e)

    df[features] = np.array(values, dtype=np.object)

    df_ = df[["Baseline_start", "Baseline_end", "peaks", "multi_peaks", "increase_in_rt", "rt_area", "rt_area_norm", "max_value", "min_value", "Baseline", "Chrome"]].copy()

    df_["recovery"] = df_.apply(lambda x: abs(x["Baseline_end"] - x["Baseline_start"]) / (x["max_value"] - x["min_value"]), axis=1)
    df_["recovery_bool"] = df_.apply(lambda x: 0 if x["recovery"] < 0.03 else 1, axis=1)
    df_["increase_in_rt_norm"] = df_.apply(lambda x: x["increase_in_rt"] / (x["max_value"] - x["min_value"]), axis=1)
    df_["increase_in_rt_bool"] = df_.apply(lambda x: 0 if x["increase_in_rt_norm"] < 0.1 else 1, axis=1)
    df_.loc[(df_["recovery"] == 0.0), "Baseline"] = 0
    df_.drop(["Baseline_start", "Baseline_end", "max_value", "min_value"], axis=1, inplace=True)
    df_.to_csv("../data/210422_Samsung_FAB-AMC_feature_added.csv", index=False)
    return df_


def get_data_task_2(path, chroma, sheet_name):
    df = pd.read_excel(open(path + f'/task2/210305_Samsung_In-Situ_{chroma}_1st.xlsx', 'rb'), engine='openpyxl',
                       sheet_name=sheet_name)

    if sheet_name == 'E01_IPA1(1hr)':
        df.loc[(df["원인"].str.replace(" ", "") == "파츠성(Syringe)") & (df["진가성"].str.strip() != "가성"), "진가성"] = "가성"
    elif sheet_name == 'Y02_IPA(1hr)':
        df.drop(df.loc[(df["원인"].str.replace(" ", "") == "정상") & (df["진가성"].str.strip() == "가성")].index, inplace=True)
    elif sheet_name == 'G02_IPA1(1hr)':
        df.loc[(df["원인"].apply(pd.isnull)) & (df["진가성"].str.strip() == "가성"), "원인"] = "Tune"
    elif sheet_name == "H02_IPA1(1hr)":
        df.loc[(df["원인"].apply(pd.isnull)) & (df["진가성"].str.strip() == "가성"), "원인"] = "파츠성(Loop)"
    elif sheet_name == 'N01_HF1(1hr)':
        df["Mn"].fillna(method="ffill", inplace=True)
    elif sheet_name == "Y01_HF1(1hr)":
        df["Mg"].replace("ㅜ", 0, inplace=True)
        df["Mg"] = df["Mg"].astype(float)
        df.loc[(df["원인"].str.replace(" ", "") == "헌팅") & (df["진가성"].str.strip() != "가성"), "진가성"] = "가성"
    elif sheet_name == "Y02_HF(1hr)":
        df.drop(df.loc[(df["원인"].str.replace(" ", "") == "정상") & (df["진가성"].str.strip() == "가성")].index, inplace=True)
    elif sheet_name == "H01_HF(1hr)":
        df.drop(df.loc[(df["원인"].apply(pd.isnull)) & (df["진가성"].str.strip() == "가성")].index, inplace=True)

    assert len(df.loc[(df["원인"].apply(pd.isnull)) & (df["진가성"].str.strip() == "가성")].index) == 0

    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.loc[df["원인"].apply(pd.isnull), "원인"] = ""

    for col in df.columns:
        if "Unnamed" in col:
            if sheet_name == "Y02_IPA(1hr)":
                df.loc[df[col].str.strip() == "가성", ["원인", "진가성"]] = df.loc[df[col].str.strip() == "가성"][["진가성", col]].values
            df.drop([col], axis=1, inplace=True)

    time = copy.deepcopy(df["Time"])
    root_causes = copy.deepcopy(df["원인"])
    labels = copy.deepcopy(df["진가성"])

    df.drop(columns=["Time", "원인", "진가성"], inplace=True)

    diff = np.diff(df.values.astype(float), axis=0).sum(axis=1)

    states = [0]
    alert = 0
    tobedeleted = []
    for r_idx, val in enumerate(diff):
        if val != 0.0:
            alert += 1
            if alert > 1:
                tobedeleted.append(r_idx)
                alert = 0
        else:
            alert = 0
        states.append(alert)

    # df.drop(labels=tobedeleted, axis=0, inplace=True)
    df.iloc[tobedeleted] = np.NaN
    root_causes[root_causes.index.isin(tobedeleted)] = ""
    labels[labels.index.isin(tobedeleted)] = "무효"

    df_drop_dup = df.dropna(how="all").drop_duplicates()

    if sheet_name in ['H01_IPA(1hr)', 'N01_HF2(1hr)']:
        df_drop_dup[df_drop_dup.columns[df_drop_dup.isnull().any()].tolist()[0]].interpolate(method="ffill", limit_direction='forward', inplace=True, axis=0)

    df.iloc[~df.index.isin(df_drop_dup.index)] = np.NaN

    df["class"] = labels[df.index].str.strip().apply(lambda x: 1 if x == "가성" else (np.NaN if x == "무효" else 0))
    df["cause"] = root_causes[df.index]
    df["time"] = time[df.index].apply(lambda x: format_time(x))
    # print(df["class"].value_counts())
    # print(df["cause"].value_counts())

    assert all(r.replace(" ", "") in cases.keys() for r in df["cause"].unique() if isinstance(r, str))
    assert all(r in cases.keys() for r in df["cause"].unique() if not isinstance(r, str))

    for v in df["cause"].unique():
        df.loc[df["cause"] == v, "root_cause"] = cases[v.replace(" ", "")][0] if isinstance(v, str) else cases[v][0]

    df.drop(columns=["cause"], inplace=True)
    df.drop(df.loc[df["root_cause"] == "deleted"].index, inplace=True)
    df = df.dropna()

    df.to_csv(f"../data/task2/{chroma}/" + sheet_name + ".csv", index=False)
    return df


def get_data_task_3(path, fn, sheet_name, sampling_mode, interpolation_mode):

    df = pd.read_excel(path + f"/task3/{fn}.xlsx", header=1 if "17L" in fn else 0, engine='openpyxl', sheet_name=sheet_name)

    for col in df.columns:
        if "Unnamed" in col:
            df.drop([col], axis=1, inplace=True)

    # forward-fill for Empty value
    df.fillna(method="ffill", inplace=True)

    date = copy.deepcopy(df["Date"])
    pm_modes = copy.deepcopy(df[[c for c in df.columns if "Mode" in c]])
    elements = [element for element in df.columns if "ACNTS" in element and element not in ["|ACNTS_09|06_SA_P07"]]

    # Exceptions
    if "17L" in fn:
        if sheet_name == "Case5":
            elements = ["|ACNTS_09|08_SA_P04", "|ACNTS_09|08Bay_P03", "|ACNTS_09|09_SA1_P02", "|ACNTS_09|09_SA2_P01"]
        if sheet_name == "Case6":
            elements = ["|ACNTS_10|11_SA2_P06", "|ACNTS_10|11_SA3_P05", "|ACNTS_10|11_SA1_P07"]
        if sheet_name == "Case15":
            elements.remove("|ACNTS_11|23_SA_P03")
            elements.remove("|ACNTS_11|23Bay_P04")
        if sheet_name == "Case17":
            elements.remove("|ACNTS_11|27Bay_P06")
        if sheet_name == "Case21":
            elements = ["|ACNTS_13|35Bay_4_P07", "|ACNTS_13|35Bay_3_P06", "|ACNTS_13|35Bay_5_P08", "|ACNTS_13|35Bay_1_P04"]
        if sheet_name == "Case22":
            elements = ["|ACNTS_13|37Bay_4_P08", "|ACNTS_13|37Bay_2_P06", "|ACNTS_13|37Bay_3_P07", "|ACNTS_13|37Bay_1_P05"]
        if sheet_name == "Case22":
            elements = ["|ACNTS_13|37Bay_4_P08", "|ACNTS_13|37Bay_2_P06", "|ACNTS_13|37Bay_3_P07", "|ACNTS_13|37Bay_1_P05"]
        if sheet_name == "Case23":
            elements = ["|ACNTS_83|40_1SA_P03", "|ACNTS_83|40_2SA_P04", "|ACNTS_83|Reticle_Cleaning_P01", "|ACNTS_83|Reticle_Cleaning_P02"]
    elif "Dataset2" in fn:
        if sheet_name == "Case5":
            elements.remove("|ACNTS_91|08Bay_P21")
        if sheet_name == "Case6":
            elements = ["06Bay_NH3", "07Bay_NH3", "09Bay_NH3"]
        if sheet_name == "Case7":
            elements = ["|Filter|06Bay_NH3", "|Filter|09Bay_NH3"]
        if sheet_name == "Case8":
            elements = ["|ACNTS_91|10Bay_P23", "|ACNTS_91|11Bay_P24"]
        if sheet_name == "Case9":
            elements = ["10Bay_NH3", "11Bay_NH3", "12Bay_NH3"]
        if sheet_name == "Case13":
            elements = ["|Filter|15Bay_NH3", "|Filter|16Bay_NH3"]
        if sheet_name == "Case16":
            elements.remove("|ACNTS_93|17Bay_P17")
        if sheet_name == "Case19":
            elements = ["20Bay_NH3", "21Bay_NH3"]
        if sheet_name == "Case21":
            elements = ["31Bay_NH3", "32Bay_NH3", "33Bay_NH3"]
        if sheet_name == "Case33":
            elements = ["43Bay_NH3", "44Bay_NH3"]

    df.drop(columns=["Date"] + [c for c in df.columns if "Mode" in c], inplace=True)

    if sampling_mode == 0:
        pass
    elif sampling_mode == 1:
        df, deleted = get_irregularly_sampled_data(df, sampling_mode)

    elif sampling_mode == 2:
        df, deleted, true_label = get_irregularly_sampled_data(df, sampling_mode)

    elif sampling_mode == 3:
        deleted = []
        train_x, train_y, y_hat = [], [], []
        for i, col in enumerate(df.columns):
            df[col], deleted_temp = get_irregularly_sampled_data(df[col], sampling_mode)
            deleted.append(deleted_temp)
            train_x.append(df[col].dropna().index.tolist())
            train_y.append(df[col][df[col].dropna().index].tolist())

    elif sampling_mode == 4:
        deleted = []
        true_labels = []
        for i, col in enumerate(df.columns):
            df[col], deleted_temp, true_label = get_irregularly_sampled_data(df[col], sampling_mode)
            true_labels.append(true_label)
            deleted.append(deleted_temp)

    if interpolation_mode == 0:
        pass

    if interpolation_mode in [1, 2, 3, 4]:
        x = df.dropna().index.to_numpy()
        y = df[elements].dropna().values

        print(f"Equipments: {elements}, X: {x.shape}, y: {y.shape}, No. of time points deleted: {len(deleted)}")

        if interpolation_mode in [1,2,3]:
            train_x = torch.from_numpy(x)
            train_y = torch.from_numpy(y).contiguous().float()
        else:
            train_x = torch.Tensor(x)
            train_y = torch.Tensor(y)

        inference_x = torch.tensor(deleted)
        num_dimension = train_y.shape[1]

        true_x = true_label.index.to_numpy()
        true_y = true_label[elements].values

        train_x, train_y = train_x.to(device), train_y.to(device)

        if interpolation_mode == 1:
            for i in range(num_dimension):
                rbf = Rbf(x, y[:, i], smooth=0.1)
                y_inf_rbf = rbf(deleted).reshape(-1, 1) if i == 0 else np.append(y_inf_rbf, rbf(deleted).reshape(-1, 1), axis=1)
            interpolants_rbf = np.empty((len(df), y.shape[1]))
            interpolants_rbf[:] = np.nan
            interpolants_rbf[x] = y
            interpolants_rbf[deleted] = y_inf_rbf

            with open(f'../data/task3/interpolants/{fn}_{sheet_name}_samplingmode_{sampling_mode}_interpolants_rbf.npy', 'wb') as f:
                np.save(f, interpolants_rbf)

        elif interpolation_mode in [2, 3, 4]:

            # GPytorch
            training_iter = 20

            with torch.cuda.device(device=7), gpytorch.settings.cg_tolerance(100):

                if interpolation_mode == 2:

                    for dim, element in zip(range(num_dimension), elements):
                        print(f"{fn}_{sheet_name} ~~ {dim+1}th Equipments: {element} / total {num_dimension}")

                        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.SmoothedBoxPrior(0.15, 1.5, sigma=0.001)).to(device)
                        model = ExactGPModel(train_x, train_y[:, dim], likelihood).to(device)

                        # Find optimal model hyperparameters
                        model.train()
                        likelihood.train()

                        # Use the adam optimizer
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.1) # Includes GaussianLikelihood parameters

                        # "Loss" for GPs - the marginal log likelihood
                        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

                        # Zero gradients from previous iteration
                        for i in range(training_iter):
                            optimizer.zero_grad()
                            # Output from model
                            output = model(train_x)
                            # Calc loss and backprop gradients
                            loss = -mll(output, train_y[:, dim])
                            loss.backward()

                            print('Iter %d/%d - Loss: %.3f - lengthscale: %.3f   noise: %.3f' % (
                                i + 1, training_iter, loss.item(), model.covar_module.base_kernel.lengthscale.item(), model.likelihood.noise.item()))
                            optimizer.step()

                        # Get into evaluation (predictive posterior) mode
                        model.eval()
                        likelihood.eval()

                        # Make predictions
                        with torch.no_grad(), gpytorch.settings.fast_pred_var():
                            prediction = likelihood(model(inference_x))
                            mean = prediction.mean
                            lower, upper = prediction.confidence_region()
                            means = mean.cpu().numpy().reshape(-1, 1) if dim == 0 else np.append(means, mean.cpu().numpy().reshape(-1, 1), axis=1)
                            lowers = lower.cpu().numpy().reshape(-1, 1) if dim == 0 else np.append(lowers, lower.cpu().numpy().reshape(-1, 1), axis=1)
                            uppers = upper.cpu().numpy().reshape(-1, 1) if dim == 0 else np.append(uppers, upper.cpu().numpy().reshape(-1, 1), axis=1)

                    train_x, train_y, inference_x = train_x.cpu().numpy(), train_y.cpu().numpy(), inference_x.cpu().numpy()
                    interpolants = np.empty((len(df), num_dimension))
                    interpolants[:] = np.nan
                    interpolants[train_x] = train_y
                    interpolants[inference_x] = means

                    with open(f'../data/task3/interpolants/{fn}_{sheet_name}_sampling_{sampling_mode}_interpolation_mode_{interpolation_mode}.npy', 'wb') as f:
                        np.save(f, interpolants)

                if interpolation_mode == 3:
                    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_dimension).to(device)
                    model = MultitaskGPModel(train_x, train_y, likelihood, num_dimension).to(device)
                    # Find optimal model hyperparameters
                    model.train()
                    likelihood.train()

                    # Use the adam optimizer
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

                    # "Loss" for GPs - the marginal log likelihood
                    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

                    for i in range(training_iter):
                        optimizer.zero_grad()
                        output = model(train_x)
                        loss = -mll(output, train_y)
                        loss.backward()
                        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
                        optimizer.step()

                    # Set into eval mode
                    model.cpu().eval()
                    likelihood.cpu().eval()
                    torch.cuda.empty_cache()

                    train_x, train_y = train_x.cpu().numpy(), train_y.cpu().numpy()

                    # Make predictions
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        predictions = likelihood(model(inference_x))
                        mean = predictions.mean
                        lower, upper = predictions.confidence_region()

                    interpolants = np.empty((len(df), num_dimension))
                    interpolants[:] = np.nan
                    interpolants[train_x] = train_y
                    interpolants[inference_x] = mean

                    plot_gps(train_x, train_y, inference_x, mean, lower, upper, fn, sheet_name, 0, -1,
                             interpolation_mode)

                    with open(f'../data/task3/interpolants/{fn}_{sheet_name}_sampling_{sampling_mode}_interpolation_mode_{interpolation_mode}.npy', 'wb') as f:
                        np.save(f, interpolants)

                if interpolation_mode == 4:

                    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
                    model = GPRegressionModel(train_x, train_y, likelihood, train_x[:int(train_x.size(0)/2)]).to(device)

                    # Use the adam optimizer
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                    # "Loss" for GPs - the marginal log likelihood
                    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

                    def train():
                        for i in range(training_iter):
                            # Zero backprop gradients
                            optimizer.zero_grad()
                            # Get output from model
                            output = model(train_x)
                            # Calc loss and backprop derivatives
                            loss = -mll(output, train_y)
                            loss.backward()
                            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
                            optimizer.step()
                            torch.cuda.empty_cache()

                    train()

                    model.cpu().eval()
                    likelihood.cpu().eval()
                    torch.cuda.empty_cache()

                    train_x, train_y = train_x.cpu().numpy(), train_y.cpu().numpy()

                    with gpytorch.settings.max_preconditioner_size(10), torch.no_grad(), gpytorch.settings.fast_pred_var():
                        preds = model(inference_x)
                        predictions = likelihood(model(inference_x))
                        mean = predictions.mean
                        lower, upper = predictions.confidence_region()


                    interpolants = np.empty((len(df), num_dimension))
                    interpolants[:] = np.nan
                    interpolants[train_x] = train_y
                    interpolants[inference_x] = mean

                    plot_gps(train_x, train_y, inference_x, mean, lower, upper, fn, sheet_name, 0, -1, interpolation_mode)

                    with open(f'../data/task3/interpolants/{fn}_{sheet_name}_sampling_{sampling_mode}_interpolation_mode_{interpolation_mode}.npy', 'wb') as f:
                        np.save(f, interpolants)

        with open(f'../data/task3/interpolants/{fn}_{sheet_name}_samplingmode_{sampling_mode}_interpolants_rbf.npy', 'rb') as f:
            interpolants_rbf = np.load(f)

        with open(f'../data/task3/interpolants/{fn}_{sheet_name}_sampling_{sampling_mode}_interpolation_mode_{interpolation_mode}.npy', 'rb') as f:
            interpolants_gp = np.load(f)

        ind = [np.where(inference_x == a)[0].item() for a in true_x]

        mse_rbf, mse_gp, mape_rbf, mape_gp = 0, 0, 0, 0

        for i in range(num_dimension):
            mse_rbf += mean_square_error(true_y[:, i], interpolants_rbf[ind, i])
            mse_gp += mean_square_error(true_y[:, i], interpolants_gp[ind, i])
            mape_rbf += mean_absolute_percentage_error(true_y[:, i], interpolants_rbf[ind, i])
            mape_gp += mean_absolute_percentage_error(true_y[:, i], interpolants_gp[ind, i])

        print(f"MSE: RBF vs GP_mode_{interpolation_mode} ~~> {mse_rbf / num_dimension}, {mse_gp / num_dimension}")
        print(f"MAPE: RBF vs GP_mode_{interpolation_mode} ~~> {mape_rbf / num_dimension}, {mape_gp / num_dimension}")

        start = 12000
        end = 12500
        plot_interpolants(df, train_x, train_y, true_x, true_y, elements, interpolants_rbf, interpolants_gp, fn, sheet_name, start, end, interpolation_mode)

    return df, date, pm_modes, elements