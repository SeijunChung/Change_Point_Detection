import itertools
import os
from collections import Counter
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000000
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
# import tensorflow as tf
import collections


def make_dirs(path):
    """Make Directory If not Exists"""
    if not os.path.exists(path):
        os.makedirs(path)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def plot_not_hits(df, i, j, original_ind, pred_y, true_y, chroma, plot_path, task, word2index):
    index2w = dict()
    for k, v in word2index.items():
        if v not in index2w.keys():
            index2w[v] = k
    plt.figure(figsize=(7, 3))
    x_pos = np.arange(len(df.columns.difference(['class', 'time', 'root_cause', 'equipment_code'])))
    y_val = df.iloc[original_ind[i]][df.columns.difference(['class', 'time', 'root_cause', 'equipment_code'])]
    plt.bar(x_pos, y_val, color='k', alpha=.7)
    # plt.title(
    #     f"{chroma}, Label {df.iloc[original_ind[i]]['class']} ~ Cause {df.iloc[original_ind[i]]['root_cause']} : {true_y} but Predict : {index2w[pred_y]} :{pred_y}")
    plt.xticks(x_pos, df.columns.difference(['class', 'time', 'root_cause', 'equipment_code']))
    plt.tight_layout()
    plt.savefig(plot_path + f"/{task}/falsealarms/{chroma}/sample_lable : {df.iloc[original_ind[i]]['root_cause']} : {true_y}_predict : {index2w[pred_y]} : {pred_y}_No. {j}.png", dpi=100)
    plt.show()


def plot_peaks_in_rt_box(signals, i, n, bls, ble, label, peaks_x, area, s, e):
    plot_path = "../data/task1/samples/"
    make_dirs(plot_path)
    x_d = np.arange(1, n + 1)
    y = signals[:n].astype(float)
    plt.figure(figsize=(8, 10))
    plt.title(f"label:{label} Sample No.{i+1}: peaks at x={peaks_x}", fontsize=15)
    plt.plot(y)
    plt.scatter(peaks_x, y[peaks_x], marker="x", s=300, c="red")
    plt.plot(np.zeros_like(y), "--", color="gray")
    plt.axvline(s, linestyle='--', color='red', alpha=.5)
    plt.axvline(e, linestyle='--', color='red', alpha=.5)
    plt.fill_between(x_d, bls, y, where=(x_d >= s) & (x_d <= e), facecolor='skyblue', alpha=.8)
    plt.hlines(y=bls, xmin=x_d[0], xmax=x_d[200], linestyle='--', color='darkred', alpha=.9, linewidth=4)
    plt.hlines(y=ble, xmin=x_d[-200], xmax=x_d[-1], linestyle='--', color='darkred', alpha=.9, linewidth=4)
    plt.text(n/2, np.max(y) / 2, 'Area=' + str(round(area, 2)), fontsize=14, horizontalalignment='center')
    plt.text(n/6, bls+np.max(y)/50, 'Baseline start=' + str(round(bls, 6)), fontsize=14, horizontalalignment='center')
    plt.text(n*5/6, ble+np.max(y)/50, 'Baseline end=' + str(round(ble, 6)), fontsize=14, horizontalalignment='center')
    plt.grid()
    plt.savefig(plot_path + f"sample No.{i + 1}.jpg")
    # plt.savefig(f"../results/plots/task1/notcorrect/sample No.{i+1}.jpg")


def plot_timeseries(df, date, pm_modes, fn, case, s, e, sampling_mode, interpolation_mode, norm=False):
    if norm:
        df = df.apply(lambda s: minmax_norm(s), axis=0)
    range_lists = []

    pm_modes = pm_modes[s:e]
    for col in pm_modes.columns:
        range_list = []
        prev_val = False
        for inx, val in pm_modes[col].iteritems():
            if prev_val != val:
                if val:
                    start = inx
                else:
                    range_list.append((start, inx))
            prev_val = val
        range_lists.append(range_list)

    fig, axes = plt.subplots(len(df.columns), 1, figsize=(15, len(df.columns) + 3), sharex=True)
    time = np.arange(len(date))
    print(df.columns)

    for i, (element, range_list) in enumerate(zip(df.columns, range_lists)):
        if "ACNTS" in element:
            element_ = element[10:]
            axes[i].set_title(element_, fontsize=15)
        else:
            axes[i].set_title(element, fontsize=15)
        if i == 0: axes[i].set_ylim(0, 1)
        if sampling_mode == 0:
            axes[i].plot(time[s:e], df[element].values[s:e], color="black")
        else:
            axes[i].plot(time[s:e], df[element].values[s:e], 'bo', markersize=3.0)
        # axes[i].plot(deleted[i], y_hat[i], 'g', markersize=0.7, alpha=0.5)
        for (start, end) in range_list:
            axes[i].axvspan(start, end, color='red', alpha=0.2)

    fig.suptitle(case, fontsize=25, weight='bold')
    fig.tight_layout()
    make_dirs(f"../results/plots/task3/timeseries_plot/{fn}")
    plt.savefig(f"../results/plots/task3/timeseries_plot/{fn}/{case}_start_{s}_end_{e}_sampling_mode_{sampling_mode}_norm_{norm}.png", dpi=1200)
    plt.show()


def plot_gps(train_x, train_y, inference_x, means, lowers, uppers, fn, case, start, end, interpolation_mode):

    # This contains predictions for both tasks, flattened out
    # The first half of the predictions is for the first dimension and the second half is for the second dimension.
    f, y_axs = plt.subplots(train_y.shape[1], 1, figsize=(40, 7*train_y.shape[1]))
    for dim in range(train_y.shape[1]):
        # Plot training data as black stars
        y_axs[dim].plot(train_x, train_y[:, dim], 'k*', color="r", markersize=1.8)
        # Predictive mean as blue line
        y_axs[dim].plot(np.arange(means.shape[0]), means[:, dim], 'green', linewidth=0.5)
        # Shade in confidence
        y_axs[dim].fill_between(inference_x.numpy(), lowers[:, dim], uppers[:, dim], alpha=0.5)
        # y_axs[dim].set_ylim([-4, 8])
        y_axs[dim].legend(['Observed Data', 'Mean', 'Confidence'], fontsize=30)
        y_axs[dim].set_title('Observed Values (Likelihood)', fontsize=25)
        y_axs[dim].set_xticks([])

    f.tight_layout()
    plt.show()
    # plt.savefig(
    #         f"../results/plots/task3/timeseries_plot/{fn}/{case}_start_{start}_end_{end}_interpolation_mode_{interpolation_mode}_Interpolation_using_RBF_vs_GP.png",
    #         dpi=1200)


def plot_interpolants(df, train_x, train_y, true_x, true_y, elements, interpolants_rbf, interpolants_simple_gp, fn, case, start, end, interpolation_mode):
    train_x_ind = np.argwhere((train_x >= start) & (train_x <= end)).flatten()
    x = np.argwhere((true_x >= start) & (true_x <= end))
    interpolants_rbf_ = interpolants_rbf.copy()
    interpolants_simple_gp_ = interpolants_simple_gp.copy()
    interpolants_simple_gp[~np.isin(np.arange(interpolants_simple_gp.shape[0]), true_x)] = np.nan
    interpolants_rbf[~np.isin(np.arange(interpolants_rbf.shape[0]), true_x)] = np.nan

    fig, axes = plt.subplots(train_y.shape[1], 1, figsize=(20, train_y.shape[1]*6 + 1), sharex=True)

    for i, (dim, element) in enumerate(zip(range(train_y.shape[1]), elements)):
        axes[i].set_title(element, fontsize=15)
        axes[i].plot(true_x[x], true_y[x, dim], 'k*', color="r", markersize=10)
        axes[i].plot(train_x[train_x_ind], train_y[train_x_ind, dim], marker='x', color="black", markersize=10)
        axes[i].plot(df[start:end].index.to_numpy(), interpolants_simple_gp[start:end, dim], 'bo', color="green", markersize=5.0)
        axes[i].plot(df[start:end].index.to_numpy(), interpolants_rbf[start:end, dim], 'bo', color="blue", markersize=5.0)
        axes[i].plot(df[start:end].index.to_numpy(), interpolants_simple_gp_[start:end, dim], "--", color="green",
                 linewidth=1.5)
        axes[i].plot(df[start:end].index.to_numpy(), interpolants_rbf_[start:end, dim], "--", color="blue", linewidth=1.5)
        if i == 0:
            axes[i].legend(['True y', 'Observation', 'Interpolated_y(Simple GP interpolation)',
                    'Interpolated_y(Similarity Function(RBF))'], fontsize=18)

    fig.suptitle('Interpolation using RBF vs GP', fontsize=20)
    fig.tight_layout()
    plt.savefig(
        f"../results/plots/task3/timeseries_plot/{fn}/{case}_start_{start}_end_{end}_interpolation_mode_{interpolation_mode}_Interpolation_using_RBF_vs_GP.png",
        dpi=1200)
    plt.show()


class Dataset(TensorDataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


def data_loader(x, y, train_split, test_split, batch_size, oversampling):
    # Split to Train, Validation and Test Set #
    train_seq, test_seq, train_label, test_label = train_test_split(x, y, train_size=train_split, shuffle=True, random_state=77, stratify=y)
    val_seq, test_seq, val_label, test_label = train_test_split(test_seq, test_label, train_size=test_split, shuffle=True, random_state=77, stratify=test_label)

    if oversampling:
        sm = SMOTE(sampling_strategy="auto", k_neighbors=2)
        # indice = train_seq[:, np.arange(train_seq.shape[1]) == 0]
        train_seq, train_label = sm.fit_resample(train_seq[:, np.arange(train_seq.shape[1]) != 0], train_label)
        print('After OverSampling, the shape of train_X: {}'.format(train_seq.shape))
        print('After OverSampling, the shape of train_y: {} \n'.format(train_label.shape))

    # Expand Dims
    train_seq = np.expand_dims(train_seq, 1).astype(np.float32)
    val_seq = np.expand_dims(val_seq, 1).astype(np.float32)
    test_seq = np.expand_dims(test_seq, 1).astype(np.float32)

    # Convert to Tensor #
    train_set = Dataset(train_seq, train_label)
    val_set = Dataset(val_seq, val_label)
    test_set = Dataset(test_seq, test_label)

    # Data Loader #
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def get_lr_scheduler(lr_scheduler, optimizer):
    """Learning Rate Scheduler"""
    if lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    else:
        raise NotImplementedError
    return scheduler


def percentage_error(actual, predicted):
    """Percentage Error"""
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def mean_square_error(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2, axis=0)


def mean_percentage_error(y_true, y_pred):
    """Mean Percentage Error"""
    return np.mean(percentage_error(np.asarray(y_true), np.asarray(y_pred))) * 100


def mean_absolute_percentage_error(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100


def plot_lower_embedded_2d(x, y, path):
    model = TSNE(learning_rate=1000)
    labels = y
    transformed = model.fit_transform(x)

    fig, ax = plt.subplots()
    plt.title("Lower embedding with TSNE")
    groups = pd.DataFrame(transformed, columns=['x', 'y']).assign(category=labels).groupby('category')
    for name, points in groups:
        ax.scatter(points.x, points.y, label=name)

    ax.legend()
    plt.savefig(path)
    plt.show()


def draw_confusion_matrix(pred_y, true_y, model_name, save_path, task, chroma=None, word2index = None):
    classes = np.unique(true_y)
    if task == "task1":
        plt.figure(figsize=(3, 3))
        plt.rc('font', size=11)
        cla = ["Normal", "Abnormal"] # [0, 1]
    else:
        plt.figure(figsize=(len(classes), len(classes)))
        plt.rc('font', size=15)
        cla = []
        for c in np.sort(classes):
            for key, case in word2index.items():
                if c == case:
                    cla.append(key)
    tick_marks = np.arange(len(classes))
    cm = confusion_matrix(true_y, pred_y)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix")
    plt.colorbar()
    plt.xticks(tick_marks, cla, rotation=45)
    plt.yticks(tick_marks, cla, rotation=45)
    thresh = cm.max()/1.2
    normalize = False
    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path + f'/{model_name}_cm_{chroma}.png', dpi=100, bbox_inches='tight')  # 그림 저장
    # plt.show()
    print(classification_report(true_y, pred_y))


def draw_roc(pred_y, true_y, model_name, save_path):
    fpr, tpr, thr = roc_curve(true_y, pred_y)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC'.format(model_name))
    plt.legend(loc="lower right")
    plt.ion()
    plt.tight_layout()
    plt.savefig(save_path + '{}_ROC.png'.format(model_name), dpi=100, bbox_inches='tight')  # 그림 저장
    plt.show()


def one_hot_encoding(w: str, word2index):
    one_hot_vector = np.zeros(len(word2index), dtype=np.int64)
    index = word2index[w]
    one_hot_vector[index] = 1
    return one_hot_vector


def label_encoding(w: str, word2index):
    return word2index[w]


def merge_dataframe(chroma: str, sheet_names: list):
    for i, sheet_name in enumerate(sheet_names):
        df = pd.read_csv(f'../data/{chroma}/{sheet_name}.csv', engine="python")
        df["equipment_code"] = sheet_name
        df_merge = df if i < 1 else pd.concat([df_merge, df], ignore_index=True)
    return df_merge


def get_irregularly_sampled_data(df_, sampling_mode):
    if sampling_mode in [2, 4]:
        differentials = np.diff(df_.values.astype(float), axis=0)
        if sampling_mode == 2:
            differentials = differentials.sum(axis=1)
        alert = 0
        states = [alert]
        tobedeleted = []
        for r_idx, val in enumerate(differentials):
            if val != 0.0:
                alert += 1
                if alert >= 2:
                    tobedeleted.append(r_idx)
            else:
                alert = 0
            states.append(alert)
        y_tobedeleted = df_.iloc[tobedeleted].copy()
        df_.iloc[tobedeleted] = np.NaN
    df_drop_duplicated = df_.dropna().drop_duplicates()
    df_.iloc[~df_.index.isin(df_drop_duplicated.index)] = np.NaN
    deleted = df_.iloc[~df_.index.isin(df_drop_duplicated.index)].index.tolist()
    if sampling_mode in [1, 3]:
        return df_, deleted
    else:
        return df_, deleted, y_tobedeleted


def minmax_norm(x):
    return (x - x.min()) / x.max() - x.min()


file_names = {"IPA":
                   ['C01_IPA1(1hr)', 'E01_IPA1(1hr)', 'E01_IPA2(8hr)', 'Y01_IPA1(1hr)', 'Y02_IPA(1hr)', 'G01_IPA1(1hr)',
                    'G01_IPA2(8hr)', 'G02_IPA1(1hr)', 'H01_IPA(1hr)', 'H02_IPA1(1hr)', 'HJ01_IPA1(1hr)','HJ01_IPA2(1hr)', 'J02_IPA1(1hr)', 'J02_IPA2(1hr)', 'N01_IPA1(1hr)', 'N01_IPA2(1hr)'],
              "HF":
                   ['C01_HF1(1hr)', 'E01_HF1(1hr)', 'Y01_HF1(1hr)', 'Y02_HF(1hr)', 'G01_HF1(1hr)', 'G02_HF1(1hr)',
                    'H01_HF(1hr)', 'H02_HF1(1hr)', 'HJ01_HF1(1hr)', 'HJ01_HF2(1hr)', 'J02_HF1(1hr)', 'J02_HF2(1hr)', 'N01_HF1(1hr)', 'N01_HF2(1hr)'],
              "17L_미세변경(Case1-12)_산학":
                  [f"Case{i+1}" for i in range(12)],
              "17L_미세변경(Case13-24)_산학":
                  [f"Case{i+13}" for i in range(12)],
              "미세변경Dataset2(Case1-12)":
                  [f"Case{i+1}" for i in range(19)],
              "미세변경Dataset2(Case20-42)":
                  [f"Case{i+20}" for i in range(23)]
              }

equipments = {
              "17L_미세변경(Case1-12)_산학":
                  {
                      "Cases": [f"Case{i+1}" for i in range(12)],
                      "equipments": [
                          ["|ACNTS_09|03Bay_P05", "|ACNTS_09|02Bay_P06"],
                          ["|ACNTS_09|06_SA_P07", "|ACNTS_09|06Bay_P06", "|ACNTS_09|05Bay_P01", "|ACNTS_09|03Bay_P05", "|ACNTS_09|04Bay_P03", "|ACNTS_09|04_SA_P04", "|ACNTS_09|05_SA_P02"],
                          ["|ACNTS_09|06_SA_P07", ],
                      ]

                  },
              "17L_미세변경(Case13-24)_산학":
                  [f"Case{i+13}" for i in range(12)],
              "미세변경Dataset2(Case1-12)":
                  [f"Case{i+1}" for i in range(19)],
              "미세변경Dataset2(Case20-42)":
                  [f"Case{i+20}" for i in range(23)]
              }


cases = {np.nan: ("Ordinary", 0),
         "": ("Ordinary", 0),
         "정상": ("Ordinary", 0),
         "진성오염": ("Normal_Pollution", 1),
         "헌팅": ("Hunting", 2),
         "시료미주입": ("Hunting", 2),
         "염생성": ("Cl", 3),
         "염생성,시료미주입": ("Cl", 3),
         "염생성,MS이상(Torch)": ("Cl", 3),
         "질산오염": ("NO3", 4),
         "질산주입이상": ("NO3", 4),
         "파츠성(Loop)": ("Loop", 5),
         "파츠성(Cone)": ("Cone", 6),
         "파츠성(Valve)": ("Valve", 7),
         "MS이상(V12Leak)": ("Valve", 7),
         "파츠성(Syringe)": ("Syringe", 8),
         "MS이상": ("MS", 9),
         "MS이상(Pump)": ("MS", 9),
         "MS이상(PlasmaOff)": ("MS", 9),
         "MS이상(PlasmaOff후)": ("MS", 9),
         "MS이상(Torch)": ("MS", 9),
         "MS이상(Torch),Tune": ("MS", 9),
         "Tune": ("Tune", 10),
         "CCT": ("CCT", 11),
         "파츠성(Syringe,Valve)": ("Syringe & Valve", 12),
         "파츠성(Cone,Loop)": ("Cone & Loop", 13),
         "파츠성(Syringe,Loop)": ("Syringe & Loop", 14),
         "파츠성(Syringe,Cone)": ("Syringe & Cone", 15),
         "파츠성(Loop),Tune": ("Loop & Tune", 16),
         "헌팅,파츠성(Loop)": ("Hunting & Loop", 17),
         "파츠성(Cone),Tune": ("Cone & Tune", 18),
         "Tune,파츠성(Cone)": ("Cone & Tune", 18),
         "파츠성(Syringe),Tune": ("Syringe & Tune", 19),
         "Tune,파츠성(Syringe)": ("Syringe & Tune", 19),
         "파츠성(Syringe),염생성": ("Syringe & Cl", 20),
         "파츠성(Syringe,Loop),Tune": ("Syringe & Loop & Tune", 21),
         "파츠성(Syringe,Valve),Tune": ("Syringe & Valve & Tune", 22),
         "염생성,Tune": ("Cl & Tune", 23),
         "MS이상(Pump),파츠성(Loop)": ("deleted", 24),
         "가성": ("deleted", 24)
         }

# The CNP takes as input a `CNPRegressionDescription` namedtuple with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tensor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that describes the total
#     number of data points used (context + target)
#   `num_context_points`: A vector containing a scalar that describes the number
#     of data points used as context

# The GPCurvesReader returns the newly sampled data in this format at each
# iteration

CNPRegressionDescription = collections.namedtuple(
    "CNPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"))


# class GPCurvesReader(object):
#     """ Generates curves using a Gaussian Process (GP).
#     Supports vector inputs (x) and vector outputs (y). Kernel is
#     mean-squared exponential, using the x-value l2 coordinate distance scaled by
#     some factor chosen randomly in a range. Outputs are independent gaussian processes.
#     """
#
#     def __init__(self, batch_size, max_num_context, x_size=1, y_size=1, l1_scale=0.4, sigma_scale=1.0, testing=False):
#         """
#         Creates a regression dataset of functions sampled from a GP.
#
#         Args:
#             batch_size: An integer.
#             max_num_context: The max number of observations in the context.
#             x_size: Integer >= 1 for length of "x values" vector.
#             y_size: Integer >= 1 for length of "y values" vector.
#             l1_scale: Float; typical scale for kernel distance function.
#             sigma_scale: Float; typical scale for variance.
#             testing: Boolean that indicates whether we are testing. If so there are more targets for visualization.
#         """
#         self._batch_size = batch_size
#         self._max_num_context = max_num_context
#         self._x_size = x_size
#         self._y_size = y_size
#         self._l1_scale = l1_scale
#         self._sigma_scale = sigma_scale
#         self._testing = testing
#
#     def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
#         """Applies the Gaussian kernel to generate curve data.
#
#         Args:
#               xdata: Tensor with shape `[batch_size, num_total_points, x_size]` with
#                   the values of the x-axis data.
#               l1: Tensor with shape `[batch_size, y_size, x_size]`, the scale
#                   parameter of the Gaussian kernel.
#               sigma_f: Float tensor with shape `[batch_size, y_size]`; the magnitude
#                   of the std.
#               sigma_noise: Float, std of the noise that we add for stability.
#
#         Returns:
#               The kernel, a float tensor with shape
#               `[batch_size, y_size, num_total_points, num_total_points]`.
#         """
#         num_total_points = tf.shape(xdata)[1]
#
#         # Expand and take the difference
#         xdata1 = tf.expand_dims(xdata, axis=1)  # [B, 1, num_total_points, x_size]
#         xdata2 = tf.expand_dims(xdata, axis=2)  # [B, num_total_points, 1, x_size]
#         diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]
#
#         # [B, y_size, num_total_points, num_total_points, x_size]
#         norm = tf.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])
#
#         norm = tf.reduce_sum(norm, -1)  # [B, data_size, num_total_points, num_total_points]
#
#         # [B, y_size, num_total_points, num_total_points]
#         kernel = tf.square(sigma_f)[:, :, None, None] * tf.exp(-0.5 * norm)
#
#         # Add some noise to the diagonal to make the cholesky work.
#         kernel += (sigma_noise ** 2) * tf.eye(num_total_points)
#
#         return kernel
#
#     def generate_curves(self):
#         """Builds the op delivering the data.
#
#         Generated functions are `float32` with x values between -2 and 2.
#
#         Returns:
#           A `CNPRegressionDescription` namedtuple.
#         """
#         num_context = tf.random_uniform(shape=[], minval=3, maxval=self._max_num_context, dtype=tf.int32)
#
#         # If we are testing we want to have more targets and have them evenly
#         # distributed in order to plot the function.
#         if self._testing:
#             num_target = 400
#             num_total_points = num_target
#             x_values = tf.tile(tf.expand_dims(tf.range(-4., 4., 1. / 50, dtype=tf.float32), axis=0), [self._batch_size, 1])
#             x_values = tf.expand_dims(x_values, axis=-1)
#             print("x_values_test:", x_values.shape)
#             quit()
#         # During training the number of target points and their x-positions are selected at random
#         else:
#             num_target = tf.random_uniform(shape=(), minval=2, maxval=self._max_num_context, dtype=tf.int32)
#             num_total_points = num_context + num_target
#             # print("num_target_training:", tf.Session().run(num_target))
#             # print("num_context_training:", tf.Session().run(num_context))
#             # print("num_total_points_training:", tf.Session().run(num_total_points))
#             x_values = tf.random_uniform([self._batch_size, num_total_points, self._x_size], -4, 4)
#             print("x_values_training:", x_values.shape)
#
#         # Set kernel parameters
#         l1 = (tf.ones(shape=[self._batch_size, self._y_size, self._x_size]) * self._l1_scale)
#         sigma_f = tf.ones(shape=[self._batch_size, self._y_size]) * self._sigma_scale
#
#         # Pass the x_values through the Gaussian kernel
#         # [batch_size, y_size, num_total_points, num_total_points]
#         kernel = self._gaussian_kernel(x_values, l1, sigma_f)
#
#         # Calculate Cholesky, using double precision for better stability:
#         cholesky = tf.cast(tf.cholesky(tf.cast(kernel, tf.float64)), tf.float32)
#
#         # Sample a curve
#         # [batch_size, y_size, num_total_points, 1]
#         y_values = tf.matmul(
#             cholesky,
#             tf.random_normal([self._batch_size, self._y_size, num_total_points, 1]))
#
#         # [batch_size, num_total_points, y_size]
#         y_values = tf.transpose(tf.squeeze(y_values, 3), [0, 2, 1])
#         # print("y_values:", y_values.shape)
#         # print("y_values:", tf.Session().run(y_values))
#
#         if self._testing:
#             # Select the targets
#             target_x = x_values
#             target_y = y_values
#
#             # Select the observations
#             idx = tf.random_shuffle(tf.range(num_target))
#             context_x = tf.gather(x_values, idx[:num_context], axis=1)
#             context_y = tf.gather(y_values, idx[:num_context], axis=1)
#
#         else:
#             # Select the targets which will consist of the context points as well as
#             # some new target points
#             print("num_target:", tf.Session().run(num_target))
#             print("num_context:", tf.Session().run(num_context))
#             print("num_context+num_target:", tf.Session().run(num_target + num_context))
#
#             target_x = x_values[:, :num_target + num_context, :]
#             target_y = y_values[:, :num_target + num_context, :]
#
#             # Select the observations
#             context_x = x_values[:, :num_context, :]
#             context_y = y_values[:, :num_context, :]
#
#         query = ((context_x, context_y), target_x)
#
#         return CNPRegressionDescription(
#             query=query,
#             target_y=target_y,
#             num_total_points=tf.shape(target_x)[1],
#             num_context_points=num_context)


def plot_functions(target_x, target_y, context_x, context_y, pred_y, var):
    """Plots the predicted mean and variance and the context points.

    Args:
      target_x: An array of shape batch size x number_targets x 1 that contains the
          x values of the target points.
      target_y: An array of shape batch size x number_targets x 1 that contains the
          y values of the target points.
      context_x: An array of shape batch size x number_context x 1 that contains
          the x values of the context points.
      context_y: An array of shape batch size x number_context x 1 that contains
          the y values of the context points.
      pred_y: An array of shape batch size x number_targets x 1  that contains the
          predicted means of the y values at the target points in target_x.
      pred_y: An array of shape batch size x number_targets x 1  that contains the
          predicted variance of the y values at the target points in target_x.
    """
    # Plot everything
    plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
    plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
    plt.plot(context_x[0], context_y[0], 'ko', markersize=10)
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - var[0, :, 0],
        pred_y[0, :, 0] + var[0, :, 0],
        alpha=0.2,
        facecolor='#65c9f7',
        interpolate=True)

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-4, 0, 4], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid('off')
    ax = plt.gca()
    ax.set_facecolor('white')
    #   ax.set_axis_bgcolor('white')
    plt.show()

