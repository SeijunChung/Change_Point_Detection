import copy
import os
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import torch
import torch.nn as nn
from models import DNN1, DNN2, DNN3, CNN, LogisticRegression, const_prior, fullcov_obs_log_likelihood, online_changepoint_detection, constant_hazard, MultivariateT, StudentT, BOCD
from utils import make_dirs, data_loader, get_lr_scheduler, draw_confusion_matrix, draw_roc, merge_dataframe, label_encoding, file_names, cases, plot_timeseries, plot_not_hits
from preprocess import get_data, get_new_features, z_score_normalization
from functools import partial

from kats.consts import TimeSeriesData
import logging
from imp import reload
reload(logging)
from kats.detectors.robust_stat_detection import RobustStatDetector

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# For Reproducibility #
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device Configuration #
device = 'cuda:7' if torch.cuda.is_available() else 'cpu'


def task1(args):
    # Fix Seed #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Make Path for Data, Weights and Plots #
    paths = [p+f"/{args.task}" for p in [args.data_path, args.weights_path, args.plots_path]]
    [args.data_path, args.weights_path, args.plots_path] = paths

    for path in paths:
        make_dirs(path)

    # Prepare Data #
    df, observations = get_data(args.task, args.data_path)

    # Plot Data in 2D space #
    # Turn on the code lines below if you want to plot the samples in 2D space
    # plot_lower_embedded_2d(observations, df["label"], args.plots_path+f"/{args.task}")

    # Z-Normalization
    observations_norm = z_score_normalization(observations, axis=1)

    if args.add_new_features:
        df_copy = copy.deepcopy(df)
        df_new_features = get_new_features(df_copy, observations)
        print(df_new_features.columns)
        # print(df_new_features[["rt_area", "rt_area_norm", "recovery", "difference"]])
        # print(df_new_features["rt_area"].min(axis=0), df_new_features["rt_area"].max(axis=0))
        # print(df_new_features["rt_area_norm"].min(axis=0), df_new_features["rt_area_norm"].max(axis=0))
        X = np.hstack((df_new_features[["rt_area_norm", "recovery", "increase_in_rt_norm"]], observations_norm))
    else:
        X = observations_norm

    y = df["label"].values.astype(np.float32)

    # plot samples (w/ newly added features)
    # Turn on the code lines below if you want to plot the samples

    # df_concat = pd.concat([df, new_features], axis=1)
    # print(df_concat.columns)
    # for i in df_concat.index:
    #     val = df_concat.loc[i, ["RT_box", "Analysis Number", "Baseline_start", "Baseline_end", "label"]]
    #     s = int(val["RT_box"][0] * 250)
    #     e = int(val["RT_box"][1] * 250)
    #     x, peaks, diff, multi_peak_bool = peaks_in_rt(observations[i, :], s, e, i)
    #     area = get_area_in_RTbox(observations[i, :], s, e)
    #     # area_normalized = get_area_in_RTbox(observations[i, :] / max(observations[i, :]), s, e)
    #     plot_peaks_in_rt_box(observations[i, :],
    #                          i,
    #                          val["Analysis Number"],
    #                          val["Baseline_start"],
    #                          val["Baseline_end"],
    #                          val["label"],
    #                          x,
    #                          area,
    #                          int(val["RT_box"][0] * 250),
    #                          int(val["RT_box"][1] * 250))

    print(sorted(Counter(y).items()))
    y = np.expand_dims(y, 1)

    # add index
    X = np.hstack((np.arange(X.shape[0]).reshape(-1, 1), X))
    print(f"X shape: {X.shape} \t y shape: {y.shape}")

    train_loader, val_loader, test_loader = data_loader(X, y, args.train_split, args.test_split, args.batch_size, args.oversampling)
    # Loss lists #
    train_losses, val_losses, test_losses, not_correct_samples = list(), list(), list(), list()

    # Constants #
    best_val_loss = 100
    best_val_improv = 0

    # Prepare Network #
    if args.model == 'DNN':
        model = DNN1(args.input_size, args.hidden_size, args.output_size).to(device)
    elif args.model == 'LR':
        model = LogisticRegression(args.input_size, args.output_size).to(device)
    else:
        raise NotImplementedError

    # Loss Function #
    criterion = torch.nn.BCELoss()

    # Optimizer #
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_scheduler = get_lr_scheduler(args.lr_scheduler, optim)

    # Train and Validation #
    if args.mode == 'train':

        # Train #
        print(f"Training {model.__class__.__name__} started with total epoch of {args.num_epochs}.")

        for epoch in range(args.num_epochs):
            model.train()
            for i, (data, label) in enumerate(train_loader):
                # Prepare Data #
                data = data.to(device, dtype=torch.float32)
                # indice = data[..., torch.arange(data.size(2)) == 0].flatten().cpu().numpy().astype(np.int16)
                data = data[..., torch.arange(data.size(2)) != 0]
                label = label.to(device, dtype=torch.float32)

                # Forward Data #
                pred = model(data)

                # Calculate Loss #
                train_loss = criterion(pred, label)

                # Initialize Optimizer, Back Propagation and Update #
                optim.zero_grad()
                train_loss.backward()
                optim.step()

                # Add item to Lists #
                train_losses.append(train_loss.item())

            # Print Statistics #
            if (epoch + 1) % args.print_every == 0:
                print("Epoch [{}/{}]".format(epoch + 1, args.num_epochs))
                print(f"{args.mode} Loss {np.average(train_losses):.4f}")

            # Learning Rate Scheduler #
            optim_scheduler.step()

            # Validation #
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (data, label) in enumerate(val_loader):
                    # Prepare Data #
                    data = data.to(device, dtype=torch.float32)
                    indice = data[..., torch.arange(data.size(2)) == 0].flatten().cpu().numpy().astype(np.int16)
                    data = data[..., torch.arange(data.size(2)) != 0]
                    label = label.to(device, dtype=torch.float32)

                    # Forward Data #
                    output = model(data)

                    # Calculate Loss #
                    val_loss = criterion(output, label)
                    val_losses.append(val_loss.item())

                    pred_val = torch.where(output > 0.5,
                                           torch.tensor(1).to(device, dtype=torch.float32),
                                           torch.tensor(0).to(device, dtype=torch.float32))

                    total += label.size(0)
                    correct += pred_val.eq(label).sum().item()

            if (epoch + 1) % args.print_every == 0:

                # Print Statistic evaluation #
                print(f"Validation Loss {np.average(val_losses):.4f}")
                print(f"Validation Acc. {100. * correct / total:.3f} (%d/%d)" % (correct, total))

                # Save the model only if validation loss decreased #
                curr_val_loss = np.average(val_losses)

                if curr_val_loss < best_val_loss:
                    best_val_loss = min(curr_val_loss, best_val_loss)
                    torch.save(model.state_dict(),
                               os.path.join(args.weights_path, f'Best_{model.__class__.__name__}_model.pkl'))
                    print("Best model is saved!\n")
                    best_val_improv = 0

                elif curr_val_loss >= best_val_loss:
                    best_val_improv += args.print_every
                    print("Best Validation has not improved for {} epochs.\n".format(best_val_improv))

    elif args.mode == 'test':
        # Load the Model Weight #
        model.load_state_dict(torch.load(os.path.join(args.weights_path, f'Best_{model.__class__.__name__}_model.pkl')))

        # Test #
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (data, label) in enumerate(test_loader):

                # Prepare Data #
                data = data.to(device, dtype=torch.float32)
                indice = data[..., torch.arange(data.size(2)) == 0].flatten().cpu().numpy().astype(np.int16)
                data = data[..., torch.arange(data.size(2)) != 0]
                label = label.to(device, dtype=torch.float32)

                # Forward Data #
                output = model(data)

                # Calculate Loss #
                test_loss = criterion(output, label)
                test_losses.append(test_loss.item())

                # Select the max value #
                pred_test = torch.where(output > 0.5,
                                        torch.tensor(1).to(device, dtype=torch.float32),
                                        torch.tensor(0).to(device, dtype=torch.float32))

                total += label.size(0)
                correct += pred_test.eq(label).sum().item()

                if i == 0:
                    pred_y = pred_test.cpu().numpy()
                    true_y = label.cpu().numpy()
                else:
                    pred_y = np.append(pred_y, pred_test.cpu().numpy())
                    true_y = np.append(true_y, label.cpu().numpy())
                not_correct_samples.extend([indice[fn] for fn in torch.where(~pred_test.eq(label))[0].cpu().numpy()])

            # Print Statistics #
            print(f'{args.mode} Loss: %.3f | {args.mode} Acc: %.3f%% (%d/%d)' % (sum(test_losses) / len(test_losses),
                                                                                 100. * correct / total,
                                                                                 correct, total))

            # Plot Figures #
            draw_confusion_matrix(pred_y, true_y, args.model, args.plots_path, args.task)
            draw_roc(pred_y, true_y, args.model, args.plots_path)

            print("The samples that Model could not hit:", not_correct_samples)

            # Turn on the code lines below if you want to plot the samples that Model could not hit
            # for ind in not_correct_samples:
            #     val = df.loc[ind, ["RT_box", "Analysis Number", "Baseline_start", "Baseline_end", "label"]]
            #     s = int(val["RT_box"][0] * 250)
            #     e = int(val["RT_box"][1] * 250)
            #     x, peaks, diff, multi_peak_bool = peaks_in_rt(observations[ind, :], s, e, ind)
            #     area = get_area_in_RTbox(observations[ind, :], s, e)
            #     plot_peaks_in_rt_box(observations[ind, :],
            #                          ind,
            #                          val["Analysis Number"],
            #                          val["Baseline_start"],
            #                          val["Baseline_end"],
            #                          val["label"],
            #                          x,
            #                          area,
            #                          int(val["RT_box"][0] * 250),
            #                          int(val["RT_box"][1] * 250))

    else:
        raise NotImplementedError


def task2(args):
    # Fix Seed #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Weights and Plots Path #
    paths = [args.weights_path, args.plots_path, args.data_path]
    for path in paths:
        make_dirs(path + f"/{args.task}")

    # Prepare Data #
    if args.preprocessing:
        for i, sheet_name in enumerate(file_names[args.chroma]):
            print(f"{sheet_name} is under preprocessing")
            df_temp = get_data(args.task, args.data_path, sheet_name, args.chroma)
            df_temp["equipment_code"] = sheet_name
            df = df_temp if i < 1 else pd.concat([df, df_temp], ignore_index=True)
    else:
        i = 0 if args.chroma == "IPA" else 1
        df = merge_dataframe(list(file_names.keys())[i], list(file_names.values())[i])

    del cases["MS이상(Pump),파츠성(Loop)"]
    del cases["가성"]

    original_ind = df.groupby(args.objective).filter(lambda x: len(x) >= args.threshold_num_samples).index
    X = df.iloc[original_ind][df.columns.difference(['class', 'time', 'root_cause', 'equipment_code'])].values

    w2index = dict()
    for case in df.iloc[original_ind][args.objective].unique():
        if case not in w2index.keys():
            w2index[case] = len(w2index)

    y = np.stack(df.iloc[original_ind][args.objective].map(lambda x: label_encoding(x, w2index)))

    print(f"the number of features: {len(sorted(Counter(y).items()))} ~> {[pair for pair in w2index.items()]}")

    # Z-Normalization
    X_norm = np.divide((X - np.mean(X, axis=0)), np.std(X, axis=0))

    # add index
    X = np.hstack((np.arange(len(original_ind)).reshape(-1, 1), X_norm))
    print(f"X shape: {X.shape} \t y shape: {y.shape}")

    # Train / Validation / Test Split
    train_loader, val_loader, test_loader = data_loader(X, y, args.train_split, args.test_split, args.batch_size, args.oversampling)

    # Loss lists #
    train_losses, val_losses, test_losses, not_correct_samples = list(), list(), list(), list()
    not_correct_samples_pred_y, not_correct_samples_true_y = list(), list()

    # Constants #
    best_val_loss = 100
    best_val_improv = 0

    # Prepare Network #
    if args.model == 'DNN':
        if args.chroma == "HF":
            model = DNN2(args.input_size, args.hidden_size, args.output_size).to(device)
        elif args.chroma == "IPA":
            model = DNN3(args.input_size, args.hidden_size, args.output_size).to(device)
    elif args.model == 'CNN':
        model = CNN(args.seq_length, args.hidden_size, args.input_size, args.output_size).to(device)
    elif args.model == 'LR':
        model = LogisticRegression(args.input_size, args.output_size).to(device)
    else:
        raise NotImplementedError

    # Loss Function #
    criterion = nn.CrossEntropyLoss().to(device) if args.objective == "root_cause" else nn.BCELoss().to(device)

    # Optimizer #
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_scheduler = get_lr_scheduler(args.lr_scheduler, optim)

    # Train and Validation #
    if args.mode == 'train':

        # Train #
        print(f"Training {model.__class__.__name__} started with total epoch of {args.num_epochs}.")

        for epoch in range(args.num_epochs):
            model.train()
            for i, (data, label) in enumerate(train_loader):
                # Prepare Data #
                data = data.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.int64)

                # Forward Data #
                hypothesis = model(data)

                # Calculate Loss #
                train_loss = criterion(hypothesis, label)

                # Initialize Optimizer, Back Propagation and Update #
                optim.zero_grad()
                train_loss.backward()
                optim.step()

                # Add item to Lists #
                train_losses.append(train_loss.item())

            # Print Statistics #
            if (epoch + 1) % args.print_every == 0:
                print("Epoch [{}/{}]".format(epoch + 1, args.num_epochs))
                print(f"{args.mode} Loss {np.average(train_losses):.4f}")

            # Learning Rate Scheduler #
            optim_scheduler.step()

            # Validation #
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (data, label) in enumerate(val_loader):
                    # Prepare Data #
                    data = data.to(device, dtype=torch.float32)
                    # indice_val = data[..., torch.arange(data.size(2)) == 0].flatten().cpu().numpy().astype(np.int32)
                    data = data[..., torch.arange(data.size(2)) != 0]
                    label = label.to(device, dtype=torch.int64)

                    # Forward Data #
                    output = model(data)
                    # Calculate Loss #
                    val_loss = criterion(output, label)
                    val_losses.append(val_loss.item())

                    # Select the max value #
                    if args.objective == "root_cause":
                        pred = output.data.max(1, keepdim=True)[1]

                    else:
                        pred_val = torch.where(output > 0.5, torch.tensor(1).to(device, dtype=torch.float32),
                                               torch.tensor(0).to(device, dtype=torch.float32))

                    total += label.size(0)
                    correct += pred.eq(label.data.view_as(pred)).sum()

            if (epoch + 1) % args.print_every == 0:

                # Print Statistic evaluation #
                print(f"Validation Loss {np.average(val_losses):.4f}")
                print(f"Validation Acc. {100. * correct / total:.3f} (%d/%d)" % (correct, total))

                # Save the model only if validation loss decreased #
                curr_val_loss = np.average(val_losses)

                if curr_val_loss < best_val_loss:
                    best_val_loss = min(curr_val_loss, best_val_loss)
                    torch.save(model.state_dict(),
                               os.path.join(args.weights_path, f'{args.task}/Best_{model.__class__.__name__}_model_{args.chroma}_num_feature{len(sorted(Counter(y).items()))}_batchsize{args.batch_size}_hiddendim{args.hidden_size}.pkl'))

                    print("Best model is saved!\n")
                    best_val_improv = 0

                elif curr_val_loss >= best_val_loss:
                    best_val_improv += args.print_every
                    print("Best Validation has not improved for {} epochs.\n".format(best_val_improv))

    elif args.mode == 'test':

        # Load the Model Weight #
        model.load_state_dict(torch.load(os.path.join(args.weights_path, f'{args.task}/Best_{model.__class__.__name__}_model_{args.chroma}_num_feature{len(sorted(Counter(y).items()))}_batchsize{args.batch_size}_hiddendim{args.hidden_size}.pkl')))

        # Test #
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (data, label) in enumerate(test_loader):
                # Prepare Data #
                data = data.to(device, dtype=torch.float32)
                indice = data[..., torch.arange(data.size(2)) == 0].flatten().cpu().numpy().astype(np.int32)
                data = data[..., torch.arange(data.size(2)) != 0]
                label = label.to(device, dtype=torch.int64)

                # Forward Data #
                output = model(data)

                # Calculate Loss #
                test_loss = criterion(output, label)
                test_losses.append(test_loss.item())

                # Select the max value #
                if args.objective == "root_cause":
                    pred_test = output.data.max(1, keepdim=True)[1]
                else:
                    pred_test = torch.where(output > 0.5,
                                            torch.tensor(1).to(device, dtype=torch.float32),
                                            torch.tensor(0).to(device, dtype=torch.float32))

                total += label.size(0)
                # correct += pred_test.eq(label).sum().item()
                correct += pred_test.eq(label.data.view_as(pred_test)).sum()
                not_correct_samples_pred_y.extend([pred_test.cpu().numpy().flatten()[k] for k in
                                                   torch.where(~pred_test.eq(label.data.view_as(pred_test)))[
                                                       0].cpu().numpy()])
                not_correct_samples_true_y.extend([label.cpu().numpy()[k] for k in
                                                   torch.where(~pred_test.eq(label.data.view_as(pred_test)))[
                                                       0].cpu().numpy()])

                if i == 0:
                    pred_y = pred_test.cpu().numpy().flatten()
                    true_y = label.cpu().numpy()
                else:
                    pred_y = np.append(pred_y, pred_test.cpu().numpy())
                    true_y = np.append(true_y, label.cpu().numpy())
                not_correct_samples.extend([indice[fn] for fn in torch.where(~pred_test.eq(label.data.view_as(pred_test)))[0].cpu().numpy()])

            # Print Statistics #
            print(f'{args.mode} Loss: %.3f | {args.mode} Acc: %.3f%% (%d/%d)' % (sum(test_losses) / (i + 1),
                                                                                 100. * correct / total,
                                                                                 correct, total))

            # Plot Figures #
            print("The number of samples that Model could not hit:", len(not_correct_samples))
            print("The samples that Model could not hit:", not_correct_samples)
            print("The pred_y that Model could not hit:", not_correct_samples_pred_y)
            print("The true_y that Model could not hit:", not_correct_samples_true_y)

            # draw_confusion_matrix(pred_y, true_y, args.model, args.plots_path, args.task, args.chroma, w2index)

            # Turn on the code lines below if you want to plot the samples that Model could not hit

            print(not_correct_samples)
            k_ = 0
            for k, (i, pred, true) in enumerate(zip(not_correct_samples, not_correct_samples_pred_y, not_correct_samples_true_y)):
                k_ = k_ + k
                plot_not_hits(df, i, k_, original_ind, pred, true, args.chroma, args.plots_path, args.task, w2index)

    else:
        raise NotImplementedError


def task3(args):
    # Fix Seed #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Weights and Plots Path #
    paths = [args.weights_path, args.plots_path]
    for path in paths:
        make_dirs(path + f"/{args.task}")

    # Prepare Data
    fns = [
        "17L_미세변경(Case1-12)_산학",
        # "17L_미세변경(Case13-24)_산학",
        # "미세변경Dataset2(Case1-12)",
        # "미세변경Dataset2(Case20-42)"
    ]
    cov_matrix = dict()
    cps = dict()
    change_points = []

    if args.preprocessing:
        for fn in fns:
            for sheet_name in file_names[fn][2:3]:
                print(f"{fn} ~ {sheet_name} is under preprocessing with sampling mode {args.sampling_mode}")
                data, date, pm_modes, equipments = get_data(args.task, args.data_path, sheet_name, fn, args.sampling_mode, args.interpolation_mode)
                print(equipments)
                equipments.remove('|ACNTS_09|05Bay_P01')
                equipments.remove('|ACNTS_09|04_SA_P04')

                if args.plot_timeseries is True:
                    plot_timeseries(data, date, pm_modes, fn, sheet_name, 0, 100, args.sampling_mode, args.interpolation_mode)
                    # cov_matrix[f"{fn}_{sheet_name}"] = pd.DataFrame(data=np.cov(data.T), columns=data.columns, index=data.columns)
                    # print(cov_matrix[f"{fn}_{sheet_name}"])
                    # with pd.ExcelWriter("cov.xlsx") as writer:
                    #     for key, value in cov_matrix.items():
                    #         value.to_excel(writer, sheet_name=key)

                # equipments = ['|ACNTS_13|17Bay_3_P03', '|ACNTS_13|17Bay_2_P02', '|ACNTS_13|17Bay_1_P01']
                time = data[equipments].index.to_numpy()
                data = data[equipments].values
                data_norm = np.divide((data - np.mean(data, axis=0)), np.std(data, axis=0))
                #start = s #62844 # 10474 # 20948 # 31791, 95113 #74179 # 52985 # 10597 # 31791 # 73977 # 63409 #42388 #84545
                #end = e #73318 # 20948 # 31422 # 42388, 10473, 21136, 105682, 84776, 63582, 21194 # 42388 # 84545 # 21194 73977 #52985 #95113
                height_limit = 5000
                divide = 5
                LAMBDA = 1000
                ALPHA = 0.1
                BETA = 1
                KAPPA = 1.
                MU = 0.
                DELAY = 10
                THRESHOLD = 0.75
                # num_increment = 15
                min_y, max_y = -2, 3
                sparsity = 10  # only plot every fifth data for faster display
                colors = ["red", "blue", "green", "brown", "coral", "brown", "violet", "lightpink", "gold"]

                # for s, e in [(d*time.shape[0]//divide, (d+1)*time.shape[0]//divide) for d in range(divide)]:
                for s, e in [(63000, 67000)]:
                    start = s
                    end = e
                    print(f"{fn} ~ {sheet_name} from start point: {start} to end point: {end}")

                    data = [data_norm[start:end, i] for i in range(data_norm.shape[1])]

                    df = pd.DataFrame(
                        data=np.concatenate((time[start:end].reshape(-1, 1), np.array(data).T), axis=1),
                        columns=["time"] + [f"sensor{i+1}" for i in range(data_norm.shape[1])]
                    )

                    sensors = df.columns.tolist()
                    sensors.pop(0)

                    file = "Dataset1" if "17L" in fn else "Dataset2"
                    timeline = np.arange(0, len(df["time"].values))
                    # height_ratios = [1.5, 1] * len(sensors)
                    fig, axes = plt.subplots(len(sensors) + 1, 1, figsize=[18, 2.5 * (len(sensors)+1)], sharex=True)
                    for sensor_id, (sensor, equipment) in enumerate(zip(sensors, equipments)):

                        uni_ts = TimeSeriesData(df.loc[:, ["time", sensor]])
                        robuststatdetector = RobustStatDetector(uni_ts)
                        robust_cps = robuststatdetector.detector(p_value_cutoff=1e-5, comparison_window=2)
                        robust_change_points = [(cp[1]._index, cp[1]._index) for cp in robust_cps]

                        # Q_full, P_full, Pcp_full = offline_changepoint_detection(df.loc[:, sensor].values,
                        #                                                          partial(const_prior, l=(df.shape[0]+1)),
                        #                                                          fullcov_obs_log_likelihood,
                        #                                                          truncate=-50)
                        # fig, ax = plt.subplots(2, 1, figsize=[18, 10], sharex=True)
                        # ax[0].set_title(f'Time Series {sheet_name}', size=20)
                        # ax[1].set_title('Probability of Change Point', size=20)
                        # ax[0].tick_params(labelsize=15)
                        # ax[1].tick_params(labelsize=15)
                        # for d in range(1):
                        #     ax[0].plot(df.loc[:, sensor].values, label="Sensor_{}".format(d+1))
                        # ax[0].legend(fontsize=15)
                        # ax[1].plot(np.exp(Pcp_full).sum(0), color="red")
                        # fig.tight_layout()
                        # plt.show()

                        import matplotlib.cm as cm
                        r, maxes = online_changepoint_detection(df.loc[:, sensor].values,
                                                                partial(constant_hazard, LAMBDA),
                                                                StudentT(ALPHA, 1, KAPPA, MU)
                                                                )

                        # fig, axes = plt.subplots(3, 1, figsize=[18, 15], sharex=True)
                        # axes[0].plot(df.loc[:, sensor].values)
                        # axes[1].pcolor(np.array(range(0, len(r[:, 0]), sparsity)),
                        #           np.array(range(0, len(r[:, 0]), sparsity)),
                        #           -np.log(r[0:-1:sparsity, 0:-1:sparsity]),
                        #           cmap=cm.Greys, vmin=0, vmax=20, alpha=0.7)
                        # # cps = np.where(r[DELAY, DELAY:-1] > THRESHOLD, 1, r[DELAY, DELAY:-1])
                        # axes[2].plot(r[DELAY, DELAY:-1], color="red")
                        # fig.tight_layout()
                        # # plt.savefig(args.plots_path + f"/{args.task}/cpd_plot/{online_changepoint_detection.__name__}_{sheet_name}_period_{start}_to_{end}.png")
                        # plt.show()

                        bocd = BOCD(partial(constant_hazard, LAMBDA), StudentT(ALPHA, BETA, KAPPA, MU))
                        changepoints = []
                        changepoints_probs = []

                        # print("bocd.t ~>", bocd.t)
                        # print("posterior probablity:", bocd.growth_probs)
                        for x in df.loc[:, sensor].values[:DELAY]:
                            bocd.update(x)
                            # print("bocd.t ~>", bocd.t)
                            # print("posterior probablity:", bocd.growth_probs)

                        for x in df.loc[:, sensor].values[DELAY:]:
                            bocd.update(x)
                            changepoints_probs.append(bocd.growth_probs[DELAY])
                            # print(f"======================={i} th iteration==================================")
                            # print("bocd.t ~>", bocd.t)
                            # print("posterior probablity:", bocd.growth_probs)
                            # print("bocd.growth_probs[DELAY]:", bocd.growth_probs[DELAY])
                            if bocd.growth_probs[DELAY] >= THRESHOLD:
                                changepoints.append(bocd.t - DELAY + 1)
                                bocd.prune(bocd.t - DELAY)

                        # print("posterior probablity:", r[DELAY, DELAY:-1])
                        # print("maxes:", maxes[:-1])
                        change_points.append(changepoints)

                        # fig_1, axes_1 = plt.subplots(4, 1, figsize=[15, 10], sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1]})
                        # axes_1[0].plot(timeline, df.loc[:, sensor].values)
                        # axes_1[0].set_title(f"Time-series of {sensor}", fontsize=18, weight="bold")
                        # axes_1[0].set_ylim(min_y, max_y)

                        # axes_1[1].plot(timeline, np.append(np.array([0, 0]), np.abs(np.diff(np.diff(df.loc[:, sensor].values)))), color="darkorange")
                        # differential_points = np.argsort(np.abs(np.diff(np.diff(df.loc[:, sensor].values))))[::-1][:num_increment]
                        # for dp in differential_points:
                        #     axes_1[1].axvline(x=dp, color='red', alpha=0.5)
                        # axes_1[1].set_title("Differentials", fontsize=18)

                        # axes_1[1].plot(timeline, df.loc[:, sensor].values)
                        # for cp in robust_change_points:
                        #     if cp[0] == cp[1]:
                        #         axes_1[1].axvline(x=cp[0], c="red")
                        #     else:
                        #         axes_1[1].axvspan(cp[0], cp[1], color='red', alpha=0.3)
                        # axes_1[1].set_title("Basic Statistical Detection", fontsize=18, weight="bold")
                        # # axes_1[1].set_ylim(min_y, max_y)
                        #
                        # axes_1[2].pcolor(np.array(range(0, len(r[:, 0]), sparsity)),
                        #           np.array(range(0, len(r[:, 0]), sparsity)),
                        #           -np.log(r[0:-1:sparsity, 0:-1:sparsity]),
                        #           cmap=cm.Greys, vmin=0, vmax=20, alpha=0.9, shading='auto')
                        # axes_1[2].set_ylim(0, height_limit)
                        # axes_1[2].set_title("Run Length", fontsize=18, weight="bold")
                        #
                        # a = np.where(np.array(changepoints_probs) > THRESHOLD, np.array(changepoints_probs), 0)
                        # axes_1[3].plot(timeline, np.append(a, np.zeros(DELAY)), color="red")
                        # axes_1[3].set_ylim(0, 1)
                        # axes_1[3].set_title("The Probability of Change Point", fontsize=18, weight="bold")
                        # axes_1[3].xaxis.set_ticklabels([int(fig_1.gca().get_xticks()[0] + start + k*np.diff(fig_1.gca().get_xticks())[0]) for k in range(len(fig.gca().get_xticks()))], fontsize=15)
                        #
                        # # fig_1.suptitle(f"{file}_{sheet_name}_{sensor}_period_{start}_to_{end}", fontsize=22, weight='bold')
                        # fig_1.suptitle(f"Univariate Time-series of {sensor}", fontsize=22, weight='bold')
                        # fig_1.tight_layout(rect=[0, 0, 1, 0.985])
                        # fig_1.savefig(
                        #     args.plots_path + f"/{args.task}/cpd_plot/comparison_{fn}_{sheet_name}_{equipment}_period_{start}_to_{end}_delay_{DELAY}.png", dpi=100)
                        # plt.show()

                        axes[sensor_id].plot(timeline, df.loc[:, sensor].values)
                        # axes[sensor_id].set_ylim(min_y, max_y)
                        # axes[2*sensor_id].set_title(f"{sensor}-{equipment} Time-series", fontsize=18)
                        axes[sensor_id].set_title(f"{sensor}", fontsize=18, weight="bold")
                        # axes[2*sensor_id+1].plot(timeline, np.append(np.array([0, 0]),
                        #                                          np.abs(np.diff(np.diff(df.loc[:, sensor].values)))), color="orange")
                        # axes[2*sensor_id+1].set_title("Differentials", fontsize=18)

                        # for i, dp in enumerate(differential_points):
                        #     if i == 0:
                        #         axes[2*sensor_id].axvline(x=dp, color="red", alpha=0.4, label=f"Top {num_increment} Highest In/Decrement")
                        #         axes[2*sensor_id].legend(fontsize=15, loc="upper left")
                        #     else:
                        #         axes[2*sensor_id].axvline(x=dp, color="red", alpha=0.4, label=None)

                        a = np.where(np.array(changepoints_probs) > THRESHOLD, np.array(changepoints_probs), 0)
                        axes[len(sensors)].plot(timeline, np.append(a, np.zeros(DELAY)), color=colors[sensor_id], alpha=0.5, linewidth=2.5, label=sensor)
                        axes[len(sensors)].set_title(f"Probablility of Change Point", fontsize=18, weight="bold")
                        axes[len(sensors)].xaxis.set_ticklabels([int(fig.gca().get_xticks()[0] + start + k * np.diff(fig.gca().get_xticks())[0]) for k in
                                                      range(len(fig.gca().get_xticks()))], fontsize=15)

                        axes[len(sensors)].legend(fontsize=15)
                    # fig.suptitle(f"{file}_{sheet_name}_period_{start}_to_{end}", fontsize=22, weight='bold')
                    fig.suptitle(f"Multivariate Time-series", fontsize=22, weight='bold')
                    fig.tight_layout(rect=[0, 0, 1, 0.985])
                    fig.savefig(args.plots_path + f"/{args.task}/cpd_plot/total_{fn}_{sheet_name}_comparison_between_sensors_period_from_{start}_to_{end}.png", dpi=100)
                    plt.show()
            quit()


if __name__ == "__main__":
    # Configuration options
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7777, help='seed for reproducibility')
    parser.add_argument('--seq_length', type=int, default=1, help='window size')
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--objective', type=str, default='root_cause', choices=['statement', 'root_cause'])
    parser.add_argument('--task', type=str, default='task2', choices=['task1', 'task2', 'task3'])
    parser.add_argument('--chroma', type=str, default='IPA', choices=['HF', 'IPA'])
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'inference'])
    parser.add_argument('--model', type=str, default='DNN', choices=['DNN', 'CNN', 'LR'])
    parser.add_argument('--input_size', type=int, default=21, help='input_size')
    parser.add_argument('--hidden_size', type=int, default=21 * 4, help='hidden_size')
    parser.add_argument('--output_size', type=int, default=11, help='output_size')
    parser.add_argument('--preprocessing', type=bool, default=True, help='Whether to preprocess or not')
    parser.add_argument('--threshold_num_samples', type=int, default=100, help='exclusive when the num of total samples is under')
    parser.add_argument('--add_new_features', type=bool, default=False, help='Whether to add new features or not')
    parser.add_argument('--data_path', type=str, default='../data', help='data path')
    parser.add_argument('--weights_path', type=str, default='../results/weights', help='weights path')
    parser.add_argument('--plots_path', type=str, default='../results/plots', help='plots path')
    parser.add_argument('--plot_timeseries', type=bool, default=False, help='plot time series')
    parser.add_argument('--train_split', type=float, default=0.6, help='train_split')
    parser.add_argument('--test_split', type=float, default=0.5, help='test_split')
    parser.add_argument('--sampling_mode', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--interpolation_mode', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--oversampling', type=bool, default=True, help='Whether Oversampling or not')
    parser.add_argument('--num_epochs', type=int, default=100, help='total epoch')
    parser.add_argument('--print_every', type=int, default=10, help='print statistics for every default epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler', choices=['step', 'cosine'])

    config = parser.parse_args()
    print(config)

    torch.cuda.empty_cache()
    if config.task == "task1":
        task1(config)
    elif config.task == "task2":
        task2(config)
    elif config.task == "task3":
        task3(config)
