from utils import file_names, GPCurvesReader, plot_functions
from models import DeterministicModel
from preprocess import get_data, get_new_features, peaks_in_rt, get_area_in_RTbox, z_score_normalization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import plotly.graph_objects as go
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ExpSineSquared, ConstantKernel as C

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

# TRAINING_ITERATIONS = int(2e5)
# MAX_CONTEXT_POINTS = 20
# PLOT_AFTER = int(2e4)
# tf.reset_default_graph()
#
# # Train dataset
# dataset_train = GPCurvesReader(batch_size=64, max_num_context=MAX_CONTEXT_POINTS)
# data_train = dataset_train.generate_curves()

# # Test dataset
# dataset_test = GPCurvesReader(batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True)
# data_test = dataset_test.generate_curves()
#
# # Sizes of the layers of the MLPs for the encoder and decoder
# # The final output layer of the decoder outputs two values, one for the mean and
# # one for the variance of the prediction at the target location
# encoder_output_sizes = [128, 128, 128, 128]
# decoder_output_sizes = [128, 128, 2]
#
# # Define the model
# model = DeterministicModel(encoder_output_sizes, decoder_output_sizes)
#
# # Define the loss
# print("query:", data_train.query)
# print("data_train.num_total_points:", data_train.num_total_points)
# print("data_train.num_context_points:", data_train.num_context_points)
# print("data_train.target_y:", data_train.target_y)
# log_prob, _, _ = model(data_train.query, data_train.num_total_points,
#                        data_train.num_context_points, data_train.target_y)
# loss = -tf.reduce_mean(log_prob)
#
# # Get the predicted mean and variance at the target points for the testing set
#
# print("data_test.query:", data_test.query)
# print("data_test.num_total_points:", data_test.num_total_points)
# print("data_test.num_context_points:", data_test.num_context_points)
# print("data_test.target_y:", data_test.target_y)
#
# _, mu, sigma = model(data_test.query, data_test.num_total_points,
#                      data_test.num_context_points)
#
# # Set up the optimizer and train step
# optimizer = tf.train.AdamOptimizer(1e-4)
# train_step = optimizer.minimize(loss)
# init = tf.initialize_all_variables()
#
# with tf.Session(config=config) as sess:
#     sess.run(init)
#     for it in range(TRAINING_ITERATIONS):
#         sess.run([train_step])
#
#         # Plot the predictions in `PLOT_AFTER` intervals
#         if it % PLOT_AFTER == 0:
#             loss_value, pred_y, var, target_y, whole_query = sess.run([loss, mu, sigma, data_test.target_y, data_test.query])
#
#             print("pred_y:", pred_y.shape)
#             print("var:", var.shape)
#             print("target_y", target_y.shape)
#             (context_x, context_y), target_x = whole_query
#             print("context_x:", context_x.shape)
#             print("context_y:", context_y.shape)
#             print("target_x:", target_x.shape)
#             print('Iteration: {}, loss: {}'.format(it, loss_value))
#
#             # Plot the prediction and the context
#             plot_functions(target_x, target_y, context_x, context_y, pred_y, var)

# for i, sheet_name in enumerate(file_names["HF"][0:1]):
#     print(f"{sheet_name} is under preprocessing")
#     df_temp, _ = get_data("task2", "../data", sheet_name, "HF")
#     df_temp["equipment_code"] = sheet_name
#
#     fig, axes = plt.subplots(21, 1, figsize=(30, 26), sharex=True)
#     time = np.arange(len(df_temp))
#     for i, element in enumerate(df_temp.columns.difference(["time", "root_cause", "equipment_code", "class"])):
#         if i == 0:
#             axes[i].set_title(sheet_name, fontsize=20)
#         axes[i].set_ylabel(element, fontsize=20, weight='bold')
#         axes[i].bar(time, df_temp[element].values, color="black", width=7)
#         axes[i].axvspan(4925, 5005, facecolor='red', alpha=0.3)
#         axes[i].axvspan(16234, 16899, facecolor='red', alpha=0.3)
#     fig.tight_layout()
#     plt.savefig("../results/plots/task2/" + sheet_name + ".png", dpi=1200)
#     plt.show()


# GP
for i, sheet_name in enumerate(file_names["HF"][:1]):
    df_all, df_observation = get_data("task2", "../data", sheet_name, "HF")

    x = np.atleast_2d(df_all.index.values).T
    # y_ = df_all[df_all.columns.difference(["class", "time", "root_cause"])].rolling(245, min_periods=1).mean()
    X = np.atleast_2d(df_observation.index.values).T
    y = df_observation.values
    y_ = df_observation["Al"].values

    print(y_.shape)

    gaussian_kernel_2 = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gaussian_kernel_1 = RBF(10, (0.5, 1))
    # periodic_kernel = 0.18 * ExpSineSquared(length_scale=1.0, periodicity=0.5)
    gp_1 = GaussianProcessRegressor(kernel=gaussian_kernel_1, n_restarts_optimizer=9)
    gp_2 = GaussianProcessRegressor(kernel=gaussian_kernel_2, n_restarts_optimizer=9)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp_1.fit(X, y)
    gp_2.fit(X, y_)

    y_pred_1, sigma_1 = gp_1.predict(x, return_std=True)
    y_pred_2, sigma_2 = gp_2.predict(x, return_std=True)

    # plt.figure(figsize=(30, 45))
    # for i in range(21):
    #     plt.plot(X, y[:, i], 'r.', markersize=5, label='Observations')
    #     # plt.plot(x, y_["Al"], color="red", label='Moving Average')
    #     plt.plot(x, y_pred_1[:, i], 'b-', label='Prediction')
    #     plt.fill(np.concatenate([x, x[::-1]]),
    #              np.concatenate([y_pred_1[:, i] - 1.9600 * sigma_1,
    #                              (y_pred_1[:, i] + 1.9600 * sigma_1)[::-1]]),
    #              alpha=.5, fc='b', ec='None', label='95% confidence interval')
    # plt.xlabel('$time$', fontsize=20)
    # plt.ylabel('$f$', fontsize=20)
    # plt.legend(loc='upper left')
    # plt.tight_layout()
    # plt.savefig("../results/plots/task2/" + sheet_name + "gp_multivariate.png", dpi=1200)
    # plt.show()

    plt.figure(figsize=(10, 3))
    plt.plot(X, y_, 'r.', markersize=5, label='Observations')
    plt.plot(x, y_pred_1[:, 0], 'b-', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred_1[:, 0] - 1.9600 * sigma_1,
                             (y_pred_1[:, 0] + 1.9600 * sigma_1)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$time$', fontsize=20)
    plt.ylabel('$f$', fontsize=20)
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig("../results/plots/task2/" + sheet_name + "gp_1.png", dpi=1200)
    plt.show()

    plt.figure(figsize=(10, 3))
    plt.plot(X, y_, 'r.', markersize=5, label='Observations')
    # plt.plot(x, y_["Al"], label='Moving Average')
    plt.plot(x, y_pred_2, 'b-', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred_2 - 1.9600 * sigma_2,
                             (y_pred_2 + 1.9600 * sigma_2)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$time$', fontsize=20)
    plt.ylabel('$f$', fontsize=20)
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig("../results/plots/task2/" + sheet_name + "gp_2.png", dpi=1200)
    plt.show()
