import string
import random
from collections import Counter
from statistics import mean

import eli5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as py
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.tools as tls
import shap
import tensorflow as tf
import tensorflow_addons as tfa
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance
from eli5.sklearn import PermutationImportance
from helper_rank import helper_rank_func
from majority_vote import majority_vote_func
from scipy import stats
from scipy.optimize import minimize
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

from optimizer import Lookahead
from radam import RAdam

plt.style.use("seaborn-pastel")

# initiate plotly and cufflinks for offline mode
py.offline.init_notebook_mode(connected=True)
#cf.go_offline()
# initiate Javascript for SHAP
shap.initjs()

def ensemble_feature_importance(NUM_FEATS, features, output, coef, informative, noise):
    
    # Function to normalise aray
    def normalise(array):
        return (array - array.min()) / (array.max() - array.min())
    
    # normalise actual coefficient so it can be compared to predicted coef
    scaled_coef = normalise(coef)
    
    # give features alphabetical name
    
    def id_generator(size=5, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    alphabet_list = list()
    for i in range(1000):
        alphabet_list.append(id_generator())
    column_names = np.array(alphabet_list[0:NUM_FEATS])
    df_features = pd.DataFrame(features, columns=column_names)
    
    df_output = pd.DataFrame(output, columns=["output"])
    
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    # split data to train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.2, shuffle=True)
    
    # fit random forest
    rf_regr = RandomForestRegressor(max_depth=7, n_estimators=700)
    rf_regr.fit(X_train, y_train)
    rf_predicted = rf_regr.predict(X_test)
    
    # model error
    rf_mae = np.abs(rf_predicted - y_test).mean()
    print(f"predicted_shape:{rf_predicted.shape} y_test shape:{y_test.shape}")
    print(f"rf_mae:{rf_mae}")
    
    # permutation importance for random forest
    pi_result_rf = permutation_importance(rf_regr, X_test, y_test, n_repeats=10, scoring=mae_scorer)
    pi_result_rf_scaled = normalise(pi_result_rf.importances_mean)
    pos = np.arange(column_names.size)
#     plt.figure(figsize=(15, 10))
#     plt.bar(pos, pi_result_rf_scaled, align="center", alpha=0.5, label="RF")
#     plt.bar(pos, scaled_coef, align="center", alpha=0.5, label="actual")
#     plt.xticks(pos, column_names)
#     plt.title("RF Permutation Importances (test set)")
#     plt.legend(loc="best")
#     plt.show()
    # fit gradient boosted tree
    gb_regr = GradientBoostingRegressor(max_depth=7, n_estimators=700)
    gb_regr.fit(X_train, y_train)
    gb_predicted = gb_regr.predict(X_test)
    
    # model error
    gb_mae = np.abs(gb_predicted - y_test).mean()
    print(f"predicted_shape:{gb_predicted.shape} y_test shape:{y_test.shape}")
    print(f"rf_mae:{gb_mae}")
    
    # permutation importance for gradient boosted tree
    pi_result_gb = permutation_importance(gb_regr, X_test, y_test, n_repeats=10, scoring=mae_scorer)
    
    pi_result_gb_scaled = normalise(pi_result_gb.importances_mean)
    pos = np.arange(column_names.size)
#     plt.figure(figsize=(15, 10))
#     plt.bar(pos, pi_result_gb_scaled, align="center", alpha=0.5, label="GB")
#     plt.bar(pos, scaled_coef, align="center", alpha=0.5, label="actual")
#     plt.xticks(pos, column_names)
#     plt.title("GB Permutation Importances (test set)")
#     # plt.xticks(rotation=90)
#     plt.legend(loc="best")
#     plt.show()
    
    # SHAP for rf and gb
    explainer_rf = shap.TreeExplainer(rf_regr)
    shap_values_rf = explainer_rf.shap_values(df_features)

    explainer_gb = shap.TreeExplainer(gb_regr)
    shap_values_gb = explainer_gb.shap_values(df_features)

    gb_sv_scaled = normalise(np.abs(shap_values_gb).mean(0))
    rf_sv_scaled = normalise(np.abs(shap_values_rf).mean(0))
    
    ############################################################################
    ## DNN
    ############################################################################
    class SimpleNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear5 = nn.Linear(NUM_FEATS, 64)
            self.relu5 = nn.Tanh()

            self.linear6 = nn.Linear(64, 32)
            self.relu6 = nn.Tanh()

            self.linear7 = nn.Linear(32, 16)
            self.relu7 = nn.Tanh()

            self.linear8 = nn.Linear(16, 8)
            self.relu8 = nn.Tanh()

            self.linear9 = nn.Linear(8, 6)
            self.relu9 = nn.Tanh()

            self.linear10 = nn.Linear(6, 4)
            self.relu10 = nn.Tanh()

            self.linear11 = nn.Linear(4, 1)
            # self.relu11 = nn.ReLU()

        def forward(self, x):
            x = self.relu5(self.linear5(x))
            x = self.relu6(self.linear6(x))
            x = self.relu7(self.linear7(x))
            x = self.relu8(self.linear8(x))
            x = self.relu9(self.linear9(x))
            x = self.relu10(self.linear10(x))
            x = self.linear11(x)
            return x


    from optimizer import Lookahead
    from radam import RAdam

    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    net = SimpleNNModel()


    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    net.apply(init_weights)

    USE_PRETRAINED_MODEL = False
    n_batches = 32
    if USE_PRETRAINED_MODEL:
        net.load_state_dict(torch.load("interpret_model.pt"))
        print("Model Loaded!")
    else:
        criterion = nn.MSELoss()
        num_epochs = 1000

        # optimizer = torch.optim.AdamW(net.parameters(), lr=0.0003, weight_decay=0.01)
        base_optim = RAdam(net.parameters(), lr=0.01)
        optimizer = Lookahead(base_optim, k=6, alpha=0.5)
        input_tensor = torch.from_numpy(X_train).type(torch.FloatTensor)
        label_tensor = torch.from_numpy(y_train).type(torch.FloatTensor)
        for epoch in range(num_epochs):
            for i in range(n_batches):
                local_X, local_y = (
                    input_tensor[i * n_batches : (i + 1) * n_batches,],
                    label_tensor[i * n_batches : (i + 1) * n_batches,],
                )
                output = net(input_tensor)
                loss = criterion(output, label_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 100 == 0:
                print("Epoch {}/{} => Loss: {:.6f}".format(epoch + 1, num_epochs, loss.item()))

        torch.save(net.state_dict(), "interpret_model.pt")
        
    #model error
    test_input_tensor = torch.from_numpy(X_test).type(torch.FloatTensor)
    test_output = net(test_input_tensor).detach().numpy()
    print(f"predicted_shape:{test_output.shape} y_test shape:{y_test.shape}")
    print("dnn_mae:", (np.abs(test_output - y_test)).mean())
    # Intergrated gradient
    ig = IntegratedGradients(net)
    test_input_tensor = torch.from_numpy(X_train).type(torch.FloatTensor)
    test_input_tensor.requires_grad_()
    attr, delta = ig.attribute(test_input_tensor, return_convergence_delta=True)
    attr = attr.detach().numpy()

    feature_names = list(column_names)
    dnn_ig_scaled = normalise(np.abs(np.mean(attr, axis=0)))
    #print(f"DNN IG error: {np.abs(dnn_ig_scaled - scaled_coef).mean()}")
#     def visualize_importances(
#         feature_names,
#         importances,
#         title="Intergrated Gradient Average Feature Importances",
#         plot=True,
#         axis_title="Features",
#     ):
#         print(title)
#         for i in range(len(feature_names)):
#             print(feature_names[i], ": ", "%.3f" % (importances[i]))
#         x_pos = np.arange(len(feature_names))
#         if plot:
#             plt.figure(figsize=(12, 6))
#             plt.bar(x_pos, importances, align="center")
#             plt.xticks(x_pos, feature_names, wrap=False)
#             plt.xlabel(axis_title)
#             plt.title(title)


#     visualize_importances(column_names, np.mean(attr, axis=0))

    def base_model():
        model = tf.keras.Sequential(
            [
                layers.Dense(
                    64, activation="tanh", input_shape=(X_train.shape[1],), bias_regularizer=tf.keras.regularizers.l2(0.01),
                ),
                layers.Dense(64, activation="tanh", bias_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(32, activation="tanh", bias_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(16, activation="tanh", bias_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(8, activation="tanh", bias_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(6, activation="tanh", bias_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(4, activation="tanh", bias_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(1, activation="linear"),
            ]
        )

        radam = tfa.optimizers.RectifiedAdam()
        ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
        # tf.keras.optimizers.Adam(1e-4)
        # Configure a model for mean-squared error regression.
        model.compile(optimizer=ranger, loss="mse", metrics=["mae"])  # mean squared error  # mean absolute error

        return model


    # reshape for deep model
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=100, verbose=0, mode="auto", baseline=None, restore_best_weights=True,
    )

    dnn_model = KerasRegressor(
        build_fn=base_model, epochs=1000, batch_size=32, verbose=0, validation_split=0.1, callbacks=[callback],
    )

    history = dnn_model.fit(X_train, y_train)
    perm = PermutationImportance(dnn_model, scoring=mae_scorer).fit(X_train, y_train)
    dnn_pi_df = eli5.explain_weights_df(perm, feature_names=column_names.tolist())
    dnn_pi_df.sort_values(by=["feature"], inplace=True)
    dnn_pi_scaled = normalise(dnn_pi_df["weight"].values)
    
    model = tf.keras.Sequential(
        [
            layers.Dense(
                64, activation="tanh", input_shape=(X_train.shape[1],), bias_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            layers.Dense(64, activation="tanh", bias_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(32, activation="tanh", bias_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(16, activation="tanh", bias_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(8, activation="tanh", bias_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(6, activation="tanh", bias_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(4, activation="tanh", bias_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(1, activation="linear"),
        ]
    )

    radam = tfa.optimizers.RectifiedAdam()
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    # tf.keras.optimizers.Adam(1e-4)
    # Configure a model for mean-squared error regression.
    model.compile(optimizer=ranger, loss="mse", metrics=["mae"])  # mean squared error  # mean absolute error


    # reshape for deep model
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=100, verbose=0, mode="auto", baseline=None, restore_best_weights=True,
    )


    model.fit(
        X_train, y_train, epochs=1000, batch_size=32, verbose=0, validation_split=0.1, callbacks=[callback],
    )
    
    # save keras model
    tf.keras.models.save_model(model, "dnn_model.h5")
    
    # SHAP for DNN
    explainer_dnn = shap.KernelExplainer(model.predict, data=shap.kmeans(df_features.iloc[:300, :], 10))
    shap_values = explainer_dnn.shap_values(df_features.iloc[:300, :], nsamples=200, l1_reg=f"num_features({NUM_FEATS})")
    dnn_sv = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)
    dnn_sv_scaled = normalise(dnn_sv)
    
    
    # combine results into single array
    rf_sv_scaled_reshaped = np.reshape(rf_sv_scaled, (rf_sv_scaled.shape[0], 1))
    pi_result_rf_scaled_reshaped = np.reshape(pi_result_rf_scaled, (pi_result_rf_scaled.shape[0], 1))
    gb_sv_scaled_reshaped = np.reshape(gb_sv_scaled, (gb_sv_scaled.shape[0], 1))
    pi_result_gb_scaled_reshaped = np.reshape(pi_result_gb_scaled, (pi_result_gb_scaled.shape[0], 1))
    dnn_pi_scaled_reshaped = np.reshape(dnn_pi_scaled, (dnn_pi_scaled.shape[0], 1))
    dnn_sv_scaled_reshaped = np.reshape(dnn_sv_scaled, (dnn_sv_scaled.shape[0], 1))
    dnn_ig_scaled_reshaped = np.reshape(dnn_ig_scaled, (dnn_ig_scaled.shape[0], 1))

    all_stacked = np.hstack(
        (
            rf_sv_scaled_reshaped,
            pi_result_rf_scaled_reshaped,
            gb_sv_scaled_reshaped,
            pi_result_gb_scaled_reshaped,
            dnn_pi_scaled_reshaped,
            dnn_sv_scaled_reshaped,
            dnn_ig_scaled_reshaped,
        )
    )

    print(f"rf_sv:{rf_sv_scaled_reshaped}")
    print("")
    print(f"rf_pi:{pi_result_rf_scaled_reshaped}")
    print("")
    print(f"gb_sv:{gb_sv_scaled_reshaped}")
    print("")
    print(f"gb_pi:{pi_result_gb_scaled_reshaped}")
    print("")
    print(f"dnn_sv:{dnn_sv_scaled_reshaped}")
    print("")
    print(f"dnn_pi:{dnn_pi_scaled_reshaped}")
    print("")
    print(f"dnn_ig:{dnn_ig_scaled_reshaped}")
    print("")
    all_stacked_df = pd.DataFrame(all_stacked, columns=["rf_sv", "rf_pi", "gb_sv", "gb_pi", "dnn_pi", "dnn_sv", "dnn_ig"])

    all_stacked_df.to_csv("all_stacked.csv")
    
    print(f"RF PI error: {np.abs(pi_result_rf_scaled - scaled_coef).mean()}")
    print(f"GB PI error: {np.abs(pi_result_gb_scaled - scaled_coef).mean()}")
    print(f"DNN PI error: {np.abs(dnn_pi_scaled - scaled_coef).mean()}")
    print(f"RF+GB+DNN PI error: {np.abs(((pi_result_gb_scaled+pi_result_rf_scaled+dnn_pi_scaled)/3) - scaled_coef).mean()}")
    print("")
    print(f"RF SV error: {np.abs(rf_sv_scaled - scaled_coef).mean()}")
    print(f"GB SV error: {np.abs(gb_sv_scaled - scaled_coef).mean()}")
    print(f"DNN SV error: {np.abs(dnn_sv_scaled - scaled_coef).mean()}")
    print(f"RF+GB+DNN SV error: {np.abs(((rf_sv_scaled+gb_sv_scaled+dnn_sv_scaled)/3) - scaled_coef).mean()}")
    print("")
    print(f"GB PI error: {np.abs(pi_result_gb_scaled - scaled_coef).mean()}")
    print(f"GB SV error: {np.abs(gb_sv_scaled - scaled_coef).mean()}")
    print(f"GB PI + GB SV error: {np.abs(((gb_sv_scaled+pi_result_gb_scaled)/2) - scaled_coef).mean()}")
    print("")
    print(f"RF PI error: {np.abs(pi_result_rf_scaled - scaled_coef).mean()}")
    print(f"RF SV error: {np.abs(rf_sv_scaled - scaled_coef).mean()}")
    print(f"RF PI + RF SV error: {np.abs(((rf_sv_scaled+pi_result_rf_scaled)/2) - scaled_coef).mean()}")
    print("")
    print(f"DNN PI error: {np.abs(dnn_pi_scaled - scaled_coef).mean()}")
    print(f"DNN SV error: {np.abs(dnn_sv_scaled - scaled_coef).mean()}")
    print(f"DNN IG error: {np.abs(dnn_ig_scaled - scaled_coef).mean()}")
    print(f"DNN PI + DNN SV + DNN IG error: {np.abs(((dnn_pi_scaled+dnn_sv_scaled+dnn_ig_scaled)/3) - scaled_coef).mean()}")
    print("")

    point_five = np.full((NUM_FEATS), 0.5)
    zeros = np.full((NUM_FEATS), 0)
    ones = np.full((NUM_FEATS), 1)
    print(f"Dummy Average of 0.0 error: {np.abs(zeros - scaled_coef).mean()}")
    print(f"Dummy Average of 0.5 error: {np.abs(point_five - scaled_coef).mean()}")
    print(f"Dummy Average of 0.1 error: {np.abs(ones - scaled_coef).mean()}")
    print("")
    
    # median
    total_scaled_median = np.median(all_stacked, axis=1)

    # mean 
    total_scaled_mean = (
        rf_sv_scaled
        + pi_result_rf_scaled
        + gb_sv_scaled
        + pi_result_gb_scaled
        + dnn_pi_scaled
        + dnn_sv_scaled
        + dnn_ig_scaled
    ) / 7


    # Box and whiskers
    total_scaled_box_whiskers = np.array([])
    for i in range(all_stacked.shape[0]):
        temp_whiskers = np.array([])
        q3 = np.quantile(all_stacked[i, :], 0.75)
        q1 = np.quantile(all_stacked[i, :], 0.25)
        upper_whiskers = q3 + (1.5 * (q3 - q1))
        lower_whiskers = q1 - (1.5 * (q3 - q1))
        for j in range(all_stacked[i, :].shape[0]):
            if (all_stacked[i, :][j] >= lower_whiskers) and (all_stacked[i, :][j] <= upper_whiskers):
                temp_whiskers = np.append(temp_whiskers, all_stacked[i, :][j])
        total_scaled_box_whiskers = np.append(total_scaled_box_whiskers, temp_whiskers.mean())

    # Thompson Tau
    # (1) calculate sample mean
    # (2) calculate delta_min = |mean - min| and delta_max|mean - max|
    # (3) tau value from tau table value for sample size 7: 1.7110
    # (4) calculate standard deviation
    # (5) multiply tau with standard deviation = tau*std threshold
    # (6) compare (3) and (5)
    tau = 1.7110


    def tau_test(test_data):
        for i in range(test_data.shape[0]):
            test_data_mean = test_data.mean()
            test_data_std = np.std(test_data, ddof=1)
            test_data_min = test_data.min()
            test_data_min_index = np.argmin(test_data)
            test_data_max = test_data.max()
            test_data_max_index = np.argmax(test_data)
            test_data_min_delta = np.abs(test_data_min - test_data_mean)
            test_data_max_delta = np.abs(test_data_max - test_data_mean)

            if (test_data_min_delta >= test_data_max_delta) and (test_data_min_delta > tau * test_data_std):
                test_data = np.delete(test_data, test_data_min_index)
            else:
                if test_data_max_delta > (tau * test_data_std):
                    test_data = np.delete(test_data, test_data_max_index)
        return test_data


    total_scaled_tau_test = np.array([])
    for i in range(all_stacked.shape[0]):
        mean_tau = np.array([tau_test(all_stacked[i, :]).mean()])
        total_scaled_tau_test = np.append(total_scaled_tau_test, mean_tau)

    total_scaled_tau_test = np.array([total_scaled_tau_test])
    total_scaled_tau_test = np.reshape(total_scaled_tau_test, (-1,))

    # Mode

    total_scaled_mode = np.array([])
    for i in range(all_stacked.shape[0]):
        params = stats.norm.fit(all_stacked[i, :])

        def your_density(x):
            return -stats.norm.pdf(x, *params)

        total_scaled_mode = np.append(total_scaled_mode, minimize(your_density, 0).x[0])

    total_scaled_mode = np.reshape(total_scaled_mode, (-1,))

    # majority vote
    total_scaled_majority_vote = majority_vote_func(all_stacked, NUM_FEATS)

    # kendall tau
    columns_name = ["rf_sv", "rf_pi", "gb_sv", "gb_pi", "dnn_pi", "dnn_sv", "dnn_ig"]
    total_scaled_kendall_tau, kendall_tau_p_value = helper_rank_func(stats.kendalltau, all_stacked, columns_name, NUM_FEATS)

    # spearman
    total_scaled_spearman, spearman_p_value = helper_rank_func(stats.spearmanr, all_stacked, columns_name, NUM_FEATS)

    # convert signifance to true or false
    def plot_rank_method_significance(p_value, rank_name, sig_threshold, columns_name):
        """
        p_value:
            Dataframe of p-value from rank method.
        rank_name:
            A string of rank method name.
        sig_threshold:
            Probability threshold for value to be considered significant.
        columns_name:
            A list of feature importance methods and models name.
        """

        p_value_sig = p_value < sig_threshold
        annotation_text = p_value_sig.values.astype(str).tolist()

        fig = ff.create_annotated_heatmap(
            p_value.values.tolist(), x=columns_name, y=columns_name, annotation_text=annotation_text, colorscale="Viridis",
        )

        fig.update_layout(
            title_text=f"Are association between different feature importance methods and models statistically significant? ({rank_name}) Informative Features: {informative}, Noise level: {noise}"
        )
        #fig.show()
        fig.write_html(f"Images/{rank_name}_results_{informative}_feat_{noise}_noise.html")


    plot_rank_method_significance(kendall_tau_p_value, "Kendall Tau", 0.05, columns_name)
    plot_rank_method_significance(spearman_p_value, "Spearman Rho", 0.05,columns_name)

    # renormalise the result

    total_scaled_median = normalise(total_scaled_median)
    total_scaled_mean = normalise(total_scaled_mean)
    total_scaled_mode = normalise(total_scaled_mode)
    total_scaled_box_whiskers = normalise(total_scaled_box_whiskers)
    total_scaled_tau_test = normalise(total_scaled_tau_test)
    total_scaled_majority_vote = normalise(total_scaled_majority_vote)
    total_scaled_kendall_tau = normalise(total_scaled_kendall_tau)
    total_scaled_spearman = normalise(total_scaled_spearman)


    # test shape
    assert total_scaled_median.shape == (
        NUM_FEATS,
    ), f"Shape mismatch: Expect shape to be ({NUM_FEATS},) but receive {(total_scaled_median.shape)}"
    assert total_scaled_mean.shape == (
        NUM_FEATS,
    ), f"Shape mismatch: Expect shape to be ({NUM_FEATS},) but receive {(total_scaled_mean.shape)}"
    assert total_scaled_mode.shape == (
        NUM_FEATS,
    ), f"Shape mismatch: Expect shape to be ({NUM_FEATS},) but receive {(total_scaled_mode.shape)}"
    assert total_scaled_box_whiskers.shape == (
        NUM_FEATS,
    ), f"Shape mismatch: Expect shape to be ({NUM_FEATS},) but receive {(total_scaled_box_whiskers.shape)}"
    assert total_scaled_tau_test.shape == (
        NUM_FEATS,
    ), f"Shape mismatch: Expect shape to be ({NUM_FEATS},) but receive {(total_scaled_tau_test.shape)}"
    assert total_scaled_majority_vote.shape == (
        NUM_FEATS,
    ), f"Shape mismatch: Expect shape to be ({NUM_FEATS},) but receive {(total_scaled_majority_vote.shape)}"
    assert total_scaled_kendall_tau.shape == (
        NUM_FEATS,
    ), f"Shape mismatch: Expect shape to be ({NUM_FEATS},) but receive {(total_scaled_kendall_tau.shape)}"
    assert total_scaled_spearman.shape == (
        NUM_FEATS,
    ), f"Shape mismatch: Expect shape to be ({NUM_FEATS},) but receive {(total_scaled_spearman.shape)}"

    # Print all final result
    print(f" All MAE (mean): {np.abs(total_scaled_mean - scaled_coef).mean()}")
    print(f" All MAE (median): {np.abs(total_scaled_median - scaled_coef).mean()}")
    print(f" All MAE (mode): {np.abs(total_scaled_mode - scaled_coef).mean()}")
    print(f" All MAE (box-whiskers): {np.abs(total_scaled_box_whiskers - scaled_coef).mean()}")
    print(f" All MAE (tau-test): {np.abs(total_scaled_tau_test - scaled_coef).mean()}")
    print(f" All MAE (majority vote): {np.abs(total_scaled_majority_vote - scaled_coef).mean()}")
    print(f" All MAE (Kendall tau): {np.abs(total_scaled_kendall_tau - scaled_coef).mean()}")
    print(f" All MAE (Spearman Rho): {np.abs(total_scaled_spearman - scaled_coef).mean()}")

    # save result in csv
    methods = np.array(
        ["Mode", "Median", "Mean", "Box-Whiskers", "Tau Test", "Majority Vote", "Kendall Tau", "Spearman Rho", "Actual"]
    )
    multiplied_importance = np.append(
        [total_scaled_mode],
        [
            total_scaled_median,
            total_scaled_mean,
            #         mi_scaled,
            total_scaled_box_whiskers,
            total_scaled_tau_test,
            total_scaled_majority_vote,
            total_scaled_kendall_tau,
            total_scaled_spearman,
            scaled_coef,
        ],
    )
    multiplied_feature = np.tile(feature_names, int(multiplied_importance.shape[0] / len(feature_names)))
    multiplied_methods = np.repeat(
        methods, int(multiplied_importance.shape[0] / methods.shape[0])
    )  # for 9 different ensemble method


    df_results = pd.DataFrame(
        {"Importance": multiplied_importance, "Features": multiplied_feature, "Methods": multiplied_methods}
    )


    df_results.to_csv(f"Tabular/results_{informative}_feat_{noise}_noise.csv")

    fig = px.scatter(df_results, x="Features", y="Importance", color="Methods")

    fig.update_layout(
        template="plotly_white", title="Feature importance of actual vs ensemble method", width=1500, height=800,
    )

    fig.update_traces(marker=dict(size=12, line=dict(width=2, color="Black")), selector=dict(mode="markers"), opacity=0.7)

    fig.write_html(f"Images/scatter_results_{informative}_feat_{noise}_noise.html")
    fig.show()
    