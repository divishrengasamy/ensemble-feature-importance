from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from sklearn import svm
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
        return ((array - array.min()) / (array.max() - array.min()))
    
    # normalise actual coefficient so it can be compared to predicted coef
    scaled_coef = normalise(coef)
    
    # normalise features
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(features)
    features = scaler.transform(features)
    
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
    
    ###############################################################
    # fit random forest
    ###############################################################
    rf_regr = RandomForestRegressor(max_depth=70, n_estimators=1400, min_samples_split=2,min_samples_leaf=2,max_features='auto',bootstrap=True)
    rf_regr.fit(X_train, y_train)
    rf_predicted = rf_regr.predict(X_test)
    
    # model error
    rf_mae = np.abs(rf_predicted - y_test).mean()
    rf_rmse = np.sqrt(np.square(rf_predicted - y_test).mean())
    print(f"predicted_shape:{rf_predicted.shape} y_test shape:{y_test.shape}")
    print(f"rf_mae:{rf_mae}")
    print(f"rf_rmse:{rf_rmse}")
    

    ###############################################################
    # fit gradient boosted tree
    ###############################################################
    gb_regr = GradientBoostingRegressor(max_depth=50, n_estimators=1600, min_samples_split=2, min_samples_leaf=4, max_features='auto',loss='lad', learning_rate=0.1)
    gb_regr.fit(X_train, y_train)
    gb_predicted = gb_regr.predict(X_test)
    
    # model error
    gb_mae = np.abs(gb_predicted - y_test).mean()
    gb_rmse = np.sqrt(np.square(gb_predicted - y_test).mean())
    print(f"predicted_shape:{gb_predicted.shape} y_test shape:{y_test.shape}")
    print(f"gb_mae:{gb_mae}")
    print(f"gb_rmse:{gb_rmse}")
    
    
    ################################################################
    # fit support vector regressor
    ###############################################################
    svr_regr = svm.SVR(kernel='linear', C=2048, epsilon=0.5, gamma=1e-07)
    svr_regr.fit(X_train, y_train)
    svr_predicted = svr_regr.predict(X_test)
    
    svr_mae = np.abs(svr_predicted - y_test).mean()
    svr_rmse = np.sqrt(np.square(svr_predicted - y_test).mean())
    print(f"predicted_shape:{svr_predicted.shape} y_test shape:{y_test.shape}")
    print(f"svr_mae:{svr_mae}")
    print(f"svr_rmse:{svr_rmse}")
    

    
    ################################################################
    # permutation importance for rf, gb, and svr
    ################################################################
    
    ###############################
    # Train
    ###############################
    train_pi_result_rf = permutation_importance(rf_regr, X_train, y_train, n_repeats=10, scoring=mae_scorer)
    train_pi_result_rf_scaled = normalise(train_pi_result_rf.importances_mean)
    pos = np.arange(column_names.size)
    
    # permutation importance for gradient boosted tree
    train_pi_result_gb = permutation_importance(gb_regr, X_train, y_train, n_repeats=10, scoring=mae_scorer)
    train_pi_result_gb_scaled = normalise(train_pi_result_gb.importances_mean)
    
    # permutation importance for svr
    train_pi_result_svr = permutation_importance(svr_regr, X_train, y_train, n_repeats=10, scoring=mae_scorer)
    train_pi_result_svr_scaled = normalise(train_pi_result_svr.importances_mean)
    
    ###############################
    # Test
    ###############################
    pi_result_rf = permutation_importance(rf_regr, X_test, y_test, n_repeats=10, scoring=mae_scorer)
    pi_result_rf_scaled = normalise(pi_result_rf.importances_mean)
    pos = np.arange(column_names.size)
    
    # permutation importance for gradient boosted tree
    pi_result_gb = permutation_importance(gb_regr, X_test, y_test, n_repeats=10, scoring=mae_scorer)
    pi_result_gb_scaled = normalise(pi_result_gb.importances_mean)
    
    # permutation importance for svr
    pi_result_svr = permutation_importance(svr_regr, X_test, y_test, n_repeats=10, scoring=mae_scorer)
    pi_result_svr_scaled = normalise(pi_result_svr.importances_mean)
    
    ################################################################
    # SHAP for rf, gb, and svr
    ################################################################
    
    ###############################
    # Train
    ###############################
    train_explainer_rf = shap.TreeExplainer(rf_regr)
    train_shap_values_rf = train_explainer_rf.shap_values(X_train)

    train_explainer_gb = shap.TreeExplainer(gb_regr)
    train_shap_values_gb = train_explainer_gb.shap_values(X_train)
    
    
    train_explainer_svr = shap.KernelExplainer(svr_regr.predict, data=shap.kmeans(X_train, 10))
    train_shap_values_svr = train_explainer_svr.shap_values(X_train, nsamples=800, l1_reg=f"num_features({NUM_FEATS})")
    

    train_gb_sv_scaled = normalise(np.abs(train_shap_values_gb).mean(0))
    train_rf_sv_scaled = normalise(np.abs(train_shap_values_rf).mean(0))
    train_svr_sv_scaled = normalise(np.abs(train_shap_values_svr).mean(0))
    
    ###############################
    # Test
    ###############################
    explainer_rf = shap.TreeExplainer(rf_regr)
    shap_values_rf = explainer_rf.shap_values(X_test)

    explainer_gb = shap.TreeExplainer(gb_regr)
    shap_values_gb = explainer_gb.shap_values(X_test)
    
    
    explainer_svr = shap.KernelExplainer(svr_regr.predict, data=shap.kmeans(X_test, 10))
    shap_values_svr = explainer_svr.shap_values(X_test, nsamples=200, l1_reg=f"num_features({NUM_FEATS})")
    

    gb_sv_scaled = normalise(np.abs(shap_values_gb).mean(0))
    rf_sv_scaled = normalise(np.abs(shap_values_rf).mean(0))
    svr_sv_scaled = normalise(np.abs(shap_values_svr).mean(0))
    
    ############################################################################
    ## DNN
    ############################################################################
    class SimpleNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear5 = nn.Linear(NUM_FEATS, 64)
            self.relu5 = nn.ReLU()

            self.linear6 = nn.Linear(64, 32)
            self.relu6 = nn.ReLU()

            self.linear7 = nn.Linear(32, 16)
            self.relu7 = nn.ReLU()

            self.linear8 = nn.Linear(16, 8)
            self.relu8 = nn.ReLU()

            self.linear9 = nn.Linear(8, 6)
            self.relu9 = nn.ReLU()

            self.linear10 = nn.Linear(6, 4)
            self.relu10 = nn.ReLU()

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
    dnn_mae = (np.abs(test_output - y_test)).mean()
    dnn_rmse = np.sqrt(np.square(test_output - y_test).mean())
    print(f"dnn_mae: {dnn_mae}")
    print(f"dnn_rmse: {dnn_rmse}")
    
    ##########################################
    # DNN Intergrated gradient
    ##########################################
    
    
    def dnn_ig(torch_net, X):
        ig = IntegratedGradients(torch_net)
        test_input_tensor = torch.from_numpy(X).type(torch.FloatTensor)
        test_input_tensor.requires_grad_()
        attr, delta = ig.attribute(test_input_tensor, return_convergence_delta=True)
        attr = attr.detach().numpy()
        ig_scaled = normalise(np.abs(np.mean(attr, axis=0)))
        return ig_scaled
    
    
    train_dnn_ig_scaled = dnn_ig(net, X_train)
    dnn_ig_scaled = dnn_ig(net, X_test)
    
    ##########################################
    
    feature_names = list(column_names)
    def base_model():
        model = tf.keras.Sequential(
            [
                layers.Dense(
                    64, activation="tanh", input_shape=(X_train.shape[1],), bias_regularizer=tf.keras.regularizers.l2(0.01),
                ),
                layers.Dense(64, activation="relu", bias_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(32, activation="relu", bias_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(16, activation="relu", bias_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(8, activation="relu", bias_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(6, activation="relu", bias_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(4, activation="relu", bias_regularizer=tf.keras.regularizers.l2(0.01)),
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
    
    ##########################################
    # DNN PI
    ##########################################
    def dnn_pi(mdl, X, y):
        perm = PermutationImportance(mdl, scoring=mae_scorer).fit(X, y)
        pi_df = eli5.explain_weights_df(perm, feature_names=column_names.tolist())
        pi_df['feature_cat'] = pd.Categorical(
            pi_df['feature'], 
            categories=column_names.tolist(), 
            ordered=True)
        pi_df.sort_values(by='feature_cat', inplace=True)
        pi_df.drop(columns=['feature_cat'],inplace=True)
        pi_scaled = normalise(pi_df["weight"].values)
        return pi_scaled
    
    
    train_dnn_pi_scaled = dnn_pi(dnn_model, X_train, y_train)
    dnn_pi_scaled = dnn_pi(dnn_model, X_test, y_test)
    
    ##########################################
    
    model = tf.keras.Sequential(
        [
            layers.Dense(
                64, activation="tanh", input_shape=(X_train.shape[1],), bias_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            layers.Dense(64, activation="relu", bias_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(32, activation="relu", bias_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(16, activation="relu", bias_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(8, activation="relu", bias_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(6, activation="relu", bias_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(4, activation="relu", bias_regularizer=tf.keras.regularizers.l2(0.01)),
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
    
    ######################################
    # DNN SHAP
    ######################################
    def dnn_shap(mdl, X, samples):
        explainer_dnn = shap.KernelExplainer(mdl.predict, data=shap.kmeans(X, 10))
        shap_values = explainer_dnn.shap_values(X, nsamples=samples, l1_reg=f"num_features({NUM_FEATS})")
        sv = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)
        sv_scaled = normalise(sv)
        return sv_scaled
    
    train_dnn_sv_scaled = dnn_shap(model, X_train, 800)
    dnn_sv_scaled = dnn_shap(model, X_test, 200)
    
    ######################################
    
    # combine results into single array (train)
    train_rf_sv_scaled_reshaped = np.reshape(train_rf_sv_scaled, (train_rf_sv_scaled.shape[0], 1))
    train_pi_result_rf_scaled_reshaped = np.reshape(train_pi_result_rf_scaled, (train_pi_result_rf_scaled.shape[0], 1))
    
    train_gb_sv_scaled_reshaped = np.reshape(train_gb_sv_scaled, (train_gb_sv_scaled.shape[0], 1))
    train_pi_result_gb_scaled_reshaped = np.reshape(train_pi_result_gb_scaled, (train_pi_result_gb_scaled.shape[0], 1))
    
    train_svr_sv_scaled_reshaped = np.reshape(train_svr_sv_scaled, (train_svr_sv_scaled.shape[0], 1))
    train_pi_result_svr_scaled_reshaped = np.reshape(train_pi_result_svr_scaled, (train_pi_result_svr_scaled.shape[0], 1))
    
    train_dnn_pi_scaled_reshaped = np.reshape(train_dnn_pi_scaled, (train_dnn_pi_scaled.shape[0], 1))
    train_dnn_sv_scaled_reshaped = np.reshape(train_dnn_sv_scaled, (train_dnn_sv_scaled.shape[0], 1))
    train_dnn_ig_scaled_reshaped = np.reshape(train_dnn_ig_scaled, (train_dnn_ig_scaled.shape[0], 1))
    
    # combine results into single array (test)
    rf_sv_scaled_reshaped = np.reshape(rf_sv_scaled, (rf_sv_scaled.shape[0], 1))
    pi_result_rf_scaled_reshaped = np.reshape(pi_result_rf_scaled, (pi_result_rf_scaled.shape[0], 1))
    
    gb_sv_scaled_reshaped = np.reshape(gb_sv_scaled, (gb_sv_scaled.shape[0], 1))
    pi_result_gb_scaled_reshaped = np.reshape(pi_result_gb_scaled, (pi_result_gb_scaled.shape[0], 1))
    
    svr_sv_scaled_reshaped = np.reshape(svr_sv_scaled, (svr_sv_scaled.shape[0], 1))
    pi_result_svr_scaled_reshaped = np.reshape(pi_result_svr_scaled, (pi_result_svr_scaled.shape[0], 1))
    
    dnn_pi_scaled_reshaped = np.reshape(dnn_pi_scaled, (dnn_pi_scaled.shape[0], 1))
    dnn_sv_scaled_reshaped = np.reshape(dnn_sv_scaled, (dnn_sv_scaled.shape[0], 1))
    dnn_ig_scaled_reshaped = np.reshape(dnn_ig_scaled, (dnn_ig_scaled.shape[0], 1))
    
    def isNaN(num):
        return num != num
    if isNaN(train_dnn_ig_scaled_reshaped.mean()) == True:
        train_dnn_ig_scaled_reshaped = (train_dnn_pi_scaled_reshaped+train_dnn_sv_scaled_reshaped)/2
    if isNaN(dnn_ig_scaled_reshaped.mean()) == True:
        dnn_ig_scaled_reshaped = (dnn_pi_scaled_reshaped+dnn_sv_scaled_reshaped)/2
    
    # train
    train_all_stacked = np.hstack(
        (
            train_rf_sv_scaled_reshaped,
            train_pi_result_rf_scaled_reshaped,
            train_gb_sv_scaled_reshaped,
            train_pi_result_gb_scaled_reshaped,
            train_svr_sv_scaled_reshaped,
            train_pi_result_svr_scaled_reshaped,
            train_dnn_pi_scaled_reshaped,
            train_dnn_sv_scaled_reshaped,
            train_dnn_ig_scaled_reshaped,
        )
    )
    
    # test
    all_stacked = np.hstack(
        (
            rf_sv_scaled_reshaped,
            pi_result_rf_scaled_reshaped,
            gb_sv_scaled_reshaped,
            pi_result_gb_scaled_reshaped,
            svr_sv_scaled_reshaped,
            pi_result_svr_scaled_reshaped,
            dnn_pi_scaled_reshaped,
            dnn_sv_scaled_reshaped,
            dnn_ig_scaled_reshaped,
        )
    )


    all_stacked_df = pd.DataFrame(all_stacked, columns=["rf_sv", "rf_pi", "gb_sv", "gb_pi","svr_sv", "svr_pi", "dnn_pi", "dnn_sv", "dnn_ig"])

    all_stacked_df.to_csv("all_stacked.csv")
    
    def err_calculation(predicted, actual, calc_type):
        if calc_type=='mae':
            return np.abs(predicted - actual).mean()
        elif calc_type=='rmse':
            return np.sqrt(np.square(predicted - actual).mean())
        elif calc_type=='r2':
            return r2_score(actual,predicted)
    #####################################################################################
    # TRAIN DATA
    #####################################################################################
    
    
    train_err_rf_pi = err_calculation(train_pi_result_rf_scaled, scaled_coef, 'mae')
    train_err_gb_pi = err_calculation(train_pi_result_gb_scaled, scaled_coef, 'mae')
    train_err_svr_pi = err_calculation(train_pi_result_svr_scaled, scaled_coef, 'mae')
    train_err_dnn_pi = err_calculation(train_dnn_pi_scaled, scaled_coef, 'mae')
    train_err_rf_sv = err_calculation(train_rf_sv_scaled, scaled_coef, 'mae')
    train_err_gb_sv = err_calculation(train_gb_sv_scaled, scaled_coef, 'mae')
    train_err_svr_sv = err_calculation(train_svr_sv_scaled, scaled_coef, 'mae')
    train_err_dnn_sv = err_calculation(train_dnn_sv_scaled, scaled_coef, 'mae')
    train_err_dnn_ig = err_calculation(train_dnn_ig_scaled, scaled_coef, 'mae')
    
    train_rmse_err_rf_pi = err_calculation(train_pi_result_rf_scaled, scaled_coef, 'rmse')
    train_rmse_err_gb_pi = err_calculation(train_pi_result_gb_scaled, scaled_coef, 'rmse')
    train_rmse_err_svr_pi = err_calculation(train_pi_result_svr_scaled, scaled_coef, 'rmse')
    train_rmse_err_dnn_pi = err_calculation(train_dnn_pi_scaled, scaled_coef, 'rmse')
    train_rmse_err_rf_sv = err_calculation(train_rf_sv_scaled, scaled_coef, 'rmse')
    train_rmse_err_gb_sv = err_calculation(train_gb_sv_scaled, scaled_coef, 'rmse')
    train_rmse_err_svr_sv = err_calculation(train_svr_sv_scaled, scaled_coef, 'rmse')
    train_rmse_err_dnn_sv = err_calculation(train_dnn_sv_scaled, scaled_coef, 'rmse')
    train_rmse_err_dnn_ig = err_calculation(train_dnn_ig_scaled, scaled_coef, 'rmse')
    
    train_r2_err_rf_pi = err_calculation(train_pi_result_rf_scaled, scaled_coef, 'r2')
    train_r2_err_gb_pi = err_calculation(train_pi_result_gb_scaled, scaled_coef, 'r2')
    train_r2_err_svr_pi = err_calculation(train_pi_result_svr_scaled, scaled_coef, 'r2')
    train_r2_err_dnn_pi = err_calculation(train_dnn_pi_scaled, scaled_coef, 'r2')
    train_r2_err_rf_sv = err_calculation(train_rf_sv_scaled, scaled_coef, 'r2')
    train_r2_err_gb_sv = err_calculation(train_gb_sv_scaled, scaled_coef, 'r2')
    train_r2_err_svr_sv = err_calculation(train_svr_sv_scaled, scaled_coef, 'r2')
    train_r2_err_dnn_sv = err_calculation(train_dnn_sv_scaled, scaled_coef, 'r2')
    train_r2_err_dnn_ig = err_calculation(train_dnn_ig_scaled, scaled_coef, 'r2')
    
    print('MAE train_ERROR:')
    print(f"RF PI MAE train_error: {train_err_rf_pi}")
    print(f"GB PI MAE train_error: {train_err_gb_pi}")
    print(f"SVR PI MAE train_error: {train_err_svr_pi}")
    print(f"DNN PI MAE train_error: {train_err_dnn_pi}")
    print("")
    print(f"RF SV MAE train_error: {train_err_rf_sv}")
    print(f"GB SV MAE train_error: {train_err_gb_sv}")
    print(f"SVR SV MAE train_error: {train_err_svr_sv}")
    print(f"DNN SV MAE train_error: {train_err_dnn_sv}")
    print(f"DNN IG MAE train_error: {train_err_dnn_ig}")
    print("")
    print(f"PI MAE train_error: {(train_err_rf_pi+train_err_gb_pi+train_err_svr_pi+train_err_dnn_pi)/4}")
    print(f"SV MAE train_error: {(train_err_rf_sv+train_err_gb_sv+train_err_svr_sv+train_err_dnn_sv)/4}")
    print(f"IG MAE train_error: {train_err_dnn_ig}")
    print("")
    print("")
    
    print('RMSE train_ERROR:')
    print(f"RF PI RMSE train_error: {train_rmse_err_rf_pi}")
    print(f"GB PI RMSE train_error: {train_rmse_err_gb_pi}")
    print(f"SVR PI RMSE train_error: {train_rmse_err_svr_pi}")
    print(f"DNN PI RMSE train_error: {train_rmse_err_dnn_pi}")
    print("")
    print(f"RF SV RMSE train_error: {train_rmse_err_rf_sv}")
    print(f"GB SV RMSE train_error: {train_rmse_err_gb_sv}")
    print(f"SVR SV RMSE train_error: {train_rmse_err_svr_sv}")
    print(f"DNN SV RMSE train_error: {train_rmse_err_dnn_sv}")
    print(f"DNN IG RMSE train_error: {train_rmse_err_dnn_ig}")
    print("")
    print(f"PI RMSE train_error: {(train_rmse_err_rf_pi+train_rmse_err_gb_pi+train_rmse_err_svr_pi+train_rmse_err_dnn_pi)/4}")
    print(f"SV RMSE train_error: {(train_rmse_err_rf_sv+train_rmse_err_gb_sv+train_rmse_err_svr_sv+train_rmse_err_dnn_sv)/4}")
    print(f"IG RMSE train_error: {train_rmse_err_dnn_ig}")
    print("")
    print("")
    
    print('R2 ERROR:')
    print(f"RF PI R2 train_error: {train_r2_err_rf_pi}")
    print(f"GB PI R2 train_error: {train_r2_err_gb_pi}")
    print(f"SVR PI R2 train_error: {train_r2_err_svr_pi}")
    print(f"DNN PI R2 train_error: {train_r2_err_dnn_pi}")
    print("")
    print(f"RF SV R2 train_error: {train_r2_err_rf_sv}")
    print(f"GB SV R2 train_error: {train_r2_err_gb_sv}")
    print(f"SVR SV R2 train_error: {train_r2_err_svr_sv}")
    print(f"DNN SV R2 train_error: {train_r2_err_dnn_sv}")
    print(f"DNN IG R2 train_error: {train_r2_err_dnn_ig}")
    print("")
    print(f"PI R2 train_error: {(train_r2_err_rf_pi+train_r2_err_gb_pi+train_r2_err_svr_pi+train_r2_err_dnn_pi)/4}")
    print(f"SV R2 train_error: {(train_r2_err_rf_sv+train_r2_err_gb_sv+train_r2_err_svr_sv+train_r2_err_dnn_sv)/4}")
    print(f"IG R2 train_error: {train_r2_err_dnn_ig}")
    print("")
    print("")
    
    #####################################################################################
    #####################################################################################
    
    #####################################################################################
    # TEST DATA
    #####################################################################################
    
    
    err_rf_pi = err_calculation(pi_result_rf_scaled, scaled_coef, 'mae')
    err_gb_pi = err_calculation(pi_result_gb_scaled, scaled_coef, 'mae')
    err_svr_pi = err_calculation(pi_result_svr_scaled, scaled_coef, 'mae')
    err_dnn_pi = err_calculation(dnn_pi_scaled, scaled_coef, 'mae')
    err_rf_sv = err_calculation(rf_sv_scaled, scaled_coef, 'mae')
    err_gb_sv = err_calculation(gb_sv_scaled, scaled_coef, 'mae')
    err_svr_sv = err_calculation(svr_sv_scaled, scaled_coef, 'mae')
    err_dnn_sv = err_calculation(dnn_sv_scaled, scaled_coef, 'mae')
    err_dnn_ig = err_calculation(dnn_ig_scaled, scaled_coef, 'mae')
    
    rmse_err_rf_pi = err_calculation(pi_result_rf_scaled, scaled_coef, 'rmse')
    rmse_err_gb_pi = err_calculation(pi_result_gb_scaled, scaled_coef, 'rmse')
    rmse_err_svr_pi = err_calculation(pi_result_svr_scaled, scaled_coef, 'rmse')
    rmse_err_dnn_pi = err_calculation(dnn_pi_scaled, scaled_coef, 'rmse')
    rmse_err_rf_sv = err_calculation(rf_sv_scaled, scaled_coef, 'rmse')
    rmse_err_gb_sv = err_calculation(gb_sv_scaled, scaled_coef, 'rmse')
    rmse_err_svr_sv = err_calculation(svr_sv_scaled, scaled_coef, 'rmse')
    rmse_err_dnn_sv = err_calculation(dnn_sv_scaled, scaled_coef, 'rmse')
    rmse_err_dnn_ig = err_calculation(dnn_ig_scaled, scaled_coef, 'rmse')
    
    r2_err_rf_pi = err_calculation(pi_result_rf_scaled, scaled_coef, 'r2')
    r2_err_gb_pi = err_calculation(pi_result_gb_scaled, scaled_coef, 'r2')
    r2_err_svr_pi = err_calculation(pi_result_svr_scaled, scaled_coef, 'r2')
    r2_err_dnn_pi = err_calculation(dnn_pi_scaled, scaled_coef, 'r2')
    r2_err_rf_sv = err_calculation(rf_sv_scaled, scaled_coef, 'r2')
    r2_err_gb_sv = err_calculation(gb_sv_scaled, scaled_coef, 'r2')
    r2_err_svr_sv = err_calculation(svr_sv_scaled, scaled_coef, 'r2')
    r2_err_dnn_sv = err_calculation(dnn_sv_scaled, scaled_coef, 'r2')
    r2_err_dnn_ig = err_calculation(dnn_ig_scaled, scaled_coef, 'r2')
    
    print('MAE ERROR:')
    print(f"RF PI MAE error: {err_rf_pi}")
    print(f"GB PI MAE error: {err_gb_pi}")
    print(f"SVR PI MAE error: {err_svr_pi}")
    print(f"DNN PI MAE error: {err_dnn_pi}")
    print("")
    print(f"RF SV MAE error: {err_rf_sv}")
    print(f"GB SV MAE error: {err_gb_sv}")
    print(f"SVR SV MAE error: {err_svr_sv}")
    print(f"DNN SV MAE error: {err_dnn_sv}")
    print(f"DNN IG MAE error: {err_dnn_ig}")
    print("")
    print(f"PI MAE error: {(err_rf_pi+err_gb_pi+err_svr_pi+err_dnn_pi)/4}")
    print(f"SV MAE error: {(err_rf_sv+err_gb_sv+err_svr_sv+err_dnn_sv)/4}")
    print(f"IG MAE error: {err_dnn_ig}")
    print("")
    print("")
    
    print('RMSE ERROR:')
    print(f"RF PI RMSE error: {rmse_err_rf_pi}")
    print(f"GB PI RMSE error: {rmse_err_gb_pi}")
    print(f"SVR PI RMSE error: {rmse_err_svr_pi}")
    print(f"DNN PI RMSE error: {rmse_err_dnn_pi}")
    print("")
    print(f"RF SV RMSE error: {rmse_err_rf_sv}")
    print(f"GB SV RMSE error: {rmse_err_gb_sv}")
    print(f"SVR SV RMSE error: {rmse_err_svr_sv}")
    print(f"DNN SV RMSE error: {rmse_err_dnn_sv}")
    print(f"DNN IG RMSE error: {rmse_err_dnn_ig}")
    print("")
    print(f"PI RMSE error: {(rmse_err_rf_pi+rmse_err_gb_pi+rmse_err_svr_pi+rmse_err_dnn_pi)/4}")
    print(f"SV RMSE error: {(rmse_err_rf_sv+rmse_err_gb_sv+rmse_err_svr_sv+rmse_err_dnn_sv)/4}")
    print(f"IG RMSE error: {rmse_err_dnn_ig}")
    print("")
    print("")
    
    print('R2 ERROR:')
    print(f"RF PI R2 error: {r2_err_rf_pi}")
    print(f"GB PI R2 error: {r2_err_gb_pi}")
    print(f"SVR PI R2 error: {r2_err_svr_pi}")
    print(f"DNN PI R2 error: {r2_err_dnn_pi}")
    print("")
    print(f"RF SV R2 error: {r2_err_rf_sv}")
    print(f"GB SV R2 error: {r2_err_gb_sv}")
    print(f"SVR SV R2 error: {r2_err_svr_sv}")
    print(f"DNN SV R2 error: {r2_err_dnn_sv}")
    print(f"DNN IG R2 error: {r2_err_dnn_ig}")
    print("")
    print(f"PI R2 error: {(r2_err_rf_pi+r2_err_gb_pi+r2_err_svr_pi+r2_err_dnn_pi)/4}")
    print(f"SV R2 error: {(r2_err_rf_sv+r2_err_gb_sv+r2_err_svr_sv+r2_err_dnn_sv)/4}")
    print(f"IG R2 error: {r2_err_dnn_ig}")
    print("")
    print("")
    
    #####################################################################################
    #####################################################################################
    
    ###################################################################################################
    # median
    ###################################################################################################
    total_scaled_median = np.median(all_stacked, axis=1)
    train_total_scaled_median = np.median(train_all_stacked, axis=1)
    
    ###################################################################################################
    
    ###################################################################################################
    # mean 
    ###################################################################################################
    total_scaled_mean = np.mean(all_stacked, axis=1)
    train_total_scaled_mean = np.mean(train_all_stacked, axis=1)
    ###################################################################################################
    
    ###################################################################################################
    # Box and whiskers
    ###################################################################################################
    def BaW(stacked):
        total_scaled_baw = np.array([])
        for i in range(stacked.shape[0]):
            temp_whiskers = np.array([])
            q3 = np.quantile(stacked[i, :], 0.75)
            q1 = np.quantile(stacked[i, :], 0.25)
            upper_whiskers = q3 + (1.5 * (q3 - q1))
            lower_whiskers = q1 - (1.5 * (q3 - q1))
            for j in range(stacked[i, :].shape[0]):
                if (stacked[i, :][j] >= lower_whiskers) and (stacked[i, :][j] <= upper_whiskers):
                    temp_whiskers = np.append(temp_whiskers, stacked[i, :][j])
            total_scaled_baw = np.append(total_scaled_baw, temp_whiskers.mean())
        return total_scaled_baw
    
    total_scaled_box_whiskers = BaW(all_stacked)
    train_total_scaled_box_whiskers = BaW(train_all_stacked)
    ###################################################################################################
    
    ###################################################################################################
    # Thompson Tau
    ###################################################################################################
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

    # test
    total_scaled_tau_test = np.array([])
    for i in range(all_stacked.shape[0]):
        mean_tau = np.array([tau_test(all_stacked[i, :]).mean()])
        total_scaled_tau_test = np.append(total_scaled_tau_test, mean_tau)

    total_scaled_tau_test = np.array([total_scaled_tau_test])
    total_scaled_tau_test = np.reshape(total_scaled_tau_test, (-1,))
    #train
    train_total_scaled_tau_test = np.array([])
    for i in range(train_all_stacked.shape[0]):
        train_mean_tau = np.array([tau_test(train_all_stacked[i, :]).mean()])
        train_total_scaled_tau_test = np.append(train_total_scaled_tau_test, train_mean_tau)

    train_total_scaled_tau_test = np.array([train_total_scaled_tau_test])
    train_total_scaled_tau_test = np.reshape(train_total_scaled_tau_test, (-1,))
    ###################################################################################################
    
    ###################################################################################################
    # Mode
    ###################################################################################################
    # test
    total_scaled_mode = np.array([])
    for i in range(all_stacked.shape[0]):
        params = stats.norm.fit(all_stacked[i, :])

        def your_density(x):
            return -stats.norm.pdf(x, *params)

        total_scaled_mode = np.append(total_scaled_mode, minimize(your_density, 0).x[0])

    total_scaled_mode = np.reshape(total_scaled_mode, (-1,))
    # train
    train_total_scaled_mode = np.array([])
    for i in range(train_all_stacked.shape[0]):
        params = stats.norm.fit(train_all_stacked[i, :])

        def your_density(x):
            return -stats.norm.pdf(x, *params)

        train_total_scaled_mode = np.append(train_total_scaled_mode, minimize(your_density, 0).x[0])
        
    train_total_scaled_mode = np.reshape(train_total_scaled_mode, (-1,))
    ###################################################################################################
    
    ###################################################################################################
    # majority vote
    ###################################################################################################
    total_scaled_majority_vote = majority_vote_func(all_stacked, NUM_FEATS)
    train_total_scaled_majority_vote = majority_vote_func(train_all_stacked, NUM_FEATS)
    ###################################################################################################

    ###################################################################################################
    # kendall tau
    ###################################################################################################
    columns_name = ["rf_sv", "rf_pi", "gb_sv", "gb_pi", "svr_sv", "svr_pi", "dnn_pi", "dnn_sv", "dnn_ig"]
    total_scaled_kendall_tau, kendall_tau_p_value, non_sig_ratio = helper_rank_func(stats.kendalltau, all_stacked, columns_name, NUM_FEATS)
    
    train_total_scaled_kendall_tau, train_kendall_tau_p_value, train_non_sig_ratio = helper_rank_func(stats.kendalltau, train_all_stacked, columns_name, NUM_FEATS)
    ###################################################################################################
    
    ###################################################################################################
    # spearman
    ###################################################################################################
    total_scaled_spearman, spearman_p_value, non_sig_ratio = helper_rank_func(stats.spearmanr, all_stacked, columns_name, NUM_FEATS)
    
    train_total_scaled_spearman, train_spearman_p_value, train_non_sig_ratio = helper_rank_func(stats.spearmanr, train_all_stacked, columns_name, NUM_FEATS)
    ###################################################################################################

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

    print("Train Plot")
    plot_rank_method_significance(train_kendall_tau_p_value, "Kendall Tau", 0.05, columns_name)
    plot_rank_method_significance(train_spearman_p_value, "Spearman Rho", 0.05,columns_name)
    print("Test Plot")
    plot_rank_method_significance(kendall_tau_p_value, "Kendall Tau", 0.05, columns_name)
    plot_rank_method_significance(spearman_p_value, "Spearman Rho", 0.05,columns_name)
          
    # renormalise the result (train)
    train_total_scaled_median = normalise(train_total_scaled_median)
    train_total_scaled_mean = normalise(train_total_scaled_mean)
    train_total_scaled_mode = normalise(train_total_scaled_mode)
    train_total_scaled_box_whiskers = normalise(train_total_scaled_box_whiskers)
    train_total_scaled_tau_test = normalise(train_total_scaled_tau_test)
    train_total_scaled_majority_vote = normalise(train_total_scaled_majority_vote)
    train_total_scaled_kendall_tau = normalise(train_total_scaled_kendall_tau)
    train_total_scaled_spearman = normalise(train_total_scaled_spearman)
          
    # renormalise the result (test)
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
    
     
          
          
    ##############################################################################################
    # Print all final result (train)
    ##############################################################################################
    # mae (train)
    train_err_mean = err_calculation(train_total_scaled_mean, scaled_coef, 'mae')
    train_err_median = err_calculation(train_total_scaled_median, scaled_coef, 'mae')
    train_err_mode = err_calculation(train_total_scaled_mode, scaled_coef, 'mae')
    train_err_box = err_calculation(train_total_scaled_box_whiskers, scaled_coef, 'mae')
    train_err_tau = err_calculation(train_total_scaled_tau_test, scaled_coef, 'mae')
    train_err_major = err_calculation(train_total_scaled_majority_vote, scaled_coef, 'mae')
    train_err_kendall = err_calculation(train_total_scaled_kendall_tau, scaled_coef, 'mae')
    train_err_spearman = err_calculation(train_total_scaled_spearman, scaled_coef, 'mae')
    
    # rmse (train)
    train_rmse_err_mean = err_calculation(train_total_scaled_mean, scaled_coef, 'rmse')
    train_rmse_err_median = err_calculation(train_total_scaled_median, scaled_coef, 'rmse')
    train_rmse_err_mode = err_calculation(train_total_scaled_mode, scaled_coef, 'rmse')
    train_rmse_err_box = err_calculation(train_total_scaled_box_whiskers, scaled_coef, 'rmse')
    train_rmse_err_tau = err_calculation(train_total_scaled_tau_test, scaled_coef, 'rmse')
    train_rmse_err_major = err_calculation(train_total_scaled_majority_vote, scaled_coef, 'rmse')
    train_rmse_err_kendall = err_calculation(train_total_scaled_kendall_tau, scaled_coef, 'rmse')
    train_rmse_err_spearman = err_calculation(train_total_scaled_spearman, scaled_coef, 'rmse')
    
    # r2 (train)
    train_r2_err_mean = err_calculation(train_total_scaled_mean, scaled_coef, 'r2')
    train_r2_err_median = err_calculation(train_total_scaled_median, scaled_coef, 'r2')
    train_r2_err_mode = err_calculation(train_total_scaled_mode, scaled_coef, 'r2')
    train_r2_err_box = err_calculation(train_total_scaled_box_whiskers, scaled_coef, 'r2')
    train_r2_err_tau = err_calculation(train_total_scaled_tau_test, scaled_coef, 'r2')
    train_r2_err_major = err_calculation(train_total_scaled_majority_vote, scaled_coef, 'r2')
    train_r2_err_kendall = err_calculation(train_total_scaled_kendall_tau, scaled_coef, 'r2')
    train_r2_err_spearman = err_calculation(train_total_scaled_spearman, scaled_coef, 'r2')
          
    ##############################################################################################
    # Print all final result (test)
    ##############################################################################################
    # mae (test)
    err_mean = err_calculation(total_scaled_mean, scaled_coef, 'mae')
    err_median = err_calculation(total_scaled_median, scaled_coef, 'mae')
    err_mode = err_calculation(total_scaled_mode, scaled_coef, 'mae')
    err_box = err_calculation(total_scaled_box_whiskers, scaled_coef, 'mae')
    err_tau = err_calculation(total_scaled_tau_test, scaled_coef, 'mae')
    err_major = err_calculation(total_scaled_majority_vote, scaled_coef, 'mae')
    err_kendall = err_calculation(total_scaled_kendall_tau, scaled_coef, 'mae')
    err_spearman = err_calculation(total_scaled_spearman, scaled_coef, 'mae')
    
    # rmse (test)
    rmse_err_mean = err_calculation(total_scaled_mean, scaled_coef, 'rmse')
    rmse_err_median = err_calculation(total_scaled_median, scaled_coef, 'rmse')
    rmse_err_mode = err_calculation(total_scaled_mode, scaled_coef, 'rmse')
    rmse_err_box = err_calculation(total_scaled_box_whiskers, scaled_coef, 'rmse')
    rmse_err_tau = err_calculation(total_scaled_tau_test, scaled_coef, 'rmse')
    rmse_err_major = err_calculation(total_scaled_majority_vote, scaled_coef, 'rmse')
    rmse_err_kendall = err_calculation(total_scaled_kendall_tau, scaled_coef, 'rmse')
    rmse_err_spearman = err_calculation(total_scaled_spearman, scaled_coef, 'rmse')
    
    # r2 (test)
    r2_err_mean = err_calculation(total_scaled_mean, scaled_coef, 'r2')
    r2_err_median = err_calculation(total_scaled_median, scaled_coef, 'r2')
    r2_err_mode = err_calculation(total_scaled_mode, scaled_coef, 'r2')
    r2_err_box = err_calculation(total_scaled_box_whiskers, scaled_coef, 'r2')
    r2_err_tau = err_calculation(total_scaled_tau_test, scaled_coef, 'r2')
    r2_err_major = err_calculation(total_scaled_majority_vote, scaled_coef, 'r2')
    r2_err_kendall = err_calculation(total_scaled_kendall_tau, scaled_coef, 'r2')
    r2_err_spearman = err_calculation(total_scaled_spearman, scaled_coef, 'r2')
    ##############################################################################################
          
    ##############################################################################################
    # Print all final result (train)
    ##############################################################################################
    err_type = 'MAE'
    train_or_test = 'train'
    print(f" All {train_or_test} {err_type} (mean): {train_err_mean}")
    print(f" All {train_or_test} {err_type} (median): {train_err_median}")
    print(f" All {train_or_test} {err_type} (mode): {train_err_mode}")
    print(f" All {train_or_test} {err_type} (box-whiskers): {train_err_box}")
    print(f" All {train_or_test} {err_type} (tau-test): {train_err_tau}")
    print(f" All {train_or_test} {err_type} (majority vote): {train_err_major}")
    print(f" All {train_or_test} {err_type} (Kendall tau): {train_err_kendall}")
    print(f" All {train_or_test} {err_type} (Spearman Rho): {train_err_spearman}")
    print("")
    print(f"PI {train_or_test} {err_type}: {(train_err_rf_pi+train_err_gb_pi+train_err_svr_pi+train_err_dnn_pi)/4}")
    print(f"SV {train_or_test} {err_type}: {(train_err_rf_sv+train_err_gb_sv+train_err_svr_sv+train_err_dnn_sv)/4}")
    print(f"IG {train_or_test} {err_type}: {train_err_dnn_ig}")
    print("")
    print("")
    
    err_type = 'RMSE'
    print(f" All {train_or_test} {err_type} (mean): {train_rmse_err_mean}")
    print(f" All {train_or_test} {err_type} (median): {train_rmse_err_median}")
    print(f" All {train_or_test} {err_type} (mode): {train_rmse_err_mode}")
    print(f" All {train_or_test} {err_type} (box-whiskers): {train_rmse_err_box}")
    print(f" All {train_or_test} {err_type} (tau-test): {train_rmse_err_tau}")
    print(f" All {train_or_test} {err_type} (majority vote): {train_rmse_err_major}")
    print(f" All {train_or_test} {err_type} (Kendall tau): {train_rmse_err_kendall}")
    print(f" All {train_or_test} {err_type} (Spearman Rho): {train_rmse_err_spearman}")
    print("")
    print(f"PI {train_or_test} {err_type}: {(train_rmse_err_rf_pi+train_rmse_err_gb_pi+train_rmse_err_svr_pi+train_rmse_err_dnn_pi)/4}")
    print(f"SV {train_or_test} {err_type}: {(train_rmse_err_rf_sv+train_rmse_err_gb_sv+train_rmse_err_svr_sv+train_rmse_err_dnn_sv)/4}")
    print(f"IG {train_or_test} {err_type}: {train_rmse_err_dnn_ig}")
    print("")
    print("")
    
    err_type = 'R2'
    print(f" All {train_or_test} {err_type} (mean): {train_r2_err_mean}")
    print(f" All {train_or_test} {err_type} (median): {train_r2_err_median}")
    print(f" All {train_or_test} {err_type} (mode): {train_r2_err_mode}")
    print(f" All {train_or_test} {err_type} (box-whiskers): {train_r2_err_box}")
    print(f" All {train_or_test} {err_type} (tau-test): {train_r2_err_tau}")
    print(f" All {train_or_test} {err_type} (majority vote): {train_r2_err_major}")
    print(f" All {train_or_test} {err_type} (Kendall tau): {train_r2_err_kendall}")
    print(f" All {train_or_test} {err_type} (Spearman Rho): {train_r2_err_spearman}")
    print("")
    print(f"PI {train_or_test} {err_type}: {(train_r2_err_rf_pi+train_r2_err_gb_pi+train_r2_err_svr_pi+train_r2_err_dnn_pi)/4}")
    print(f"SV {train_or_test} {err_type}: {(train_r2_err_rf_sv+train_err_gb_sv+train_r2_err_svr_sv+train_r2_err_dnn_sv)/4}")
    print(f"IG {train_or_test} {err_type}: {train_r2_err_dnn_ig}")
    print("")
    print("")
          
    ##############################################################################################
    
    ##############################################################################################
    # Print all final result (test)
    ##############################################################################################
    err_type = 'MAE'
    train_or_test = 'test'
    print(f" All {train_or_test} {err_type} (mean): {err_mean}")
    print(f" All {train_or_test} {err_type} (median): {err_median}")
    print(f" All {train_or_test} {err_type} (mode): {err_mode}")
    print(f" All {train_or_test} {err_type} (box-whiskers): {err_box}")
    print(f" All {train_or_test} {err_type} (tau-test): {err_tau}")
    print(f" All {train_or_test} {err_type} (majority vote): {err_major}")
    print(f" All {train_or_test} {err_type} (Kendall tau): {err_kendall}")
    print(f" All {train_or_test} {err_type} (Spearman Rho): {err_spearman}")
    print("")
    print(f"PI {train_or_test} {err_type}: {(err_rf_pi+err_gb_pi+err_svr_pi+err_dnn_pi)/4}")
    print(f"SV {train_or_test} {err_type}: {(err_rf_sv+err_gb_sv+err_svr_sv+err_dnn_sv)/4}")
    print(f"IG {train_or_test} {err_type}: {err_dnn_ig}")
    print("")
    print("")
    
    err_type = 'RMSE'
    print(f" All {train_or_test} {err_type} (mean): {rmse_err_mean}")
    print(f" All {train_or_test} {err_type} (median): {rmse_err_median}")
    print(f" All {train_or_test} {err_type} (mode): {rmse_err_mode}")
    print(f" All {train_or_test} {err_type} (box-whiskers): {rmse_err_box}")
    print(f" All {train_or_test} {err_type} (tau-test): {rmse_err_tau}")
    print(f" All {train_or_test} {err_type} (majority vote): {rmse_err_major}")
    print(f" All {train_or_test} {err_type} (Kendall tau): {rmse_err_kendall}")
    print(f" All {train_or_test} {err_type} (Spearman Rho): {rmse_err_spearman}")
    print("")
    print(f"PI {train_or_test} {err_type}: {(rmse_err_rf_pi+rmse_err_gb_pi+rmse_err_svr_pi+rmse_err_dnn_pi)/4}")
    print(f"SV {train_or_test} {err_type}: {(rmse_err_rf_sv+rmse_err_gb_sv+rmse_err_svr_sv+rmse_err_dnn_sv)/4}")
    print(f"IG {train_or_test} {err_type}: {rmse_err_dnn_ig}")
    print("")
    print("")
    
    err_type = 'R2'
    print(f" All {train_or_test} {err_type} (mean): {r2_err_mean}")
    print(f" All {train_or_test} {err_type} (median): {r2_err_median}")
    print(f" All {train_or_test} {err_type} (mode): {r2_err_mode}")
    print(f" All {train_or_test} {err_type} (box-whiskers): {r2_err_box}")
    print(f" All {train_or_test} {err_type} (tau-test): {r2_err_tau}")
    print(f" All {train_or_test} {err_type} (majority vote): {r2_err_major}")
    print(f" All {train_or_test} {err_type} (Kendall tau): {r2_err_kendall}")
    print(f" All {train_or_test} {err_type} (Spearman Rho): {r2_err_spearman}")
    print("")
    print(f"PI {train_or_test} {err_type}: {(r2_err_rf_pi+r2_err_gb_pi+r2_err_svr_pi+r2_err_dnn_pi)/4}")
    print(f"SV {train_or_test} {err_type}: {(r2_err_rf_sv+err_gb_sv+r2_err_svr_sv+r2_err_dnn_sv)/4}")
    print(f"IG {train_or_test} {err_type}: {r2_err_dnn_ig}")
    print("")
    print("")
          
    ##############################################################################################
          
    ##############################################################################################
    # display result in plots (train)
    ##############################################################################################
    methods = np.array(
        ["Mode", "Median", "Mean", "Box-Whiskers", "Tau Test", "Majority Vote", "Kendall Tau", "Spearman Rho", "Actual"]
    )
    train_multiplied_importance = np.append(
        [train_total_scaled_mode],
        [
            train_total_scaled_median,
            train_total_scaled_mean,
            train_total_scaled_box_whiskers,
            train_total_scaled_tau_test,
            train_total_scaled_majority_vote,
            train_total_scaled_kendall_tau,
            train_total_scaled_spearman,
            scaled_coef,
        ],
    )
    train_multiplied_feature = np.tile(feature_names, int(train_multiplied_importance.shape[0] / len(feature_names)))
    train_multiplied_methods = np.repeat(
        methods, int(train_multiplied_importance.shape[0] / methods.shape[0])
    )  # for 9 different ensemble method


    train_df_results = pd.DataFrame(
        {"Importance": train_multiplied_importance, "Features": train_multiplied_feature, "Methods": train_multiplied_methods}
    )


    #df_results.to_csv(f"Tabular/results_{informative}_feat_{noise}_noise.csv")

    fig = px.scatter(train_df_results, x="Features", y="Importance", color="Methods")

    fig.update_layout(
        template="plotly_white", title="Feature importance of actual vs ensemble method (Train)", width=1500, height=800,
    )

    fig.update_traces(marker=dict(size=12, line=dict(width=2, color="Black")), selector=dict(mode="markers"), opacity=0.7)

    #fig.write_html(f"Images/scatter_results_{informative}_feat_{noise}_noise.html")
    fig.show()
    ##############################################################################################
          
    ##############################################################################################
    # display result in plots (test)
    ##############################################################################################
    methods = np.array(
        ["Mode", "Median", "Mean", "Box-Whiskers", "Tau Test", "Majority Vote", "Kendall Tau", "Spearman Rho", "Actual"]
    )
    multiplied_importance = np.append(
        [total_scaled_mode],
        [
            total_scaled_median,
            total_scaled_mean,
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


    #df_results.to_csv(f"Tabular/results_{informative}_feat_{noise}_noise.csv")

    fig = px.scatter(df_results, x="Features", y="Importance", color="Methods")

    fig.update_layout(
        template="plotly_white", title="Feature importance of actual vs ensemble method (Test)", width=1500, height=800,
    )

    fig.update_traces(marker=dict(size=12, line=dict(width=2, color="Black")), selector=dict(mode="markers"), opacity=0.7)

    #fig.write_html(f"Images/scatter_results_{informative}_feat_{noise}_noise.html")
    fig.show()
    ##############################################################################################
    
    
    mae_result = [rf_mae, gb_mae, svr_mae, dnn_mae, err_rf_pi, err_gb_pi,err_svr_pi, err_dnn_pi, err_rf_sv, err_gb_sv,err_svr_sv, err_dnn_sv, err_dnn_ig, err_mean, err_median, err_mode, err_box, err_tau, err_major, err_kendall, err_spearman]
    
    rmse_result = [rf_rmse, gb_rmse, svr_rmse, dnn_rmse, rmse_err_rf_pi, rmse_err_gb_pi,rmse_err_svr_pi, rmse_err_dnn_pi, rmse_err_rf_sv, rmse_err_gb_sv,rmse_err_svr_sv, rmse_err_dnn_sv, rmse_err_dnn_ig, rmse_err_mean, rmse_err_median, rmse_err_mode, rmse_err_box, rmse_err_tau, rmse_err_major, rmse_err_kendall, rmse_err_spearman]
    
    r2_result = [r2_err_rf_pi, r2_err_gb_pi,r2_err_svr_pi, r2_err_dnn_pi, r2_err_rf_sv, r2_err_gb_sv,r2_err_svr_sv, r2_err_dnn_sv, r2_err_dnn_ig, r2_err_mean, r2_err_median, r2_err_mode, r2_err_box, r2_err_tau, r2_err_major, r2_err_kendall, r2_err_spearman]
          
    train_mae_result = [train_err_rf_pi, train_err_gb_pi,train_err_svr_pi, train_err_dnn_pi, train_err_rf_sv, train_err_gb_sv,train_err_svr_sv, train_err_dnn_sv, train_err_dnn_ig, train_err_mean, train_err_median, train_err_mode, train_err_box, train_err_tau, train_err_major, train_err_kendall, train_err_spearman]
    
    train_rmse_result = [train_rmse_err_rf_pi, train_rmse_err_gb_pi,train_rmse_err_svr_pi, train_rmse_err_dnn_pi, train_rmse_err_rf_sv, train_rmse_err_gb_sv,train_rmse_err_svr_sv, train_rmse_err_dnn_sv, train_rmse_err_dnn_ig, train_rmse_err_mean, train_rmse_err_median, train_rmse_err_mode, train_rmse_err_box, train_rmse_err_tau, train_rmse_err_major, train_rmse_err_kendall, train_rmse_err_spearman]
    
    train_r2_result = [train_r2_err_rf_pi, train_r2_err_gb_pi,train_r2_err_svr_pi, train_r2_err_dnn_pi, train_r2_err_rf_sv, train_r2_err_gb_sv,train_r2_err_svr_sv, train_r2_err_dnn_sv, train_r2_err_dnn_ig, train_r2_err_mean, train_r2_err_median, train_r2_err_mode, train_r2_err_box, train_r2_err_tau, train_r2_err_major, train_r2_err_kendall, train_r2_err_spearman]
    
    return [mae_result,rmse_result,r2_result, train_mae_result, train_rmse_result, train_r2_result]