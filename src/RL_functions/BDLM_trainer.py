from examples.data_loader import TimeSeriesDataloader
import numpy as np
import pandas as pd
from pytagi import Normalizer as normalizer
from matplotlib import gridspec
import matplotlib.pyplot as plt
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential
from pytagi.hybrid import *
from tqdm import tqdm
import pytagi.metric as metric
import copy
plt.rcParams["figure.autolayout"] = True

class BDLM_trainer:
    def __init__(self, **kwargs):
        self.num_epochs = kwargs.get('num_epochs', None)
        self.batch_size = kwargs.get('batch_size', None)
        self.sigma_v = 1E-12
        self.output_col = [0]
        self.num_features = kwargs.get('num_features', None)
        self.input_seq_len = kwargs.get('input_seq_len', None)
        self.output_seq_len = kwargs.get('output_seq_len', None)
        self.seq_stride = kwargs.get('seq_stride', None)

        self.Sigma_AA_ratio = kwargs.get('Sigma_AA_ratio', 1e-10)
        self.phi_AA = kwargs.get('phi_AA', 0.99)
        self.components = kwargs.get('components', 'AA + AR')
        self.use_auto_AR = kwargs.get('use_auto_AR', True)

    def load_datasets(self, observation_file_paths, datetime_file_paths, time_covariates):
        self.observation_file_paths = observation_file_paths
        self.datetime_file_paths = datetime_file_paths
        self.time_covariates = time_covariates
        self.train_dtl = TimeSeriesDataloader(
                            x_file = observation_file_paths[0],
                            date_time_file = datetime_file_paths[0],
                            output_col = self.output_col,
                            input_seq_len = self.input_seq_len,
                            output_seq_len = self.output_seq_len,
                            num_features = self.num_features,
                            stride = self.seq_stride,
                            time_covariates = time_covariates,  # 'hour_of_day','day_of_week', 'week_of_year', 'month_of_year','quarter_of_year', 'day_of_year'
                        )
        self.val_dtl = TimeSeriesDataloader(
                            x_file = observation_file_paths[1],
                            date_time_file = datetime_file_paths[1],
                            output_col = self.output_col,
                            input_seq_len = self.input_seq_len,
                            output_seq_len = self.output_seq_len,
                            num_features = self.num_features,
                            stride = self.seq_stride,
                            x_mean=self.train_dtl.x_mean,
                            x_std=self.train_dtl.x_std,
                            time_covariates = time_covariates,  # 'hour_of_day','day_of_week', 'week_of_year', 'month_of_year','quarter_of_year', 'day_of_year'
                        )
        self.test_dtl = TimeSeriesDataloader(
                            x_file = observation_file_paths[2],
                            date_time_file = datetime_file_paths[2],
                            output_col = self.output_col,
                            input_seq_len = self.input_seq_len,
                            output_seq_len = self.output_seq_len,
                            num_features = self.num_features,
                            stride = self.seq_stride,
                            x_mean=self.train_dtl.x_mean,
                            x_std=self.train_dtl.x_std,
                            time_covariates = time_covariates,  # 'hour_of_day','day_of_week', 'week_of_year', 'month_of_year','quarter_of_year', 'day_of_year'
                        )

    def estimate_initial_baseline(self, plot = False):
        x_data = np.array(pd.read_csv(self.observation_file_paths[0], skiprows=1, delimiter=",", header=None).values.T[0])
        x_data = normalizer.standardize(data=x_data, mu=self.train_dtl.x_mean[self.output_col], std=self.train_dtl.x_std[self.output_col])
        time_idx = np.arange(0, len(x_data))

        # Fit a first-order regression model, ax+b, to the data x_data, y_data
        valid_indices = ~np.isnan(x_data)
        time_idx_filtered = time_idx[valid_indices]
        x_data_filtered = x_data[valid_indices]
        self.speed_init, self.level_init = np.polyfit(time_idx_filtered, x_data_filtered, 1)

        # Plot x_data
        if plot:
            fig = plt.figure(figsize=(10, 2))
            gs = gridspec.GridSpec(1, 1)
            ax0 = plt.subplot(gs[0])
            ax0.plot(time_idx, x_data, label='Synthetic time series', alpha=1)
            # Plot baseline with initial level and speed
            baseline = self.level_init + self.speed_init * time_idx
            ax0.plot(time_idx, baseline, label='Estimated baseline', alpha=1)
            ax0.legend()
            ax0.set_title('Baseline estimation using first-order regression')

    def train(self, plot = False, true_phiAR = 0.5, true_SigmaAR = 0.15):
        # Network
        net = Sequential(
            LSTM(self.num_features, 30, self.input_seq_len),
            LSTM(30, 30, self.input_seq_len),
            Linear(30 * self.input_seq_len, 1),
        )
        net.set_threads(8)
        # #net.to_device("cuda")

        # # # State-space models: for baseline hidden states
        LA_var_stationary = self.Sigma_AA_ratio*1/(1-self.phi_AA**2)
        # # Autoregressive acceleration + online AR
        hybrid = LSTM_SSM(
            neural_network = net,           # LSTM
            baseline = self.components, # 'level', 'trend', 'acceleration', 'ETS'
            zB  = np.array([self.level_init, self.speed_init, 0, 0.5, -0.05]),
            SzB = np.array([1E-16, 1E-10, LA_var_stationary, 0.5**2, 0.15**2]),
            use_auto_AR = self.use_auto_AR,
            mu_W2b_init = 1**2,
            var_W2b_init = 1**2,
        )

        # Training
        mses = []
        pbar = tqdm(range(self.num_epochs), desc="Training Progress")
        for epoch in pbar:
            batch_iter = self.train_dtl.create_data_loader(self.batch_size, shuffle=False)

            # Decaying observation's variance
            # sigma_v = exponential_scheduler(
            #     curr_v=1E-12, min_v=1E-12, decaying_factor=1, curr_iter=epoch
            # )
            var_y = np.full((self.batch_size * len(self.output_col),), self.sigma_v**2, dtype=np.float32)

            # Initialize list to save
            hybrid.init_ssm_hs()
            mu_preds_lstm = []
            var_preds_lstm = []
            mu_preds_norm = []
            var_preds_norm = []
            obs_norm = []
            mu_phiar = []
            var_phiar = []
            mu_aa = []
            var_aa = []
            mu_sigma_ar = []
            var_sigma_ar = []
            mu_ar = []
            var_ar = []

            for x, y in batch_iter:
                mu_x, var_x = process_input_ssm(
                    mu_x = x, mu_preds_lstm = mu_preds_lstm, var_preds_lstm = var_preds_lstm,
                    input_seq_len = self.input_seq_len, num_features = self.num_features,
                    )

                # Feed forward
                y_pred, Sy_pred, z_pred, Sz_pred, m_pred, v_pred = hybrid(mu_x, var_x)
                # Backward
                hybrid.backward(mu_obs = y, var_obs = var_y)

                # Training metric
                pred = normalizer.unstandardize(
                    y_pred.flatten(), self.train_dtl.x_mean[self.output_col], self.train_dtl.x_std[self.output_col]
                )
                obs = normalizer.unstandardize(
                    y, self.train_dtl.x_mean[self.output_col], self.train_dtl.x_std[self.output_col]
                )
                mse = metric.mse(pred, obs)
                mses.append(mse)
                mu_preds_lstm.extend(m_pred)
                var_preds_lstm.extend(v_pred)
                obs_norm.extend(y)
                mu_preds_norm.extend(y_pred[0])
                var_preds_norm.extend(Sy_pred[0] + self.sigma_v**2)
                mu_phiar.append(z_pred[-3].item())
                var_phiar.append(Sz_pred[-3][-3])
                mu_aa.append(z_pred[2].item())
                var_aa.append(Sz_pred[2][2])
                mu_ar.append(z_pred[-2].item())
                var_ar.append(Sz_pred[-2][-2])
                mu_sigma_ar.append(np.sqrt(hybrid.mu_W2b_posterior.item()))
                var_sigma_ar.append(np.sqrt(hybrid.var_W2b_posterior.item()))

            # Smoother
            hybrid.smoother()

            mu_smoothed = np.array(hybrid.mu_smoothed)
            cov_smoothed = np.array(hybrid.cov_smoothed)

            plt.switch_backend('Agg')
            fig = plt.figure(figsize=(10, 7))
            gs = gridspec.GridSpec(3, 1, height_ratios=[2,1,1])
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])
            ax2 = plt.subplot(gs[2])
            ax0.plot(obs_norm, color='r',label=r"obs.")
            ax0.plot(mu_preds_norm,color='b',label=r"pred.")
            ax0.fill_between(np.arange(len(mu_preds_norm)), np.array(mu_preds_norm).flatten() - np.sqrt(var_preds_norm), np.array(mu_preds_norm).flatten() + np.sqrt(var_preds_norm), color='blue', alpha=0.3, label='_nolegend_')
            ax0.plot(mu_smoothed[:,0,:],color='k',label=r"level")
            ax0.fill_between(np.arange(len(mu_smoothed[:,0,:])), np.array(mu_smoothed[:,0,:]).flatten() - np.sqrt(cov_smoothed[:, 0, 0]), np.array(mu_smoothed[:,0,:]).flatten() + np.sqrt(cov_smoothed[:, 0, 0]), color='k', alpha=0.3, label='_nolegend_')
            ax0.plot(mu_smoothed[:,-2,:],color='g',label=r"AR")
            ax0.fill_between(np.arange(len(mu_smoothed[:,-2,:])), np.array(mu_smoothed[:,-2,:]).flatten() - np.sqrt(cov_smoothed[:, -2, -2]), np.array(mu_smoothed[:,-2,:]).flatten() + np.sqrt(cov_smoothed[:, -2, -2]), color='g', alpha=0.3, label='_nolegend_')
            ax0.plot(mu_smoothed[:,-1,:],color='orange',label=r"LSTM")
            ax0.fill_between(np.arange(len(mu_smoothed[:,-1,:])), np.array(mu_smoothed[:,-1,:]).flatten() - np.sqrt(cov_smoothed[:, -1, -1]), np.array(mu_smoothed[:,-1,:]).flatten() + np.sqrt(cov_smoothed[:, -1, -1]), color='orange', alpha=0.3, label='_nolegend_')
            ax0.legend(ncol = 1, loc='upper left', bbox_to_anchor=(1, 1.1), frameon=False)
            ax0.set_ylabel('Norm. obs.')
            ax0.set_xticklabels([])

            ax1.plot(np.arange(len(mu_phiar)),mu_phiar,color='b',label=r"AR")
            ax1.fill_between(np.arange(len(mu_phiar)), np.array(mu_phiar) - np.sqrt(var_phiar), np.array(mu_phiar) + np.sqrt(var_phiar), color='blue', alpha=0.3, label='±1 SD')
            ax1.axhline(y=true_phiAR, color='r', linestyle='--', label='True phi')
            ax1.set_ylim(-0.1, 1.1)
            ax1.set_xticklabels([])
            ax1.set_ylabel('$\phi_{\mathtt{AR}}$')
            ax2.plot(np.arange(len(mu_sigma_ar)),mu_sigma_ar,color='b',label=r"AR")
            ax2.fill_between(np.arange(len(mu_sigma_ar)), np.array(mu_sigma_ar) - np.sqrt(var_sigma_ar), np.array(mu_sigma_ar) + np.sqrt(var_sigma_ar), color='blue', alpha=0.3, label='±1 SD')
            ax2.axhline(y=np.sqrt(true_SigmaAR) / (self.train_dtl.x_std[self.output_col] + 1e-10),
                        color='r', linestyle='--', label='True sigma_AR')
            ax2.set_ylabel('$\sigma_{\mathtt{AR}}$')
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_xlabel('Time step')

            # plt.savefig('hidden_states_epoch0.pdf')
            filename = f'saved_results/debug_0726/hs_epoch_#{epoch}.png'
            plt.savefig(filename)
            plt.close()

            # Progress bar
            pbar.set_description(
                f"Epoch {epoch + 1}/{self.num_epochs}| mse: {np.nanmean(mses):>7.2f}",
                refresh=True,
            )

        # -------------------------------------------------------------------------#
        # Test on validation set
        val_batch_iter = self.val_dtl.create_data_loader(self.batch_size, shuffle=False)

        # Initialize list to save
        mu_preds = []
        var_preds = []
        y_val = []
        obs_val_unnorm = []
        #

        for x, y in val_batch_iter:
            mu_x, var_x = process_input_ssm(
                mu_x = x, mu_preds_lstm = mu_preds_lstm, var_preds_lstm = var_preds_lstm,
                input_seq_len = self.input_seq_len, num_features = self.num_features,
            )
            # Feed forward
            y_pred, Sy_red, z_pred, Sz_pred, m_pred, v_pred = hybrid(mu_x, var_x)
            hybrid.backward(mu_obs = np.nan, var_obs = np.nan, train_LSTM=False)

            mu_preds.extend(y_pred)
            var_preds.extend(Sy_red + self.sigma_v**2)
            y_val.extend(y)
            mu_preds_lstm.extend(m_pred)
            var_preds_lstm.extend(v_pred)

        mu_preds = np.array(mu_preds)
        std_preds = np.array(var_preds) ** 0.5
        y_val = np.array(y_val)
        obs_val_norm = y_val
        mu_preds_norm_val = mu_preds
        std_preds_norm_val = std_preds

        mu_preds = normalizer.unstandardize(
            mu_preds, self.train_dtl.x_mean[self.output_col], self.train_dtl.x_std[self.output_col]
        )
        std_preds = normalizer.unstandardize_std(std_preds, self.train_dtl.x_std[self.output_col])

        y_val = normalizer.unstandardize(
            y_val, self.train_dtl.x_mean[self.output_col], self.train_dtl.x_std[self.output_col]
        )

        # Compute log-likelihood
        mse = metric.mse(mu_preds, y_val)
        log_lik = metric.log_likelihood(
            prediction=mu_preds, observation=y_val, std=std_preds
        )

        #
        obs = np.concatenate((obs_norm,obs_val_norm), axis=0)
        idx_train = range(0,len(obs_norm))

        idx_val = range(len(obs_norm),len(obs))
        idx = np.concatenate((idx_train,idx_val),axis=0)
        mu_preds_norm_val = mu_preds_norm_val.flatten()
        std_preds_norm_val = std_preds_norm_val.flatten()

        print("#############")
        print(f"MSE           : {mse: 0.2f}")
        print(f"Log-likelihood: {log_lik: 0.2f}")

        self.model = hybrid
        self.phi_AR = mu_phiar[-1]
        self.Sigma_AR = mu_sigma_ar[-1]**2
        self.var_phi_AR = var_phiar[-1]
        self.var_Sigma_AR = var_sigma_ar[-1]**2

        if plot:
            fig = plt.figure(figsize=(8, 7))
            gs = gridspec.GridSpec(4, 1)
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])
            ax2 = plt.subplot(gs[2])
            ax3 = plt.subplot(gs[3])
            ax0.plot(np.arange(len(mu_phiar)),mu_phiar,color='b',label=r"AR")
            ax0.fill_between(np.arange(len(mu_phiar)), np.array(mu_phiar) - np.sqrt(var_phiar), np.array(mu_phiar) + np.sqrt(var_phiar), color='blue', alpha=0.3, label='±1 SD')
            ax0.axhline(y=true_phiAR, color='r', linestyle='--', label='True phi')
            ax0.set_ylim(-0.1, 1.1)
            ax0.set_ylabel('phi_AR')
            ax1.plot(np.arange(len(mu_sigma_ar)),mu_sigma_ar,color='b',label=r"AR")
            ax1.fill_between(np.arange(len(mu_sigma_ar)), np.array(mu_sigma_ar) - np.sqrt(var_sigma_ar), np.array(mu_sigma_ar) + np.sqrt(var_sigma_ar), color='blue', alpha=0.3, label='±1 SD')
            ax1.axhline(y=np.sqrt(true_SigmaAR) / (self.train_dtl.x_std[self.output_col] + 1e-10),
                        color='r', linestyle='--', label='True sigma_AR')
            ax1.set_ylabel('sigma_AR')
            ax2.plot(idx,obs, color='r',label=r"data")
            ax2.plot(idx_val, mu_preds_norm_val, color='b',label=r"validation prediction")
            ax2.fill_between(idx_val, mu_preds_norm_val - std_preds_norm_val, mu_preds_norm_val + std_preds_norm_val, color='blue', alpha=0.3, label='±1 SD')
            ax2.plot(idx_train,mu_smoothed[:,0,:],color='k',label=r"level")
            ax2.plot(idx_train, mu_preds_norm,color='g', label=r"train prediction")
            ax2.fill_between(idx_train, mu_preds_norm - np.sqrt(var_preds_norm), mu_preds_norm + np.sqrt(var_preds_norm), color='green', alpha=0.3, label='±1 SD')
            ax3.plot(np.arange(len(mu_ar)),mu_ar,color='b',label=r"AR")
            ax3.fill_between(np.arange(len(mu_ar)), np.array(mu_ar) - np.sqrt(var_ar), np.array(mu_ar) + np.sqrt(var_ar), color='blue', alpha=0.3, label='±1 SD')
            ax3.set_ylabel('AR')
            # ax2.plot(np.arange(len(mu_aa)),mu_aa,color='b',label=r"AA")
            # ax2.fill_between(np.arange(len(mu_aa)), np.array(mu_aa) - np.sqrt(var_aa), np.array(mu_aa) + np.sqrt(var_aa), color='blue', alpha=0.3, label='±1 SD')
            # ax2.set_ylabel('AA')
            # set title
            ax0.set_title('Check if AR and LSTM are correctly trained')

    def save_LSTM_model(self, path):
        self.model.net.save(filename = path)
        self.model_path = path
        self.net_test = self.model.net

    def load_LSTM_model(self, path):
        self.model_path = path
        self.net_test = Sequential(
            LSTM(self.num_features, 30, self.input_seq_len),
            LSTM(30, 30, self.input_seq_len),
            Linear(30 * self.input_seq_len, 1),
        )
        self.net_test.set_threads(8)
        self.net_test.load(filename = self.model_path)

    def get_testing_model_initials(self, val_datetime_values, plot = False):
        # # # # State-space models: for baseline hidden states
        LA_var_stationary = self.Sigma_AA_ratio*self.Sigma_AR/(1-self.phi_AA**2)
        AR_var_stationary = self.Sigma_AR /(1-self.phi_AR**2)
        hybrid_test = LSTM_SSM(
            neural_network = self.net_test,           # LSTM
            baseline = 'AA + AR_fixed', # 'level', 'trend', 'acceleration', 'ETS'
            zB  = np.array([self.level_init, self.speed_init, 0, -0.05]),
            SzB = np.array([1E-6, 1E-6, LA_var_stationary, AR_var_stationary]),
            phi_AR = self.phi_AR,
            Sigma_AR = self.Sigma_AR,
            Sigma_AA_ratio = self.Sigma_AA_ratio,
            phi_AA = self.phi_AA,
            use_auto_AR = False,
        )


        # hybrid_test = LSTM_SSM(
        #     neural_network = self.net_test,           # LSTM
        #     baseline = self.components, # 'level', 'trend', 'acceleration', 'ETS'
        #     zB  = np.array([self.level_init, self.speed_init, 0, 0.5, -0.05]),
        #     SzB = np.array([1E-6, 1E-6, LA_var_stationary, 0.5**2, 0.15**2]),
        #     use_auto_AR = self.use_auto_AR,
        #     mu_W2b_init = 1**2,
        #     var_W2b_init = 1**2,
        # )

        # Run the model on the training set + validation set again without training the LSTM, in ordr to get the initial states for the test set
        var_y = np.full((self.batch_size * len(self.output_col),), self.sigma_v**2, dtype=np.float32)
        obs_norm = []
        obs_unnorm = []
        mu_preds_norm = []
        var_preds_norm = []
        mu_LL = []
        var_LL = []
        mu_AR = []
        var_AR = []
        mu_lstm = []
        var_lstm = []
        mu_AA = []
        var_AA = []

        hybrid_test.init_ssm_hs()
        train_batch_iter = self.train_dtl.create_data_loader(self.batch_size, shuffle=False)
        for x, y in train_batch_iter:
            mu_x, var_x = process_input_ssm(
                mu_x = x, mu_preds_lstm = mu_lstm, var_preds_lstm = var_lstm,
                input_seq_len = self.input_seq_len, num_features = self.num_features,
                )

            # Feed forward
            y_pred, Sy_red, z_pred, Sz_pred, m_pred, v_pred = hybrid_test(mu_x, var_x)
            # Backward
            hybrid_test.backward(mu_obs = y, var_obs = var_y, train_LSTM=False)

            y_unnorm = normalizer.unstandardize(
                    y, self.train_dtl.x_mean[self.output_col], self.train_dtl.x_std[self.output_col]
                )

            obs_norm.extend(y)
            obs_unnorm.extend(y_unnorm)
            mu_preds_norm.extend(y_pred[0])
            var_preds_norm.extend(Sy_red[0] + self.sigma_v**2)
            mu_LL.append(z_pred[0].item())
            var_LL.append(Sz_pred[0][0])
            mu_AR.append(z_pred[-2].item())
            var_AR.append(Sz_pred[-2][-2])
            mu_lstm.extend(m_pred)
            var_lstm.extend(v_pred)
            mu_AA.append(z_pred[2].item())
            var_AA.append(Sz_pred[2][2])

        val_batch_iter = self.val_dtl.create_data_loader(self.batch_size, shuffle=False)
        for x, y in val_batch_iter:
            mu_x, var_x = process_input_ssm(
                mu_x = x, mu_preds_lstm = mu_lstm, var_preds_lstm = var_lstm,
                input_seq_len = self.input_seq_len,num_features = self.num_features,
            )
            # Feed forward
            y_pred, Sy_red, z_pred, Sz_pred, m_pred, v_pred = hybrid_test(mu_x, var_x)
            hybrid_test.backward(mu_obs = y, var_obs = var_y, train_LSTM=False)

            y_unnorm = normalizer.unstandardize(
                    y, self.train_dtl.x_mean[self.output_col], self.train_dtl.x_std[self.output_col]
                )

            obs_norm.extend(y)
            obs_unnorm.extend(y_unnorm)
            mu_preds_norm.extend(y_pred[0])
            var_preds_norm.extend(Sy_red[0] + self.sigma_v**2)
            mu_LL.append(z_pred[0].item())
            var_LL.append(Sz_pred[0][0])
            mu_AR.append(z_pred[-2].item())
            var_AR.append(Sz_pred[-2][-2])
            mu_lstm.extend(m_pred)
            var_lstm.extend(v_pred)
            mu_AA.append(z_pred[2].item())
            var_AA.append(Sz_pred[2][2])

        self.init_mu_lstm = copy.deepcopy(mu_lstm)
        self.init_var_lstm = copy.deepcopy(var_lstm)
        self.init_z = hybrid_test.z
        self.init_Sz = hybrid_test.Sz
        self.last_seq_obs = obs_unnorm[-26:]
        self.last_seq_datetime = val_datetime_values[-26:]
        self.last_lstm_x = copy.deepcopy(x)
        self.model = hybrid_test
        self.init_mu_W2b = hybrid_test.mu_W2b_posterior
        self.init_var_W2b = hybrid_test.var_W2b_posterior

        if plot:
            AR_var_stationary = self.Sigma_AR /(1-self.phi_AR**2)
            fig = plt.figure(figsize=(10, 6))
            gs = gridspec.GridSpec(5, 1)
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])
            ax2 = plt.subplot(gs[2])
            ax3 = plt.subplot(gs[3])
            ax4 = plt.subplot(gs[4])

            ax0.plot(np.arange(len(obs_norm)),obs_norm,color='b',label=r"obs")
            ax0.plot(np.arange(len(mu_preds_norm)),mu_preds_norm,color='r',label=r"pred")
            ax0.fill_between(np.arange(len(mu_preds_norm)), np.array(mu_preds_norm) - np.sqrt(var_preds_norm), np.array(mu_preds_norm) + np.sqrt(var_preds_norm), color='red', alpha=0.3)
            ax0.set_ylabel('obs')
            ax0.legend()

            ax1.plot(np.arange(len(mu_LL)),mu_LL,color='b',label=r"LL")
            ax1.fill_between(np.arange(len(mu_LL)), np.array(mu_LL) - np.sqrt(var_LL), np.array(mu_LL) + np.sqrt(var_LL), color='blue', alpha=0.3, label='±1 SD')
            ax1.set_ylabel('LL')

            ax2.fill_between(np.arange(len(mu_AR)), np.zeros_like(len(mu_AR))-3*np.sqrt(AR_var_stationary), np.zeros_like(len(mu_AR))+3*np.sqrt(AR_var_stationary), color='red', alpha=0.1)
            ax2.plot(np.arange(len(mu_AR)),mu_AR,color='b',label=r"AR")
            ax2.fill_between(np.arange(len(mu_AR)), np.array(mu_AR) - np.sqrt(var_AR), np.array(mu_AR) + np.sqrt(var_AR), color='blue', alpha=0.3, label='±1 SD')
            ax2.set_ylabel('AR')

            ax3.plot(np.arange(len(mu_lstm)),mu_lstm,color='b',label=r"LSTM")
            ax3.fill_between(np.arange(len(mu_lstm)), np.array(mu_lstm) - np.sqrt(var_lstm), np.array(mu_lstm) + np.sqrt(var_lstm), color='blue', alpha=0.3, label='±1 SD')
            ax3.set_ylabel('LSTM')

            ax4.plot(np.arange(len(mu_AA)),mu_AA,color='b',label=r"AA")
            ax4.fill_between(np.arange(len(mu_AA)), np.array(mu_AA) - np.sqrt(var_AA), np.array(mu_AA) + np.sqrt(var_AA), color='blue', alpha=0.3, label='±1 SD')
            ax4.set_ylabel('AA')

            ax0.set_title('Get initialization for the test set')
        return self.net_test, self.init_mu_lstm, self.init_var_lstm, self.init_z, self.init_Sz, self.init_mu_W2b, self.init_var_W2b, self.last_seq_obs, self.last_seq_datetime, self.last_lstm_x

    def check_AA(self, plot = False):
        # # # State-space models: for baseline hidden states
        # LA_var_stationary = self.Sigma_AA_ratio*self.Sigma_AR/(1-self.phi_AA**2)
        hybrid_test = LSTM_SSM(
            neural_network = self.net_test,           # LSTM
            baseline = 'AA + AR_fixed', # 'level', 'trend', 'acceleration', 'ETS'
            z_init = self.init_z,
            Sz_init = self.init_Sz,
            phi_AR = self.phi_AR,
            Sigma_AR = self.Sigma_AR,
            use_auto_AR = False,
            Sigma_AA_ratio = self.Sigma_AA_ratio,
            phi_AA = self.phi_AA,
        )

        # hybrid_test = LSTM_SSM(
        #     neural_network = self.net_test,           # LSTM
        #     baseline = self.components, # 'level', 'trend', 'acceleration', 'ETS'
        #     z_init  = self.init_z,
        #     Sz_init = self.init_Sz,
        #     use_auto_AR = self.use_auto_AR,
        #     mu_W2b_init = self.init_mu_W2b,
        #     var_W2b_init = self.init_var_W2b,
        # )
        self.model = hybrid_test

        batch_iter = self.test_dtl.create_data_loader(self.batch_size, shuffle=False)
        var_y = np.full((self.batch_size * len(self.output_col),), self.sigma_v**2, dtype=np.float32)
        obs_norm = []
        mu_preds_norm = []
        var_preds_norm = []
        mu_LL = []
        var_LL = []
        mu_AR = []
        var_AR = []
        mu_lstm = copy.deepcopy(self.init_mu_lstm)
        var_lstm = copy.deepcopy(self.init_var_lstm)
        mu_AA = []
        var_AA = []

        hybrid_test.init_ssm_hs()
        for x, y in batch_iter:
            mu_x, var_x = process_input_ssm(
                mu_x = x, mu_preds_lstm = mu_lstm, var_preds_lstm = var_lstm,
                input_seq_len = self.input_seq_len, num_features = self.num_features,
                )

            # Feed forward
            y_pred, Sy_red, z_pred, Sz_pred, m_pred, v_pred = hybrid_test(mu_x, var_x)
            # Backward
            hybrid_test.backward(mu_obs = y, var_obs = var_y, train_LSTM=False)

            obs_norm.extend(y)
            mu_preds_norm.extend(y_pred[0])
            var_preds_norm.extend(Sy_red[0] + self.sigma_v**2)
            mu_LL.append(z_pred[0].item())
            var_LL.append(Sz_pred[0][0])
            mu_AA.append(z_pred[2].item())
            var_AA.append(Sz_pred[2][2])
            mu_AR.append(z_pred[-2].item())
            var_AR.append(Sz_pred[-2][-2])
            mu_lstm.extend(m_pred)
            var_lstm.extend(v_pred)
        # Delete the first len(init_mu_lstm) elements in mu_lstm and var_lstm
        mu_lstm = mu_lstm[len(self.init_mu_lstm):]
        var_lstm = var_lstm[len(self.init_var_lstm):]

        AR_var_stationary = self.Sigma_AR /(1-self.phi_AR**2)

        if plot:
            fig = plt.figure(figsize=(10, 8))
            gs = gridspec.GridSpec(5, 1)
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])
            ax2 = plt.subplot(gs[2])
            ax3 = plt.subplot(gs[3])
            ax4 = plt.subplot(gs[4])

            ax0.plot(np.arange(len(obs_norm)),obs_norm,color='b',label=r"obs")
            ax0.plot(np.arange(len(mu_preds_norm)),mu_preds_norm,color='r',label=r"pred")
            ax0.fill_between(np.arange(len(mu_preds_norm)), np.array(mu_preds_norm) - np.sqrt(var_preds_norm), np.array(mu_preds_norm) + np.sqrt(var_preds_norm), color='red', alpha=0.3)
            ax0.set_ylabel('obs')
            ax0.legend()

            ax1.plot(np.arange(len(mu_LL)),mu_LL,color='b',label=r"LL")
            ax1.fill_between(np.arange(len(mu_LL)), np.array(mu_LL) - np.sqrt(var_LL), np.array(mu_LL) + np.sqrt(var_LL), color='blue', alpha=0.3, label='±1 SD')
            ax1.set_ylabel('LL')

            ax2.plot(np.arange(len(mu_lstm)),mu_lstm,color='b',label=r"LSTM")
            ax2.fill_between(np.arange(len(mu_lstm)), np.array(mu_lstm) - np.sqrt(var_lstm), np.array(mu_lstm) + np.sqrt(var_lstm), color='blue', alpha=0.3, label='±1 SD')
            ax2.set_ylabel('LSTM')

            ax3.plot(np.arange(len(mu_AR)),mu_AR,color='b',label=r"AR")
            ax3.fill_between(np.arange(len(mu_AR)), np.array(mu_AR) - np.sqrt(var_AR), np.array(mu_AR) + np.sqrt(var_AR), color='blue', alpha=0.3, label='±1 SD')
            ax3.fill_between(np.arange(len(mu_AR)), np.zeros_like(len(mu_AR))-3*np.sqrt(AR_var_stationary), np.zeros_like(len(mu_AR))+3*np.sqrt(AR_var_stationary), color='red', alpha=0.1)
            ax3.set_ylabel('AR')
            ax3.set_ylim(-1.1, 1.1)

            ax4.plot(np.arange(len(mu_AA)),mu_AA,color='b',label=r"AA")
            ax4.fill_between(np.arange(len(mu_AA)), np.array(mu_AA) - np.sqrt(var_AA), np.array(mu_AA) + np.sqrt(var_AA), color='blue', alpha=0.3, label='±1 SD')
            ax4.set_ylabel('AA')

            ax0.set_title('Check if AA is correctly')


