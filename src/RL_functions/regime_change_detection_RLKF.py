from pytagi.hybrid import LSTM_SSM
from pytagi import Normalizer
from tqdm import tqdm
import copy
import numpy as np
from pytagi.hybrid import process_input_ssm
import matplotlib.pyplot as plt
from src.RL_functions.BDLM_trainer import BDLM_trainer
import csv

class regime_change_detection_RLKF():
    def __init__(self, **kwargs):
        self.trained_BDLM = kwargs.get('trained_BDLM', None)
        self.val_datetime_values = kwargs.get('val_datetime_values', None)

        self.LSTM_test_net = self.trained_BDLM.net_test
        self.init_mu_lstm = self.trained_BDLM.init_mu_lstm
        self.init_var_lstm = self.trained_BDLM.init_var_lstm
        self.init_z = self.trained_BDLM.init_z
        self.init_Sz = self.trained_BDLM.init_Sz
        self.last_seq_obs = self.trained_BDLM.last_seq_obs
        self.last_lstm_x = self.trained_BDLM.last_lstm_x

        self.train_xmean = self.trained_BDLM.train_dtl.x_mean
        self.train_xstd = self.trained_BDLM.train_dtl.x_std
        self.time_covariates = self.trained_BDLM.time_covariates
        self.input_seq_len = self.trained_BDLM.input_seq_len
        self.num_features = self.trained_BDLM.num_features
        self.output_col = self.trained_BDLM.output_col
        self.phi_AR = self.trained_BDLM.phi_AR
        self.Sigma_AR = self.trained_BDLM.Sigma_AR

        self.Sigma_AA_ratio = kwargs.get('Sigma_AA_ratio', 1e-14)
        self.phi_AA = kwargs.get('phi_AA', 0.99)

    def generate_synthetic_ts(self, num_syn_ts, syn_ts_len, plot = True):
        hybrid_gen=LSTM_SSM(
            neural_network = self.LSTM_test_net,           # LSTM
            baseline = 'AA + AR_fixed', # 'level', 'trend', 'acceleration', 'ETS'
            z_init = self.init_z,
            Sz_init = self.init_Sz,
            phi_AR = self.phi_AR,
            Sigma_AR = self.Sigma_AR,
        )

        self.syn_ts_all = []
        for j in tqdm(range(num_syn_ts)):
            syn_ts_i=copy.deepcopy(self.last_seq_obs)
            gen_datetime = np.array(self.val_datetime_values, dtype='datetime64')
            self.datetime_values_tosave = []
            for i in range(25):
                self.datetime_values_tosave.append(gen_datetime[-(25-i)])
            current_date_time = gen_datetime[-1] + np.timedelta64(7, 'D')
            self.datetime_values_tosave.append(current_date_time)

            x = copy.deepcopy(self.last_lstm_x)
            gen_mu_lstm = copy.deepcopy(self.init_mu_lstm)
            gen_var_lstm = copy.deepcopy(self.init_var_lstm)

            hybrid_gen.init_ssm_hs()
            hybrid_gen.z = copy.deepcopy(self.init_z)
            hybrid_gen.Sz = copy.deepcopy(self.init_Sz)

            for i in range(syn_ts_len):
                # remove the first two elements in x, and add two new at the end
                gen_datetime = np.append(gen_datetime, [gen_datetime[-1] + np.timedelta64(7, 'D')]).reshape(-1, 1)
                next_date = self._normalize_date(gen_datetime[-1], self.train_xmean[-1], self.train_xstd[-1], self.time_covariates)
                x[0:-2] = x[2:]
                x[-2] = gen_mu_lstm[-1].item()
                x[-1] = next_date.item()

                x_input = np.copy(x)
                mu_x_, var_x_ = process_input_ssm(
                        mu_x = x_input, mu_preds_lstm = gen_mu_lstm, var_preds_lstm = gen_var_lstm,
                        input_seq_len = self.input_seq_len, num_features = self.num_features,
                        )
                # Feed forward
                y_pred, Sy_red, z_prior, Sz_prior, m_pred, v_pred = hybrid_gen(mu_x_, var_x_)
                hybrid_gen.backward(mu_obs = np.nan, var_obs = np.nan, train_LSTM=False)

                # Sample
                z_sample = np.random.multivariate_normal(z_prior.flatten(), Sz_prior)
                y_sample = np.dot(hybrid_gen.F, z_sample)

                obs_sample = Normalizer.unstandardize(
                    y_sample, self.train_xmean[self.output_col], self.train_xstd[self.output_col]
                )

                gen_mu_lstm.extend(m_pred)
                gen_var_lstm.extend(v_pred)
                syn_ts_i.extend(obs_sample)
                current_date_time = gen_datetime[-1] + np.timedelta64(7, 'D')
                self.datetime_values_tosave.append(current_date_time[0])

            self.syn_ts_all.append(syn_ts_i)

        if plot:
            COLORS = self._get_cmap(10)
            plt.figure(figsize=(20, 9))
            for i in range(10):
                plt.plot(self.syn_ts_all[i], color = COLORS(i))
            plt.show()

    def save_synthetic_ts(self, datetime_save_path=None, observation_save_path=None):
        with open(datetime_save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['date_time'])  # Write header
            for dt in self.datetime_values_tosave:
                writer.writerow([dt])  # Write formatted datetime string

        transposed_data = list(zip(*self.syn_ts_all))

        with open(observation_save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(transposed_data)


    def _normalize_date(self, date_time_i, mean, std, time_covariates):
        for time_cov in time_covariates:
            if time_cov == 'hour_of_day':
                hour_of_day = date_time_i.astype('datetime64[h]').astype(int) % 24
                output = hour_of_day
            elif time_cov == 'day_of_week':
                day_of_week = date_time_i.astype('datetime64[D]').astype(int) % 7
                output = day_of_week
            elif time_cov == 'week_of_year':
                week_of_year = date_time_i.astype('datetime64[W]').astype(int) % 52 + 1
                output = week_of_year
            elif time_cov == 'month_of_year':
                month_of_year = date_time_i.astype('datetime64[M]').astype(int) % 12 + 1
                output = month_of_year
            elif time_cov == 'quarter_of_year':
                month_of_year = date_time_i.astype('datetime64[M]').astype(int) % 12 + 1
                quarter_of_year = (month_of_year - 1) // 3 + 1
                output = quarter_of_year
            elif time_cov == 'day_of_year':
                day_of_year = date_time_i.astype('datetime64[D]').astype(int) % 365
                output = day_of_year

        output = Normalizer.standardize(data=output, mu=mean, std=std)
        return output

    def _get_cmap(self, n, name='rainbow'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)
