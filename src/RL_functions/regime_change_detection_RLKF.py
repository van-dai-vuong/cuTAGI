from pytagi.hybrid import LSTM_SSM
from pytagi import Normalizer
from tqdm import tqdm
import copy
import numpy as np
from pytagi.hybrid import process_input_ssm
import matplotlib.pyplot as plt
from src.RL_functions.BDLM_trainer import BDLM_trainer
import csv
import pandas as pd
# from src.RL_functions.dqn_lstm_agent import *

import random
import math
from src.RL_functions.helpers import *
from pytagi.LSTM_KF_RL_Env import LSTM_KF_Env
from examples.data_loader import TimeSeriesDataloader, SyntheticTimeSeriesDataloader
from itertools import count
import matplotlib
from matplotlib import gridspec
from collections import namedtuple, deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class regime_change_detection_RLKF():
    def __init__(self, **kwargs):
        self.trained_BDLM = kwargs.get('trained_BDLM', None)
        self.val_datetime_values = kwargs.get('val_datetime_values', None)

        self.n_observations = kwargs.get('n_observations', None)
        self.n_actions = kwargs.get('n_actions', None)
        self.phi_AA = self.trained_BDLM.phi_AA
        self.Sigma_AA_ratio = self.trained_BDLM.Sigma_AA_ratio
        self.device = kwargs.get('device', 'cpu')

        self.LSTM_test_net = self.trained_BDLM.net_test
        self.init_mu_lstm = self.trained_BDLM.init_mu_lstm
        self.init_var_lstm = self.trained_BDLM.init_var_lstm
        self.init_z = self.trained_BDLM.init_z
        self.init_Sz = self.trained_BDLM.init_Sz
        # self.init_mu_W2b = self.trained_BDLM.init_mu_W2b
        # self.init_var_W2b = self.trained_BDLM.init_var_W2b
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

        self.memory = ReplayMemory(1000)
        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.optimal_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimal_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=0.01, amsgrad=True)
        self.steps_done = 0
        self.episode_rewards = []
        self.episode_f1t = []
        self.GAMMA = 0.999

    def generate_synthetic_ts(self, num_syn_ts, syn_ts_len, plot = True):
        self.syn_ts_all = []
        for j in tqdm(range(num_syn_ts)):
            # hybrid_gen = LSTM_SSM(
            #             neural_network = self.LSTM_test_net,           # LSTM
            #             baseline = 'AA + AR',
            #             z_init  = self.init_z,
            #             Sz_init = self.init_Sz,
            #             use_auto_AR = True,
            #             mu_W2b_init = self.init_mu_W2b,
            #             var_W2b_init = self.init_var_W2b,
            #             Sigma_AA_ratio = self.Sigma_AA_ratio,
            #             phi_AA = self.phi_AA,
            #         )
            hybrid_gen = LSTM_SSM(
                        neural_network = self.LSTM_test_net,           # LSTM
                        baseline = 'AA + AR_fixed',
                        z_init  = self.init_z,
                        Sz_init = self.init_Sz,
                        use_auto_AR = False,
                        # Sample phi_AR following a Gaussian with mean self.phi_AR and variance self.trained_BDLM.var_phi_AR
                        # phi_AR = np.random.normal(self.phi_AR, np.sqrt(self.trained_BDLM.var_phi_AR)),
                        # Sigma_AR = np.random.normal(self.Sigma_AR, np.sqrt(self.trained_BDLM.var_Sigma_AR)),
                        phi_AR = self.phi_AR,
                        Sigma_AR = self.Sigma_AR,
                        Sigma_AA_ratio = self.Sigma_AA_ratio,
                        phi_AA = self.phi_AA,
                    )

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
                Q_gen = copy.deepcopy(hybrid_gen.Q)
                # Q_gen[-2, -2] = hybrid_gen.mu_W2b_posterior
                # Q_gen[2, 2] = hybrid_gen.Sigma_AR * hybrid_gen.Sigma_AA_ratio
                Q_gen[-1, -1] = v_pred
                z_sample = z_prior.flatten() + np.random.multivariate_normal(0*z_prior.flatten(), Q_gen)
                y_sample = np.dot(hybrid_gen.F, z_sample)
                hybrid_gen.z = z_sample.reshape(-1, 1)

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

    def load_synthetic_ts(self, datetime_save_path=None, observation_save_path=None):
        self.datetime_values = self._load_data_from_csv(datetime_save_path)
        self.syn_ts_all = self._load_data_from_csv(observation_save_path).T
        self.datetime_save_path = datetime_save_path
        self.observation_save_path = observation_save_path

    def plot_synthetic_ts(self):
        COLORS = self._get_cmap(10)
        plt.figure(figsize=(20, 9))
        for i in range(10):
            plt.plot(self.syn_ts_all[i], color = COLORS(i))
        plt.show()

    def train(self, num_episodes, step_look_back, abnormal_ts_percentage, anomaly_range,
              init_z, init_Sz, init_mu_preds_lstm, init_var_preds_lstm,
              batchsize, TAU, plot_samples=False, learning_curve_ylim = None,
              early_stopping = False, patience = 10, validation_episode_num = 0, early_stop_start = 0,
              agent_net_save_path='./saved_param/CASC_LGA007PIAP_E010_2024_07/agent_net.pth', cost_of_intervention=None):
        num_steps_per_episode = len(self.syn_ts_all[0])
        track_intervention_taken_times = np.zeros(num_episodes-validation_episode_num)
        optim_F1t = -1E8
        optim_episode = 0
        # Set seed for numpy random
        np.random.seed(0)
        validation_rand_samples = [np.random.random() for _ in range(validation_episode_num)]
        anm_positions_val = np.random.randint(step_look_back + self.trained_BDLM.input_seq_len, num_steps_per_episode, validation_episode_num)
        anm_magnitudes_val = np.random.uniform(anomaly_range[0], anomaly_range[1], validation_episode_num)
        print(validation_rand_samples)
        print(anm_positions_val)
        print(anm_magnitudes_val)

        # Estimatethe cost of intervention
        if cost_of_intervention is None:
            print('Estimating the cost of intervention...')
            self.cost_intervention = self._estimate_intervention_cost(num_episodes, validation_episode_num, init_z, init_Sz, init_mu_preds_lstm, init_var_preds_lstm)
            print('The cost of intervention is: ', self.cost_intervention)
            print('=====================================')
        else:
            self.cost_intervention = cost_of_intervention

        for i_episode in range(num_episodes-validation_episode_num):
            anm_pos = np.random.randint(step_look_back + self.trained_BDLM.input_seq_len, num_steps_per_episode)

            sample = random.random()
            if sample < abnormal_ts_percentage:
                train_dtl = SyntheticTimeSeriesDataloader(
                    x_file=self.observation_save_path,
                    select_column=i_episode,
                    date_time_file=self.datetime_save_path,
                    add_anomaly = True,
                    anomaly_magnitude=np.random.uniform(anomaly_range[0], anomaly_range[1]),
                    anomaly_start=anm_pos,
                    x_mean=self.trained_BDLM.train_dtl.x_mean,
                    x_std=self.trained_BDLM.train_dtl.x_std,
                    output_col=self.trained_BDLM.output_col,
                    input_seq_len=self.trained_BDLM.input_seq_len,
                    output_seq_len=self.trained_BDLM.output_seq_len,
                    num_features=self.trained_BDLM.num_features,
                    stride=self.trained_BDLM.seq_stride,
                    time_covariates=self.trained_BDLM.time_covariates,
                )
                anomaly_injected = True
            else:
                train_dtl = SyntheticTimeSeriesDataloader(
                    x_file=self.observation_save_path,
                    select_column = i_episode,
                    date_time_file=self.datetime_save_path,
                    x_mean=self.trained_BDLM.train_dtl.x_mean,
                    x_std=self.trained_BDLM.train_dtl.x_std,
                    output_col=self.trained_BDLM.output_col,
                    input_seq_len=self.trained_BDLM.input_seq_len,
                    output_seq_len=self.trained_BDLM.output_seq_len,
                    num_features=self.trained_BDLM.num_features,
                    stride=self.trained_BDLM.seq_stride,
                    time_covariates=self.trained_BDLM.time_covariates,
                )
                anomaly_injected = False

            # train_dtl = self.trained_BDLM.test_dtl
            # anomaly_injected = False

            env = LSTM_KF_Env(render_mode=None, data_loader=train_dtl, step_look_back=step_look_back)

            state, info = env.reset(z=init_z, Sz=init_Sz, mu_preds_lstm = copy.deepcopy(init_mu_preds_lstm), var_preds_lstm = copy.deepcopy(init_var_preds_lstm),
                        net_test = self.LSTM_test_net, init_mu_W2b = None, init_var_W2b = None, phi_AR = self.phi_AR, Sigma_AR = self.Sigma_AR,
                        phi_AA = self.phi_AA, Sigma_AA_ratio = self.Sigma_AA_ratio)
            state = state['hidden_states']
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Nomalize the state
            # Compute the stationary standard deviation for AR
            AR_std_stationary = np.sqrt(self.trained_BDLM.Sigma_AR/(1 - self.trained_BDLM.phi_AR**2))
            LA_var_stationary = self.trained_BDLM.Sigma_AA_ratio *  self.trained_BDLM.Sigma_AR/(1 - self.trained_BDLM.phi_AA**2)
            if step_look_back == 64:
                seg_len = 8
            state = normalize_tensor_two_parts(state, 0, np.sqrt(LA_var_stationary), \
                                            0, AR_std_stationary, seg_len)

            total_reward_one_episode = 0
            dummy_steps = 0
            Q_values_all = []
            for t in count():
                action = self._select_action(state)

                observation, reward, terminated, truncated, info = env.step(action.item(), cost_intervention=self.cost_intervention)

                Q_values_t = self._track_Qvalues(state)[0].tolist()
                Q_values_all.append(Q_values_t)

                dummy_steps += 1

                if action.item() == 1:
                    track_intervention_taken_times[i_episode] += 1
                    intervention_taken = True

                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated
                total_reward_one_episode += reward

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation['hidden_states'],\
                            dtype=torch.float32, device=self.device).unsqueeze(0)
                    next_state = normalize_tensor_two_parts(next_state, 0, np.sqrt(LA_var_stationary),\
                                                            0, AR_std_stationary, seg_len)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self._optimize_model(batchsize)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_rewards.append(total_reward_one_episode)
                    if early_stopping is not True:
                        self._plot_rewards(metric = self.episode_rewards, ylim=learning_curve_ylim)
                    else:
                        self._plot_rewards(metric = self.episode_f1t, ylim=[-0.1, 1.1])
                    break

            # Fill 65 rows of [nan, nan] values in front of Q_values_all
            Q_values_all = [[np.nan, np.nan]] * 65 + Q_values_all
            Q_values_all = np.array(Q_values_all).T

            print(track_intervention_taken_times)
            print(self.episode_f1t)

            if plot_samples:
                timesteps = np.arange(0, len(info['measurement_one_episode']), 1)
                mu_hidden_states_one_episode = np.array(info['hidden_state_one_episode']['mu'])
                var_hidden_states_one_episode = np.array(info['hidden_state_one_episode']['var'])
                mu_prediction_one_episode = np.array(info['prediction_one_episode']['mu']).flatten()
                var_prediction_one_episode = np.array(info['prediction_one_episode']['var']).flatten()
                # if track_intervention_taken_times[i_episode] == 1:
                if True:
                    # Plot prediction
                    fig = plt.figure(figsize=(15, 12))
                    gs = gridspec.GridSpec(5, 1)
                    ax0 = plt.subplot(gs[0])
                    ax1 = plt.subplot(gs[2])
                    ax2 = plt.subplot(gs[3])
                    ax3 = plt.subplot(gs[4])
                    ax4 = plt.subplot(gs[1])

                    ax0.plot(timesteps, info['measurement_one_episode'], label='True')
                    # plot the standard deviation of the prediction
                    ax0.plot(timesteps, mu_prediction_one_episode, label='Predicted')
                    ax0.fill_between(timesteps, mu_prediction_one_episode - np.sqrt(var_prediction_one_episode),\
                                        mu_prediction_one_episode + np.sqrt(var_prediction_one_episode), color='gray', alpha=0.2)
                    ax0.set_title(f'Predicted vs True, epoch {i_episode+1}')
                    ax0.set_ylabel('y')
                    delta_max_min = np.max(mu_prediction_one_episode) - np.min(mu_prediction_one_episode)
                    ax0.set_ylim(np.min(mu_prediction_one_episode)-0.05*delta_max_min, np.max(mu_prediction_one_episode)+0.05*delta_max_min)
                    if anomaly_injected:
                        anomaly_pos = timesteps[anm_pos-step_look_back]
                        ax0.axvline(x=anomaly_pos, color='gray', linestyle='--')
                    ax0.legend()

                    ax1.plot(timesteps, mu_hidden_states_one_episode[:,2], label='LA')
                    ax1.fill_between(timesteps, mu_hidden_states_one_episode[:,2] - np.sqrt(var_hidden_states_one_episode[:,2,2]),\
                                        mu_hidden_states_one_episode[:,2] + np.sqrt(var_hidden_states_one_episode[:,2,2]), color='gray', alpha=0.2)
                    ax1.set_ylabel('LA')

                    ax2.plot(timesteps, mu_hidden_states_one_episode[:,-1], label='PD')
                    ax2.fill_between(timesteps, mu_hidden_states_one_episode[:,-1] - np.sqrt(var_hidden_states_one_episode[:,-1,-1]),\
                                        mu_hidden_states_one_episode[:,-1] + np.sqrt(var_hidden_states_one_episode[:,-1,-1]), color='gray', alpha=0.2)
                    ax2.set_ylabel('PD')

                    ax3.fill_between(timesteps, np.zeros_like(timesteps)-3*AR_std_stationary, np.zeros_like(timesteps)+3*AR_std_stationary, color='red', alpha=0.1)
                    ax3.plot(timesteps, mu_hidden_states_one_episode[:,-2], label='AR')
                    ax3.fill_between(timesteps, mu_hidden_states_one_episode[:,-2] - np.sqrt(var_hidden_states_one_episode[:,-2,-2]),\
                                        mu_hidden_states_one_episode[:,-2] + np.sqrt(var_hidden_states_one_episode[:,-2,-2]), color='gray', alpha=0.2)
                    ax3.set_ylabel('AR')

                    ax4.plot(Q_values_all[0], label='Q_value_0')
                    ax4.plot(Q_values_all[1], label='Q_value_1')
                    ax4.set_ylabel('Q_values')
                    ax4.set_xlim(ax0.get_xlim())
                    ax4.legend(loc='upper left')
                    plt.show()
                    # filename = f'saved_results/Qvalues_training/Qvalues_episode#{i_episode}.png'
                    # plt.savefig(filename)
                    # plt.close()


            # Early stopping
            if early_stopping:
                if i_episode >= early_stop_start:
                    FP = 0
                    FN = 0
                    TP = 0
                    lambdas_all = []
                    for i_val_episode in range(validation_episode_num):
                        anm_pos = anm_positions_val[i_val_episode]

                        sample = validation_rand_samples[i_val_episode]
                        if sample < abnormal_ts_percentage:
                            train_dtl = SyntheticTimeSeriesDataloader(
                                x_file=self.observation_save_path,
                                select_column = num_episodes-validation_episode_num+i_val_episode,
                                date_time_file=self.datetime_save_path,
                                add_anomaly = True,
                                anomaly_magnitude = anm_magnitudes_val[i_val_episode],
                                anomaly_start=anm_pos,
                                x_mean=self.trained_BDLM.train_dtl.x_mean,
                                x_std=self.trained_BDLM.train_dtl.x_std,
                                output_col=self.trained_BDLM.output_col,
                                input_seq_len=self.trained_BDLM.input_seq_len,
                                output_seq_len=self.trained_BDLM.output_seq_len,
                                num_features=self.trained_BDLM.num_features,
                                stride=self.trained_BDLM.seq_stride,
                                time_covariates=self.trained_BDLM.time_covariates,
                            )
                            anomaly_injected = True
                        else:
                            train_dtl = SyntheticTimeSeriesDataloader(
                                x_file=self.observation_save_path,
                                select_column = num_episodes-validation_episode_num+i_val_episode,
                                date_time_file=self.datetime_save_path,
                                x_mean=self.trained_BDLM.train_dtl.x_mean,
                                x_std=self.trained_BDLM.train_dtl.x_std,
                                output_col=self.trained_BDLM.output_col,
                                input_seq_len=self.trained_BDLM.input_seq_len,
                                output_seq_len=self.trained_BDLM.output_seq_len,
                                num_features=self.trained_BDLM.num_features,
                                stride=self.trained_BDLM.seq_stride,
                                time_covariates=self.trained_BDLM.time_covariates,
                            )
                            anomaly_injected = False

                        env = LSTM_KF_Env(render_mode=None, data_loader=train_dtl, step_look_back=step_look_back)
                        state, info = env.reset(z=init_z, Sz=init_Sz, mu_preds_lstm = copy.deepcopy(init_mu_preds_lstm), var_preds_lstm = copy.deepcopy(init_var_preds_lstm),
                                    net_test = self.LSTM_test_net, init_mu_W2b = None, init_var_W2b = None, phi_AR = self.phi_AR, Sigma_AR = self.Sigma_AR,
                                    phi_AA = self.phi_AA, Sigma_AA_ratio = self.Sigma_AA_ratio)
                        state = state['hidden_states']
                        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                        # Nomalize the state
                        AR_std_stationary = np.sqrt(self.trained_BDLM.Sigma_AR/(1 - self.trained_BDLM.phi_AR**2))
                        LA_var_stationary = self.trained_BDLM.Sigma_AA_ratio *  self.trained_BDLM.Sigma_AR/(1 - self.trained_BDLM.phi_AA**2)
                        state = normalize_tensor_two_parts(state, 0, np.sqrt(LA_var_stationary), 0, AR_std_stationary, seg_len)

                        # Run agent on the i_val_episode time series
                        val_ep_steps = 0
                        for t in count():
                            action = self._select_action(state, greedy=True)
                            observation, _, terminated, truncated, info = env.step(action.item())

                            done = terminated or truncated

                            if action.item() == 1:
                                if anomaly_injected:
                                    # Alarm is triggered
                                    trigger_pos = val_ep_steps + step_look_back + self.trained_BDLM.input_seq_len
                                    if trigger_pos >= anm_pos:
                                        # True alarm
                                        TP += 1
                                        lambda_i = 1 - (trigger_pos - anm_pos) / (num_steps_per_episode - anm_pos)
                                    else:
                                        # False alarm
                                        FP += 1
                                        lambda_i = 1
                                    lambdas_all.append(lambda_i)
                                else:
                                    FP += 1
                                done = True

                            if terminated:
                                # When comes to here, alarm is never triggered and the time series ends
                                if anomaly_injected:
                                    FN += 1
                                    lambda_i = 0
                                    lambdas_all.append(lambda_i)
                                next_state = None
                            else:
                                next_state = torch.tensor(observation['hidden_states'], dtype=torch.float32, device=self.device).unsqueeze(0)
                                next_state = normalize_tensor_two_parts(next_state, 0, np.sqrt(LA_var_stationary), 0, AR_std_stationary, seg_len)
                            state = next_state
                            val_ep_steps += 1

                            if done:
                                print('For validation episode:', i_val_episode)
                                print('Anomaly is injected:', anomaly_injected)
                                print(f'TP = {TP}, FP = {FP}, FN = {FN}, lambda = {lambdas_all}')
                                break

                    current_F1t = np.mean(lambdas_all)*2*TP/(2*TP+FP+FN)
                    self.episode_f1t.append(current_F1t)

                    if current_F1t > optim_F1t:
                        print('------------------------------------')
                        print(f'Optimal F1t: {optim_F1t}, episode: {optim_episode}')
                        print(f'Current F1t: {current_F1t}, episode: {i_episode}')
                        print('****** Update the optimal model ******')
                        print('------------------------------------')
                        optim_F1t = current_F1t
                        optim_episode = i_episode
                        self.optimal_net.load_state_dict(self.policy_net.state_dict())
                    else:
                        print('------------------------------------')
                        print(f'Optimal F1t: {optim_F1t}, episode: {optim_episode}')
                        print(f'Current F1t: {current_F1t}, episode: {i_episode}')
                        print('Optimal model remains')
                        print('------------------------------------')
                        if i_episode - optim_episode > patience:
                            print('Early stopping is triggered, training stopped, model saved at epoch:', optim_episode)
                            self.policy_net.load_state_dict(self.optimal_net.state_dict())
                            # Save policy net
                            torch.save(self.policy_net.state_dict(), agent_net_save_path)
                            break
                    if i_episode == num_episodes-validation_episode_num-1:
                        print('Finished training in all episodes.')
                        # Save policy net
                        torch.save(self.policy_net.state_dict(), agent_net_save_path)


        print('Complete')
        if early_stopping is not True:
            self._plot_rewards(metric = self.episode_rewards, show_result=True, ylim=learning_curve_ylim)
        else:
            self._plot_rewards(metric = self.episode_f1t, show_result=True, ylim=[-0.1, 1.1])
        # plot a bar chart of the number of interventions taken
        plt.figure(2)
        plt.bar(np.arange(num_episodes-validation_episode_num)+1, track_intervention_taken_times)
        plt.title('Number of interventions taken')
        plt.xlabel('Episode')
        plt.ylabel('Number of interventions taken')
        plt.show()
        plt.ioff()

    def _estimate_intervention_cost(self, num_episodes, validation_episode_num, init_z, init_Sz, init_mu_preds_lstm, init_var_preds_lstm):
        Q_estimates = []
        for i_episode in tqdm(range(num_episodes-validation_episode_num)):
            from itertools import count

            train_dtl = SyntheticTimeSeriesDataloader(
                        x_file=self.observation_save_path,
                        select_column=i_episode,
                        date_time_file=self.datetime_save_path,
                        add_anomaly = False,
                        x_mean=self.trained_BDLM.train_dtl.x_mean,
                        x_std=self.trained_BDLM.train_dtl.x_std,
                        output_col=self.trained_BDLM.output_col,
                        input_seq_len=self.trained_BDLM.input_seq_len,
                        output_seq_len=self.trained_BDLM.output_seq_len,
                        num_features=self.trained_BDLM.num_features,
                        stride=self.trained_BDLM.seq_stride,
                        time_covariates=self.trained_BDLM.time_covariates,
                    )
            step_look_back = 64
            env = LSTM_KF_Env(render_mode=None, data_loader=train_dtl, step_look_back=step_look_back)
            env.reset(z=init_z, Sz=init_Sz, mu_preds_lstm = copy.deepcopy(init_mu_preds_lstm), var_preds_lstm = copy.deepcopy(init_var_preds_lstm),
                        net_test = self.LSTM_test_net, init_mu_W2b = None, init_var_W2b = None, phi_AR = self.phi_AR, Sigma_AR = self.Sigma_AR,
                        phi_AA = self.phi_AA, Sigma_AA_ratio = self.Sigma_AA_ratio)
            Q_estimate = 0
            for t in count():
                _, reward, terminated, truncated, _ = env.step(0)
                Q_estimate += self.GAMMA**(t) * reward
                done = terminated or truncated
                if done:
                    break

            Q_estimates.append(Q_estimate)

        # Compute the mean and std of Q_estimates
        std_Q = np.std(np.array(Q_estimates))

        return std_Q

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

    def _load_data_from_csv(self, data_file: str, skip_row = 0, select_column = None) -> pd.DataFrame:
        """Load data from csv file"""

        data = pd.read_csv(data_file, skiprows=skip_row, delimiter=",", header=None)

        # if select_column is not given by the user, reshape the data to (-1, 1)
        if select_column is None:
            return data.values
        else:
            # if select_column is given by the user, return the selected column
            return data.values[:, select_column].reshape(-1, 1)

    def _optimize_model(self, BATCH_SIZE):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def _plot_rewards(self, metric, show_result=False, ylim=None):
        plt.figure(1)
        durations_t = torch.tensor(metric, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.plot(durations_t.numpy())
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])

        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def _select_action(self, state, greedy=False, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=1000):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if not greedy:
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return self.policy_net(state).max(1).indices.view(1, 1)
            else:
                sample2 = random.random()
                if sample2 < 1/100:
                    # select action 1
                    return torch.tensor([[1]], device=self.device, dtype=torch.long)
                else:
                    return torch.tensor([[0]], device=self.device, dtype=torch.long)
                # # # random action
                # return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)

    def _track_Qvalues(self, state):
        with torch.no_grad():
            Q_values = self.policy_net(state)
            return Q_values