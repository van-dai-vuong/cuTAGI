import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import time
import random

from src.RL_functions.kalman_filter import KalmanFilter

import gymnasium as gym
from gymnasium import spaces

from scipy.stats import norm

from src.RL_functions.helpers import *


class KalmanInterventionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, time_series_datasets=np.zeros(100), step_look_back = 8, hyperparameters = {}, smoothing_length = 0, LA_process_error_var = 1e-18):
        self.time_series_datasets = time_series_datasets
        self.dim_hidden_state = len(time_series_datasets['initial_states']['mu'])
        self.step_look_back = step_look_back
        self.hyperparameters = hyperparameters
        self.smoothing_length = smoothing_length
        self.LA_process_error_var = LA_process_error_var

        # Observations are dictionaries with the hidden states values
        self.observation_space = spaces.Dict(
            {
                "KF_hidden_states": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=float),  # two dinemsional values in real space
                "measurement": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=float),  # target value in time series at time t
            }
        )

        # We have 2 actions, corresponding to "remains", "intervene"
        self.action_space = spaces.Discrete(2)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the intervention we will take in if that action is taken.
        I.e. 0 corresponds to "remains", 1 to "intervene".
        """
        intervention_var = np.zeros_like(time_series_datasets['initial_states']['var'])
        intervention_var[1][1] = 1e2
        self._action_to_intervention = {
            0: np.zeros_like(intervention_var),
            1: intervention_var,
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


    def _get_obs(self):
        return {"KF_hidden_states": self._agent_vision, "measurement": self._measurement}

    def _get_info(self):
        return {
            "Current time step": self.current_step+1,
            "MAE": np.abs(self.y_pred['mu'] - self._measurement).mean(),
            "y_pred": self.y_pred_track,
            'hidden_states': self.hidden_state_one_episode,
            'smoothed_initial_states': self.time_series_datasets['initial_states']
        }

    def reset(self, seed=None, mode='train'):
        super().reset(seed=seed)

        # Set up the Kalman filter
        self.kalman_filter = KalmanFilter(self.time_series_datasets['components'], \
                                     self.time_series_datasets['time_step'],
                                     self.time_series_datasets['hyperparameters'])
        if 'kernel' in self.time_series_datasets['components']:
            self.hyperparameters['kernel']['control_time_begin'] = self.time_series_datasets['timestamps'][0]

        ############# Run Kalman filter until the end of the smoothing initialization #############
        if self.smoothing_length == 0:
            pass
        else:
            x_pred_ks = {'mu': np.zeros((self.smoothing_length, self.dim_hidden_state)), \
                        'var': np.zeros((self.smoothing_length, self.dim_hidden_state, self.dim_hidden_state))}
            x_upt_ks = {'mu': np.zeros((self.smoothing_length, self.dim_hidden_state)), \
                        'var': np.zeros((self.smoothing_length, self.dim_hidden_state, self.dim_hidden_state))}
            A_ks = np.zeros((self.smoothing_length, self.dim_hidden_state, self.dim_hidden_state))

            x_last_step =  self.time_series_datasets['initial_states']
            for i in range(self.smoothing_length):
                # Predict
                _, xi_pred, A_i = self.kalman_filter.predict(x_last_step, current_time_stamp = self.time_series_datasets['timestamps'][i], LA_process_error_var = self.LA_process_error_var)
                # Update
                if np.isnan(self.time_series_datasets['measurement'][i]).any():
                    xi_update = xi_pred
                else:
                    xi_update, _, _ = self.kalman_filter.update(self.time_series_datasets['measurement'][i])

                x_last_step = xi_update

                x_pred_ks['mu'][i, :] = xi_pred['mu']
                x_pred_ks['var'][i, :, :] = xi_pred['var']
                x_upt_ks['mu'][i, :] = xi_update['mu']
                x_upt_ks['var'][i, :, :] = xi_update['var']
                A_ks[i, :, :] = A_i

            # Smooth the hidden states from self.smoothing_length to 0
            smoothed_x = {'mu': x_upt_ks['mu'][-1, :], 'var': x_upt_ks['var'][-1, :, :]}
            for i in range(self.smoothing_length - 2, -1, -1):
                x_upt_current = {'mu': x_upt_ks['mu'][i, :], 'var': x_upt_ks['var'][i, :, :]}
                x_pred_next = {'mu': x_pred_ks['mu'][i+1, :], 'var': x_pred_ks['var'][i+1, :, :]}
                A_next = A_ks[i+1, :, :]
                smoothed_x = self.kalman_filter.smooth(x_upt_current, x_pred_next, smoothed_x, A_next)

            x_pred_next = {'mu': x_pred_ks['mu'][0, :], 'var': x_pred_ks['var'][0, :, :]}
            smoothed_x_init = self.kalman_filter.smooth(self.time_series_datasets['initial_states'], x_pred_next, smoothed_x, A_ks[0, :, :])
            # Set the entries outside diagonal to 0
            smoothed_x_init['var'] = np.diag(np.diag(smoothed_x_init['var']))
            smoothed_x_init['mu'][2] = 0

            self.time_series_datasets['initial_states'] = smoothed_x_init

        ############# Run Kalman filter using the new initialization until we have enouth steps to look back #############
        if mode == 'train':
            sample_init = random.random()
            if sample_init < 0.1:
                x_init = {'mu': self.time_series_datasets['initial_states']['mu'], \
                            'var': self.time_series_datasets['initial_states']['var']}
            else:
                x_init = {'mu': self.time_series_datasets['initial_states']['mu'], \
                          'var': self.time_series_datasets['initial_states']['var_stabilized']}
        elif mode == 'get_stabilized_var':
            x_init = {'mu': self.time_series_datasets['initial_states']['mu'], \
                    'var': self.time_series_datasets['initial_states']['var']}
        else:
            x_init = {'mu': self.time_series_datasets['initial_states']['mu'], \
                    'var': self.time_series_datasets['initial_states']['var_stabilized']}

        x_last_step =  x_init
        self.hidden_state_one_episode = {'mu': np.zeros((len(self.time_series_datasets['measurement']), self.dim_hidden_state)), \
                                         'var': np.zeros((len(self.time_series_datasets['measurement']), self.dim_hidden_state, self.dim_hidden_state))}
        self.y_pred_track = {'mu': np.zeros(len(self.time_series_datasets['measurement'])), 'var': np.zeros(len(self.time_series_datasets['measurement']))}
        for i in range(self.step_look_back + 1):
            # Predict
            self.y_pred, x_pred, _ = self.kalman_filter.predict(x_last_step, current_time_stamp = self.time_series_datasets['timestamps'][i], LA_process_error_var = self.LA_process_error_var)

            # Update
            if np.isnan(self.time_series_datasets['measurement'][i]).any():
                x_update = x_pred
            else:
                x_update, _, _ = self.kalman_filter.update(self.time_series_datasets['measurement'][i])

            x_last_step = x_update
            # Save the hidden state estimation
            self.hidden_state_one_episode['mu'][i, :] = x_update['mu']
            self.hidden_state_one_episode['var'][i, :, :] = x_update['var']
            self.y_pred_track['mu'][i] = self.y_pred['mu']
            self.y_pred_track['var'][i] = self.y_pred['var']

        self.x_last_step = x_last_step
        self.current_step = self.step_look_back # Current time step is the python index

        # Collect hidden states and define agent's vision for the initial RL step (not the initial KF step)
        hidden_states_temp = self._hidden_states_collector(self.current_step, self.hidden_state_one_episode)
        # self._agent_vision = hidden_states_temp['mu'][:, -1].T.reshape(-1)
        self._agent_vision = np.hstack((hidden_states_temp['mu'][:, 2], hidden_states_temp['mu'][:, -1]))
        self._measurement = self.time_series_datasets['measurement'][self.current_step]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1}) to the intervention
        intervention = self._action_to_intervention[action]

        # Execute action and get the new hidden state
        if action == 1:
            # self.x_last_step['var'] = self.time_series_datasets['initial_states']['var']
            self.x_last_step['var'][2, :] = self.time_series_datasets['initial_states']['var'][2,:]
            self.x_last_step['var'][:, 2] = self.time_series_datasets['initial_states']['var'][:,2]
            self.x_last_step['mu'][2] = self.time_series_datasets['initial_states']['mu'][2]
        self.x_last_step['var'] += intervention

        # Run Kalman filter
        self.y_pred, x_pred, _ = self.kalman_filter.predict(self.x_last_step, current_time_stamp = self.time_series_datasets['timestamps'][self.current_step], LA_process_error_var = self.LA_process_error_var)
        if np.isnan(self.time_series_datasets['measurement'][self.current_step + 1]).any():
            x_update = x_pred
            likelihood = norm.pdf(self.y_pred['mu'], loc=self.y_pred['mu'], scale=np.sqrt(self.y_pred['var']))
        else:
            x_update, likelihood, mse = self.kalman_filter.update(self.time_series_datasets['measurement'][self.current_step + 1])
        self.x_last_step = x_update
        # Save the hidden state estimation
        self.hidden_state_one_episode['mu'][self.current_step + 1, :] = x_update['mu']
        self.hidden_state_one_episode['var'][self.current_step + 1, :, :] = x_update['var']
        self.y_pred_track['mu'][self.current_step + 1] = self.y_pred['mu']
        self.y_pred_track['var'][self.current_step + 1] = self.y_pred['var']

        # Collect hidden states and define agent's vision for the next time step
        hidden_states_temp = self._hidden_states_collector(self.current_step + 1, self.hidden_state_one_episode)
        self._agent_vision = np.hstack((hidden_states_temp['mu'][:, 2], hidden_states_temp['mu'][:, -1]))
        self._measurement = self.time_series_datasets['measurement'][self.current_step + 1]

        # An episode is done if the agent has reached the end of the time series
        terminated = self.current_step + 1 == len(self.time_series_datasets['measurement']) - 1 # Next time step is the last one

        # Reward is the likelihood of the next measurement given the prediction made using the current hidden states
        AR_var_stationary = self.hyperparameters['ar']['process_error_var']/(1-self.hyperparameters['ar']['phi']**2)
        clip_value_ar = np.log(evaluate_standard_gaussian_probability(x = 3*np.sqrt(x_update['var'][-1, -1]+AR_var_stationary), \
                                                               mu = 0, std = np.sqrt(x_update['var'][-1, -1]+AR_var_stationary)))
        clip_value_la = np.log(evaluate_standard_gaussian_probability(x = 1*np.sqrt(x_update['var'][2, 2]+self.time_series_datasets['initial_states']['var'][2,2]), \
                                                               mu = 0, std = np.sqrt(x_update['var'][2, 2]+self.time_series_datasets['initial_states']['var'][2,2])))

        reward = float(
                # likelihood
                # np.log(likelihood)
                np.clip(np.log(likelihood),-1e3, np.inf)
                + np.clip(np.log(evaluate_standard_gaussian_probability(x_update['mu'][-1], 0, np.sqrt(x_update['var'][-1, -1]+AR_var_stationary))),\
                            -1e3, clip_value_ar) - clip_value_ar\
                + np.clip(np.log(evaluate_standard_gaussian_probability(x_update['mu'][2], 0, np.sqrt(x_update['var'][2, 2]+self.hidden_state_one_episode['var'][0,2,2]))),\
                            -1e3, clip_value_la) - clip_value_la\
                )

        observation = self._get_obs()
        info = self._get_info()

        self.current_step += 1

        return observation, reward, terminated, False, info

    def _get_look_back_time_steps(self, current_step):
        look_back_step_list = [0]

        # Start with 1 since the next value after 0 is 1
        current = 1

        # Keep doubling until the current value exceeds n
        while current <= self.step_look_back:
            look_back_step_list.append(current)
            current *= 2  # Double the current value

        look_back_step_list = [current_step - i for i in look_back_step_list]

        # # keep the first and the last element of look_back_step_list
        # look_back_step_list = [look_back_step_list[0], look_back_step_list[-1]]

        return look_back_step_list


    def _hidden_states_collector(self, current_step, hidden_states_all_step):
        look_back_steps_list = self._get_look_back_time_steps(current_step)
        hidden_states_collected = {'mu': hidden_states_all_step['mu'][look_back_steps_list, :], \
                                    'var': hidden_states_all_step['var'][look_back_steps_list, :, :]}
        return hidden_states_collected

    def render(self):
        pass

    def close(self):
        pass