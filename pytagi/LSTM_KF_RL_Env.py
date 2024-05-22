import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pytagi.hybrid import process_input_ssm

class LSTM_KF_Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, data_loader = None, train_x_mean = 0, train_x_std = 1,
                 ts_model = None, step_look_back = 8):
        self.data_loader = data_loader
        self.train_x_mean = train_x_mean
        self.train_x_std = train_x_std
        self.ts_model = ts_model
        self.step_look_back = step_look_back
        # Observations are dictionaries with the hidden states values
        self.observation_space = spaces.Dict(
            {
                "hidden_states": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=float),  # two dinemsional values in real space
                "measurement": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=float),  # target value in time series at time t
            }
        )

        # We have 2 actions, corresponding to "remains", "intervene"
        self.action_space = spaces.Discrete(2)
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.batch_iter = data_loader.create_data_loader(batch_size=1, shuffle=False)
        self.time_series_len = self._count_time_series_len(data_loader.create_data_loader(batch_size=1, shuffle=False))

    def _get_obs(self):
        return {"hidden_states": self._agent_vision, "measurement": self._measurement}

    def _get_info(self):
        return {
            'hidden_state_one_episode': self.hidden_state_one_episode,
            'measurement_one_episode': self.obs_unnorm,
            'prediction_one_episode': self.prediction_one_episode,
        }

    def _count_time_series_len(self, dataloader):
        return sum(1 for _ in dataloader)

    def _get_look_back_time_steps(self, current_step):
        look_back_step_list = [0]
        current = 1
        while current <= self.step_look_back:
            look_back_step_list.append(current)
            current *= 2
        look_back_step_list = [current_step - i for i in look_back_step_list]

        return look_back_step_list

    def _hidden_states_collector(self, current_step, hidden_states_all_step):
        hidden_states_all_step_numpy = np.copy(hidden_states_all_step)
        hidden_states_all_step_numpy = {'mu': np.array(hidden_states_all_step['mu']), \
                                  'var': np.array(hidden_states_all_step['var'])}
        look_back_steps_list = self._get_look_back_time_steps(current_step)
        hidden_states_collected = {'mu': hidden_states_all_step_numpy['mu'][look_back_steps_list, :], \
                                    'var': hidden_states_all_step_numpy['var'][look_back_steps_list, :, :]}
        return hidden_states_collected

    def reset(self, seed=None):
        super().reset(seed=seed)

        sigma_v = 1E-12
        self.var_y = np.full((len(self.data_loader.output_col),), sigma_v**2, dtype=np.float32)

        self.mu_preds_lstm = []
        self.var_preds_lstm = []
        self.obs_unnorm = []
        self.ts_model.init_ssm_hs()

        self.hidden_state_one_episode = {'mu': [], \
                                         'var': []}
        self.prediction_one_episode = {'mu': [], \
                                        'var': []}

        for i, (x, y) in enumerate(self.batch_iter):
            mu_x, var_x = process_input_ssm(
                mu_x = x, mu_preds_lstm = self.mu_preds_lstm, var_preds_lstm = self.var_preds_lstm,
                input_seq_len = self.data_loader.input_seq_len, num_features = self.data_loader.num_features,
                )

            # Feed forward
            y_pred, Sy_pred, z_pred, Sz_pred, m_pred, v_pred = self.ts_model(mu_x, var_x)
            # Backward
            z_updata, Sz_update = self.ts_model.backward(mu_obs = y, var_obs = self.var_y, train_LSTM = False)

            self.hidden_state_one_episode['mu'].append(z_updata.flatten().tolist())
            self.hidden_state_one_episode['var'].append(Sz_update.tolist())
            self.prediction_one_episode['mu'].append(y_pred.tolist())
            self.prediction_one_episode['var'].append(Sy_pred.tolist())

            self.mu_preds_lstm.extend(m_pred)
            self.var_preds_lstm.extend(v_pred)
            self.obs_unnorm.extend(y)
            if i == self.step_look_back:
                break

        self.current_step = self.step_look_back # Current time step is the python index
        hidden_states_temp = self._hidden_states_collector(self.current_step, self.hidden_state_one_episode)

        self._agent_vision = np.hstack((hidden_states_temp['mu'][:, 2], hidden_states_temp['mu'][:, -2]))
        self._measurement = self.obs_unnorm[-1]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Action
        # TODO

        # Run Kalman filter
        self.current_step += 1
        terminated = False

        for i, (x, y) in enumerate(self.batch_iter, start=self.current_step):
            if i == self.current_step:
                mu_x, var_x = process_input_ssm(
                    mu_x = x, mu_preds_lstm = self.mu_preds_lstm, var_preds_lstm = self.var_preds_lstm,
                    input_seq_len = self.data_loader.input_seq_len, num_features = self.data_loader.num_features,
                    )
                # Feed forward
                y_pred, Sy_pred, z_pred, Sz_pred, m_pred, v_pred = self.ts_model(mu_x, var_x)
                # Backward
                z_updata, Sz_update = self.ts_model.backward(mu_obs = y, var_obs = self.var_y, train_LSTM = False)

                self.hidden_state_one_episode['mu'].append(z_updata.flatten().tolist())
                self.hidden_state_one_episode['var'].append(Sz_update.tolist())
                self.prediction_one_episode['mu'].append(y_pred.tolist())
                self.prediction_one_episode['var'].append(Sy_pred.tolist())

                self.mu_preds_lstm.extend(m_pred)
                self.var_preds_lstm.extend(v_pred)
                self.obs_unnorm.extend(y)
                break

        hidden_states_temp = self._hidden_states_collector(self.current_step, self.hidden_state_one_episode)

        self._agent_vision = np.hstack((hidden_states_temp['mu'][:, 2], hidden_states_temp['mu'][:, -2]))
        self._measurement = self.obs_unnorm[-1]

        observation = self._get_obs()
        info = self._get_info()

        # Reward
        # TODO
        reward = 0

        if i == self.time_series_len - 1:
            terminated = True

        return observation, reward, terminated, False, info


