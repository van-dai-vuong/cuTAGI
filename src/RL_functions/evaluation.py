from src.RL_functions.build_Matrix import *
from src.RL_functions.kalman import *
from src.RL_functions.helpers import *
from src.gym_kalman.env_revised_rewards import *
from itertools import count

class SKF():
    # Run SKF on one time series
    def __init__(self, synthetic_timestamps, synthetic_observations):
        self.synthetic_timestamps = synthetic_timestamps
        self.synthetic_observations = synthetic_observations

    def import_configuration(self, configuration):
        self.LT_sigma_w = configuration["LT_sigma_w"]
        self.LA_sigma_w = configuration["LA_sigma_w"]
        self.KR_p = configuration["KR_p"]
        self.KR_ell = configuration["KR_ell"]
        self.KR_sigma_w = configuration["KR_sigma_w"]
        self.KR_sigma_hw = configuration["KR_sigma_hw"]
        self.AR_phi = configuration["AR_phi"]
        self.AR_sigma_w = configuration["AR_sigma_w"]
        self.sigma_v = configuration["sigma_v"]
        self.X_init = configuration["X_init"]
        self.V_init = configuration["V_init"]
        self.Z_11 = configuration["Z_11"]
        self.Z_22 = configuration["Z_22"]
        self.pi_1 = configuration["pi_1"]
        self.pi_2 = configuration["pi_2"]
        self.isBounded = configuration["isBounded"]
        self.BAR_gamma_value = configuration["BAR_gamma_value"]
        self.sigma_12 = configuration["sigma_12"]

    def get_alarm(self):
        # Set BAR option
        options_BAR = [self.AR_phi, self.AR_sigma_w**2, self.BAR_gamma_value]

        # Get reference time step
        timestep_values, timestep_counts = np.unique(np.diff(self.synthetic_timestamps), return_counts=True)
        timestep_ref = timestep_values[np.argmax(timestep_counts)]

        num_hidden_state = len(self.X_init)

        # Initiation
        ## mu
        mu = np.vstack((self.X_init, self.X_init))
        ## cov
        cov = np.vstack((self.V_init, self.V_init))
        cov = cov.reshape((2, num_hidden_state, num_hidden_state))

        # Array to fill
        ## Probability of non-stationary regime
        prob_ns_regime = np.zeros_like(self.synthetic_timestamps, dtype=float)

        # Probability of being at a regime
        pi = np.array([self.pi_1,self.pi_2])
        Z_12 = 1-self.Z_11
        Z_21 = 1-self.Z_22
        Z = np.array([[self.Z_11, Z_12],[Z_21, self.Z_22]])

        R = np.array([self.sigma_v**2, self.sigma_v**2, self.sigma_v**2, self.sigma_v**2]).reshape((2, 2, 1))
        L_t = np.zeros((2,2), dtype=tuple)

        KR_xcontrol=self.X_init[3:14]

        # Run
        y_pred = np.zeros_like(self.synthetic_observations, dtype=float)
        x_mu = np.zeros((len(self.synthetic_observations), 15))
        x_var = np.zeros((len(self.synthetic_observations), 15))
        for i in range(len(self.synthetic_timestamps)):
            timestep = 1

            ### Model 1
            LTcA_A, LTcA_C, LTcA_Q = LA_design_matrix(self.LT_sigma_w * np.sqrt(timestep/timestep_ref), timestep)
            LTcA_A[:,2] = 0

            ### Model 2
            LA_A, LA_C, LA_Q = LA_design_matrix(self.LA_sigma_w * np.sqrt(timestep/timestep_ref), timestep)

            KR_A, KR_C, KR_Q = KR_design_matrix(self.KR_sigma_hw*np.sqrt(timestep/timestep_ref), self.KR_sigma_w*np.sqrt(timestep/timestep_ref), self.KR_p, self.KR_ell, self.synthetic_timestamps[i], self.synthetic_timestamps[0], len(KR_xcontrol)-1)
            AR_A, AR_C, AR_Q = AR_design_matrix(self.AR_sigma_w * np.sqrt(timestep/timestep_ref), self.AR_phi ** (timestep/timestep_ref))

            # Assemble A, C, Q, R
            A_s = scipy.linalg.block_diag(LTcA_A,KR_A,AR_A)
            Q_s_s = scipy.linalg.block_diag(LTcA_Q,KR_Q,AR_Q)
            Q_s_ns = scipy.linalg.block_diag(LTcA_Q,KR_Q,AR_Q)
            Q_s_ns[2][2] = self.sigma_12

            A_ns = scipy.linalg.block_diag(LA_A,KR_A,AR_A)
            Q_ns_ns = scipy.linalg.block_diag(LA_Q,KR_Q,AR_Q)
            Q_ns_s = scipy.linalg.block_diag(LA_Q,KR_Q,AR_Q)

            C = np.hstack((LTcA_C,KR_C,AR_C))

            A = np.vstack((A_s,A_s,A_ns,A_ns)).reshape((2, 2, num_hidden_state, num_hidden_state))
            Q = np.vstack((Q_s_s,Q_s_ns,Q_ns_s,Q_ns_ns)).reshape((2, 2, num_hidden_state, num_hidden_state))
            C = np.vstack((C,C,C,C)).reshape((2, 2, num_hidden_state))

            mu,cov,pi,L_t,mu_pred,var_pred = merge_SKF(mu, cov, pi, L_t, A, C, Q, R, self.synthetic_observations[i], \
                                                       Z, self.isBounded, options_BAR)

            prob_ns_regime[i] = pi[1]

            y_pred[i] = mu_pred@C[0,0].T
            x_mu[i] = mu_pred
            x_var[i] = var_pred

        return prob_ns_regime, x_mu, x_var, y_pred


def f1t_evaluation(take_action, anomaly_magnitudes, components, hyperparameters, x_init,
                   device, num_steps, time_step,
                   use_real_data = False, real_timestamps = [], real_observations = [],
                   reps = 50, compare_with_skf = False, step_look_back = 64):
    '''
    This function evaluates the performance of the intervention policy using F1t score
    Inputs:
    - take_action: the function that action taken at each time step
    - anomaly_range: the range of the anomaly
    '''
    from tqdm import tqdm
    f1t_all = []
    f1t_skf_all = []
    detection_window_len = int(num_steps/2)
    for i, anomaly in enumerate(tqdm(anomaly_magnitudes)):
        false_positive = 0
        true_positive = 0
        false_negative = 0
        delta_t_all = []
        false_negative_skf = 0
        false_positive_skf = 0
        true_positive_skf = 0

        delta_t_all_skf = []
        for rep in range(reps):
            anomaly_pos = np.random.randint(step_look_back, int(num_steps/2))
            if use_real_data:
                syn_ts = insert_anomalies(timestamps = real_timestamps, observations = real_observations, \
                                        time_step = time_step, anomaly_timesteps = [anomaly_pos], anomaly_LT = [anomaly])
            else:
                syn_ts = generate_time_series_0(components, time_step, hyperparameters, num_steps, x_init, \
                                            anm_timestep = anomaly_pos,\
                                            anm_LT = anomaly)

            if compare_with_skf:
                ################################# Baseline SKF #################################
                SKF_class=SKF(syn_ts['timesteps'], syn_ts['y'])

                configuration_skf_bar = {
                        "LT_sigma_w": 0,
                        "LA_sigma_w": 0,
                        "KR_p": 365.2422,
                        "KR_ell": 0.98461,
                        "KR_sigma_w": 0,
                        "KR_sigma_hw": 0,
                        "AR_phi": 0.91225,
                        "AR_sigma_w": 0.0060635,
                        "sigma_v": 0.001,
                        "X_init": np.array([0.294, 0.00027, 0, 0, -0.035, -0.256, -0.163, 0.0281, -0.0273, 0.0258, 0.344, 0.26, -0.132, -0.0577, -0.0621]),
                        "V_init": np.diag([0.00531, 2.61E-12, 1E-16, 0.0544, 0.0259, 0.0259, 0.0259, 0.026, 0.026, 0.0259, 0.0259, 0.0258, 0.0259, 0.0258, 6.36E-05]),
                        "Z_11": 0.9999999,
                        "Z_22": 0.9999999,
                        "pi_1": 0.999,
                        "pi_2": 0.001,
                        "isBounded": False,
                        "BAR_gamma_value": 0,
                        "sigma_12": 1e-06
                        }

                SKF_class.import_configuration(configuration_skf_bar)

                prob_ns_regime, x_mu, x_var, y_mu = SKF_class.get_alarm()

                # Update metrics
                TP, FP, FN, delta_t = get_TP_FP_FN(prob_ns_regime, anomaly_pos, detection_window_len)
                true_positive_skf += TP
                false_positive_skf += FP
                false_negative_skf += FN
                if delta_t is not None:
                    delta_t_all_skf.append(delta_t)

            ################################# RL-based intervention #################################
            # Set the dataset for the environment
            i_ts_datasets = {
                'measurement': syn_ts['y'],
                'time_step': time_step,
                'timestamps': syn_ts['timesteps'],
                'components': components,
                'hyperparameters': hyperparameters,
                'initial_states': x_init,
            }
            env = KalmanInterventionEnv(render_mode=None, time_series_datasets=i_ts_datasets, step_look_back = step_look_back, hyperparameters=hyperparameters, smoothing_length=0)
            state, info = env.reset(mode = 'test')
            intervention_taken = False
            AR_std_stationary = np.sqrt(hyperparameters['ar']['process_error_var']/(1-hyperparameters['ar']['phi']**2))
            LA_var_stationary = hyperparameters['autoregressive_acceleration']['LA_process_error_var']/(1-hyperparameters['autoregressive_acceleration']['phi']**2)
            if step_look_back == 64:
                seg_len = 8
            for t in count():
                # state = torch.tensor(np.hstack((state['KF_hidden_states'], intervention_taken)),\
                #                     dtype=torch.float32, device=device).unsqueeze(0)
                print_info = False
                if np.isnan(state['KF_hidden_states']).any():
                    print_info = True
                    print('nan is network input')
                state = torch.tensor(state['KF_hidden_states'],\
                                    dtype=torch.float32, device=device).unsqueeze(0)
                # state = normalize_tensor_two_parts(state, 0, np.sqrt(LA_var_stationary),\
                #                                     0, AR_std_stationary, seg_len)
                state = normalize_tensor_two_parts(state, 0, 1e-8,\
                                                    0, AR_std_stationary, seg_len)
                action = take_action(state = state, greedy=True)
                if print_info:
                    print('action is', action)
                state, _, terminated, truncated, _ = env.step(action.item())

                done = terminated or truncated

                if action.item() == 1:
                    if t < anomaly_pos - step_look_back - 1:
                        false_positive += 1
                        done = True
                        delta_t = 0
                        delta_t_all.append(delta_t)
                    elif t >= anomaly_pos - step_look_back - 1 and t <= anomaly_pos + detection_window_len - step_look_back - 1:
                        true_positive += 1
                        # record triggering time
                        done = True
                        delta_t = t + step_look_back + 1 - anomaly_pos
                        delta_t_all.append(delta_t)
                    else:
                        done = True
                        false_negative += 1
                    intervention_taken = True

                if done:
                    if not intervention_taken:
                        false_negative += 1
                    break

        # Compute the f1 score
        if true_positive == 0:
            f1 = 0
        else:
            precision = true_positive/(true_positive+false_positive)
            recall = true_positive/(true_positive+false_negative)
            f1 = 2*precision*recall/(precision+recall)

        # # Compute f1 score for skf
        f1_skf = compute_f1(true_positive_skf, false_positive_skf, false_negative_skf)

        avg_delta_t = np.mean(delta_t_all)
        avg_delta_t_skf = np.mean(delta_t_all_skf)
        # Compute the penalty ratio linearly decay from 1 (avg_delta_t <= 0) to 0 (avg_delta_t = num_steps - anomaly_pos)
        penalty_ratio = 1 - avg_delta_t/detection_window_len
        penalty_ratio_skf = 1 - avg_delta_t_skf/detection_window_len

        f1t = f1*penalty_ratio
        f1t_skf = f1_skf*penalty_ratio_skf

        f1t_all.append(f1t)
        f1t_skf_all.append(f1t_skf)

        print(f'Anomaly magnitude: {anomaly}')
        print('RL:', true_positive, false_positive, false_negative, penalty_ratio)
        print('SKF:', true_positive_skf, false_positive_skf, false_negative_skf, penalty_ratio_skf)
    return anomaly_magnitudes, f1t_all, f1t_skf_all
