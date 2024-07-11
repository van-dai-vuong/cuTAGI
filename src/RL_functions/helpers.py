import numpy as np
import scipy.linalg
from scipy.stats import norm
import torch
import matplotlib.pyplot as plt
import pandas as pd

def build_matrices(components, time_step, hyperparameters, current_time_stamp = 0):
    # Match component name strings to their index in the list
    '''
    Input: list of component names (strings)
    Output: dictionary with component names as keys and their index in the list as values
    '''
    A = np.array([])
    F = np.array([])
    Q = np.array([])
    for i, component in enumerate(components):
        if component == 'trend':
            A = scipy.linalg.block_diag(A, np.array([[1.,time_step],[0.,1.]]))
            F = np.hstack((F, np.array([1.,0.])))
            Q = scipy.linalg.block_diag(Q, hyperparameters['trend']['process_error_var'] * \
                                                  np.array([[time_step**4/4,time_step**3/2], [time_step**3/2,time_step]]))
        if component == 'acceleration':
            A = scipy.linalg.block_diag(A, np.array([[1.,time_step, time_step**2],[0.,1.,time_step],[0.,0.,1.]]))
            F = np.hstack((F, np.array([1.,0.,0.])))
            Q = scipy.linalg.block_diag(Q, hyperparameters['acceleration']['process_error_var'] * \
                                                  np.array([[time_step**4/4,    time_step**3/2, time_step**2/2],
                                                            [time_step**3/2,    time_step**2,   time_step],
                                                            [time_step**2/2,    time_step,      1]]))
        if component == 'autoregressive_acceleration':
            phi = hyperparameters['autoregressive_acceleration']['phi']
            A = scipy.linalg.block_diag(A, np.array([[1.,time_step, phi * time_step**2],[0.,1.,phi * time_step],[0.,0.,phi]]))
            F = np.hstack((F, np.array([1.,0.,0.])))
            Q_AA = hyperparameters['autoregressive_acceleration']['process_error_var'] * \
                                                  np.array([[time_step**4/4,    time_step**3/2, time_step**2/2],
                                                            [time_step**3/2,    time_step**2,   time_step],
                                                            [time_step**2/2,    time_step,      1]])
            Q_AA[2,2] = hyperparameters['autoregressive_acceleration']['LA_process_error_var']
            Q = scipy.linalg.block_diag(Q, Q_AA)
        if component == 'fourrier':
            w = 2*np.pi*time_step/hyperparameters['fourrier']['period']
            A = scipy.linalg.block_diag(A, np.array([[np.cos(w) , np.sin(w)],[-np.sin(w), np.cos(w)]]))
            F = np.hstack((F, np.array([1.,0.])))
            Q = scipy.linalg.block_diag(Q, hyperparameters['fourrier']['process_error_var'] * np.array([[1,0],[0,1]]))
        if component == 'kernel':
            kapa=0
            #Kapa: autocorrelation between kernel pattern values at consecutive time, 0-no, 1-yes

            T_cp=np.linspace(hyperparameters['kernel']['control_time_begin'], hyperparameters['kernel']['control_time_begin'] \
                             + hyperparameters['kernel']['period'], num= hyperparameters['kernel']['n_cp'], endpoint=False)
            T=np.repeat(current_time_stamp, len(T_cp))

            K_raw=np.exp(-2/hyperparameters['kernel']['kernel_length']**2*np.sin((np.pi*(T-T_cp) / hyperparameters['kernel']['period']))**2)
            K_norm=K_raw/(np.sum(K_raw)+10**(-8))

            A = scipy.linalg.block_diag(A,np.vstack((np.hstack(( np.array([kapa]) , K_norm )),\
                                                    np.hstack(( np.zeros((hyperparameters['kernel']['n_cp'],1)) ,np.eye(hyperparameters['kernel']['n_cp']))))))
            F = np.hstack((F, np.concatenate((np.array([1]), np.zeros(hyperparameters['kernel']['n_cp'])),axis=0)))
            Q = scipy.linalg.block_diag(Q, np.vstack((np.hstack((np.array([hyperparameters['kernel']['sigma_KR0']**2]), np.zeros((hyperparameters['kernel']['n_cp'])))),\
                                            np.hstack((np.zeros((hyperparameters['kernel']['n_cp'],1)), np.eye(hyperparameters['kernel']['n_cp'])*hyperparameters['kernel']['sigma_KR1']**2)))))
        if component == 'ar':
            A = scipy.linalg.block_diag(A, np.array(hyperparameters['ar']['phi']))
            F = np.hstack((F, np.array([1.0])))
            Q = scipy.linalg.block_diag(Q, np.array([hyperparameters['ar']['process_error_var']]))

    # Remove the first dummy row caused by block_diag a empty array at the beginning
    A = np.delete(A, 0, axis=0)
    Q = np.delete(Q, 0, axis=0)

    R = np.array([hyperparameters['observation']['error']])

    return A, F, Q, R

def generate_time_series_0(components, time_step, hyperparameters, num_steps, x_init,\
                         anm_timestep, anm_LT):
    num_component = len(x_init['mu'])

    time_series = {'y': [], 'timesteps': []}
    time_series['timesteps'] = [i*time_step for i in range(num_steps)]

    # Decide if 'kernel' is in components
    if 'kernel' in components:
        hyperparameters['kernel']['control_time_begin'] = time_series['timesteps'][0]

    x_ls = x_init['mu']

    for i in range(num_steps):
        A, F, Q, R = build_matrices(components, time_step, hyperparameters, current_time_stamp = i)
        # LT anomaly
        if i == anm_timestep:
            x_ls[1] += anm_LT
        # # LA anomaly
        # if i > anm_timestep:
        #     x_ls[1] += anm_LT
        x = np.dot(A, x_ls)+np.random.multivariate_normal(np.zeros(num_component), Q)
        y = np.dot(F, x)+np.random.normal(0, R)
        time_series['y'].append(float(y))
        x_ls = x

    return time_series

def generate_time_series(**kwargs):
    # Extract parameters
    components = kwargs.get('components', [])
    time_step = kwargs.get('time_step', None)
    hyperparameters = kwargs.get('hyperparameters', {})
    num_steps = kwargs.get('num_steps', None)
    x_init = kwargs.get('x_init', {})
    insert_anomaly = kwargs.get('insert_anomaly', False)
    anomaly_timesteps = kwargs.get('anomaly_timesteps', [])
    anomaly_LT = kwargs.get('anomaly_LT', [])

    num_component = len(x_init['mu'])
    time_series = {'y': [], 'timesteps': []}
    time_series['timesteps'] = [i*time_step for i in range(num_steps)]

    if 'kernel' in components:
        hyperparameters['kernel']['control_time_begin'] = time_series['timesteps'][0]

    x_laststep = x_init['mu']

    j = 0
    for i in range(num_steps):
        A, F, Q, R = build_matrices(components, time_step, hyperparameters, current_time_stamp = i)
        if insert_anomaly and i in anomaly_timesteps:
            x_laststep[1] += anomaly_LT[j]
            j += 1
        x = np.dot(A, x_laststep)+np.random.multivariate_normal(np.zeros(num_component), Q)
        y = np.dot(F, x)+np.random.normal(0, R)
        time_series['y'].append(float(y))
        x_laststep = x

    return time_series

def insert_anomalies(**kwargs):
    # Extract parameters
    timestamps = kwargs.get('timestamps', [])
    time_step = kwargs.get('time_step', None)
    observations = kwargs.get('observations', [])
    anomaly_timesteps = kwargs.get('anomaly_timesteps', [])
    anomaly_LT = kwargs.get('anomaly_LT', [])

    X_init = np.array([0., 0., 0.])
    components = ['acceleration']
    hyperparameters = {'acceleration': {'process_error_var': 0.0},
                       'observation': {'error': 0.0}}
    A, F, Q, R = build_matrices(components, time_step, hyperparameters, current_time_stamp = 0)

    time_series = {'y': [], 'timesteps': []}

    x_laststep = X_init
    anm_baseline = []
    j = 0
    for i in range(len(timestamps)):
        if i in anomaly_timesteps:
            x_laststep[1] += anomaly_LT[j]
            j+=1
        x = np.dot(A, x_laststep) + np.random.multivariate_normal(np.zeros(len(X_init)), Q)
        y = np.dot(F, x) + np.random.normal(0, R)

        anm_baseline.append(float(y))
        x_laststep = x
        time_series['y'].append(float(observations[i]+y))

    time_series['timesteps'] = list(timestamps)
    return time_series


def generate_time_series_2anomalies(components, time_step, hyperparameters, num_steps, x_init,\
                         anm_timestep1, anm_LT1, anm_timestep2, anm_LT2):
    num_component = len(x_init['mu'])

    time_series = {'y': [], 'time': []}
    time_series['time'] = [i*time_step for i in range(num_steps)]

    # Decide if 'kernel' is in components
    if 'kernel' in components:
        hyperparameters['kernel']['control_time_begin'] = time_series['time'][0]

    x_ls = x_init['mu']

    for i in range(num_steps):
        A, F, Q, R = build_matrices(components, time_step, hyperparameters, current_time_stamp = i)
        # LT anomaly
        if i == anm_timestep1:
            x_ls[1] += anm_LT1

        if i == anm_timestep2:
            x_ls[1] += anm_LT2


        x = np.dot(A, x_ls)+np.random.multivariate_normal(np.zeros(num_component), Q)
        y = np.dot(F, x)+np.random.normal(0, R)
        time_series['y'].append(float(y))
        x_ls = x

    return time_series


def coefficient_times_gaussian(coefficient, gaussian):
    '''
    Input:
    coefficient (array): N*N array
    gaussian (dictionary): keys are mu and var, mu is N*1 array, var is N*N array
    Output: result (array)
    '''
    result = {'mu': np.zeros_like(gaussian['mu']), 'var': np.zeros_like(gaussian['var'])}
    result['mu'] = coefficient @ gaussian['mu']
    result['var'] = coefficient @ gaussian['var'] @ coefficient.T
    return result
# # Test
# coefficient_dummy = np.array([[2, 1], [0, 1]])
# gaussian_dummy = {'mu': np.array([1, 2]), 'var': np.array([[1, 0], [0, 1]])}
# result_dummy = coefficient_times_gaussian(coefficient_dummy, gaussian_dummy)

def gaussian_plus_gaussian(gaussian1, gaussian2):
    '''
    Input:
    gaussian1 (dictionary): keys are mu and var, mu is N*1 array, var is N*N array
    gaussian2 (dictionary): keys are mu and var, mu is N*1 array, var is N*N array
    Output: result (dictionary)
    '''
    result = {'mu': np.zeros_like(gaussian1['mu']), 'var': np.zeros_like(gaussian1['var'])}
    result['mu'] = gaussian1['mu'] + gaussian2['mu']
    result['var'] = gaussian1['var'] + gaussian2['var']
    return result

def normalize_tensor_two_parts(tensor, mu1, std1, mu2, std2, segment_len):
    first_seg = tensor[:, :segment_len]
    last_seg = tensor[:, segment_len:]

    norm_first_seg = (first_seg - mu1) / std1
    norm_last_seg = (last_seg - mu2) / std2

    norm_tensor = torch.cat((norm_first_seg, norm_last_seg), dim=1)
    return norm_tensor

def normalize_array(arr, mu, std):
    return (arr - mu) / std

def get_TP_FP_FN(prob_ns_regime, anomaly_pos, detection_window_len):
    intervention_index = np.where(prob_ns_regime > 0.5)[0]
    if len(intervention_index) == 0:
        false_negative_skf = 1
        false_positive_skf = 0
        true_positive_skf = 0
        delta_t = None
    else:
        intervention_index = intervention_index[0]
        if intervention_index < anomaly_pos:
            false_positive_skf = 1
            true_positive_skf = 0
            false_negative_skf = 0
            delta_t = 0
        elif intervention_index >= anomaly_pos and intervention_index <= anomaly_pos + detection_window_len:
            true_positive_skf = 1
            false_positive_skf = 0
            false_negative_skf = 0
            delta_t = intervention_index - anomaly_pos
        else:
            false_negative_skf = 1
            false_positive_skf = 0
            true_positive_skf = 0
            delta_t = None

    return true_positive_skf, false_positive_skf, false_negative_skf, delta_t


def compute_f1(true_positive, false_positive, false_negative):
    if true_positive == 0:
        f1_skf = 0
    else:
        precision_skf = true_positive/(true_positive+false_positive)
        recall_skf = true_positive/(true_positive+false_negative)
        f1_skf = 2*precision_skf*recall_skf/(precision_skf+recall_skf)

    return f1_skf


def evaluate_standard_gaussian_probability(x, mu, std):
    # Calculate the z-score (standardized value)
    z = (x - mu) / std

    # Evaluate the probability in the standard normal distribution
    probability = norm.pdf(z)

    return probability

def kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    """
    Compute KL divergence between two Gaussian distributions.

    mu_p: mean of the first Gaussian distribution
    sigma_p: standard deviation of the first Gaussian distribution
    mu_q: mean of the second Gaussian distribution
    sigma_q: standard deviation of the second Gaussian distribution
    """
    kl_div = np.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2) - 0.5
    return kl_div

def summarize_time_series(real_timestamps, real_observations, resample_timestamps, resample_observations):
    """
    Summarize real time series data in terms of: time step inteval, number of observation, missing data, plotting, etc.
    """
    timestep_values, timestep_counts=np.unique(np.diff(real_timestamps), return_counts=True)
    timestep = timestep_values[np.argmax(timestep_counts)]

    # Count the number of missing data in the observation
    num_missing_data = np.isnan(real_observations).sum()
    num_missing_data_resampled = np.isnan(resample_observations).sum()

    # Count the number of total observation
    num_total_observations = len(real_timestamps)
    num_total_observations_resampled = len(resample_timestamps)

    # Print info
    print('================== Summary of imported time series data ==================')
    print('Most frequent time step interval:            ', timestep)
    print('Number of total real observations:           ', num_total_observations)
    print('Number of missing data in real observation:  ', num_missing_data)
    print('Number of total resampled observations:      ', num_total_observations_resampled)
    print('Number of missing data in real observation:  ', num_missing_data_resampled)
    print('Plot:')

    plt.figure(figsize=(10,4))
    # Find the index of missing data in observation
    idxnan_real = np.isnan(real_observations)
    idxnan_resample = np.isnan(resample_observations)
    plt.plot(real_timestamps[~idxnan_real],real_observations[~idxnan_real], label='Real', alpha=0.5)
    plt.scatter(real_timestamps, real_observations, 2, alpha=0.5)
    plt.plot(resample_timestamps[~idxnan_resample],resample_observations[~idxnan_resample], label='Resample', alpha=0.5)
    plt.scatter(resample_timestamps, resample_observations, 2, alpha=0.5)
    plt.legend()
    plt.show()

class DataProcessor():
    def __init__(self):
        pass

    def convert_matlabDateNum_to_datetime(self, matlabDateNum):
        """
        Convert a numpy array including Matlab date number to a numpy array including datetime64
        Input: numpy array including Matlab date number
        Output: numpy array including datetime64
        """
        from datetime import datetime, timedelta
        return np.array([np.datetime64(datetime.fromordinal(int(t)) + timedelta(days=t%1) - timedelta(days = 366)) for t in matlabDateNum])

    def convert_datetime_to_matlabDateNum(self, numpy_datetime):
        """
        Convert a numpy array including datetime64 to a numpy array including Matlab date number
        Input: numpy array including datetime64
        Output: numpy array including Matlab date number
        """
        from datetime import datetime, timedelta
        return np.array([(t - np.datetime64('0000-01-01')) / np.timedelta64(1, 'D') + 1 for t in numpy_datetime])

    def resample_time_series(self, matlab_datenum, observations, frequency):
        """
        Resample time series data to a specific frequency
        Input:
        matlab_datenum: numpy array including Matlab date number
        observation: numpy array including observation data
        frequency: resampling frequency
        Output: resampled_matlab_datenum, resampled_observation
        """
        print(type(matlab_datenum[0]))
        numpy_datetime = self.convert_matlabDateNum_to_datetime(matlab_datenum)

        df = pd.DataFrame({'y': observations}, index=numpy_datetime)

        # Resample at different frequencies (e.g., daily, weekly)
        df_freq = df.resample(frequency).interpolate(method='linear')
        resampled_datetime = df_freq.index
        # Convert resampled_datetime to numpy datetime64
        resampled_datetime = np.array([np.datetime64(t) for t in resampled_datetime])
        # Convert resample datetime to matlab datenum
        resampled_datenum = np.array([(t - np.datetime64('0000-01-01')) / np.timedelta64(1, 'D') + 1 for t in resampled_datetime])

        df_nan = df.resample(frequency).bfill(limit=1)
        y_ = df_nan['y'].values
        idxnan = np.isnan(y_)

        # Resample yraw corresponding to the traw at the resampled_datenum
        y = np.interp(resampled_datenum, matlab_datenum, observations)
        y[idxnan] = np.nan

        return resampled_datenum, y
