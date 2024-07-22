import numpy as np
import scipy.linalg
from datetime import timedelta
import csv

class generate_one_synthetic_time_series:
    def __init__(self, **kwargs):
        self.components = kwargs.get('components', [])
        self.time_step_interval = kwargs.get('time_step_interval', None)
        self.hyperparameters = kwargs.get('hyperparameters', {})
        self.num_steps = kwargs.get('num_steps', None)
        self.x_init = kwargs.get('x_init', {})
        self.insert_anomaly = kwargs.get('insert_anomaly', False)
        self.anomaly_timesteps = kwargs.get('anomaly_timesteps', [])
        self.anomaly_LT = kwargs.get('anomaly_LT', [])

        self.num_component = len(self.x_init['mu'])
        self.time_series = {'y': [], 'timesteps': []}
        self.time_series['timesteps'] = [i*self.time_step_interval for i in range(self.num_steps)]

        if 'kernel' in self.components:
            self.hyperparameters['kernel']['control_time_begin'] = self.time_series['timesteps'][0]

        self.x_laststep = self.x_init['mu']

        self.generate_time_series()

    def generate_time_series(self):
        j = 0
        for i in range(self.num_steps):
            A, F, Q, R = self._build_matrices(self.components, self.time_step_interval, self.hyperparameters, current_time_stamp = i)
            if self.insert_anomaly and i in self.anomaly_timesteps:
                self.x_laststep[1] += self.anomaly_LT[j]
                j += 1
            x = np.dot(A, self.x_laststep)+np.random.multivariate_normal(np.zeros(self.num_component), Q)
            y = np.dot(F, x)+np.random.normal(0, R)
            self.time_series['y'].append(float(y))
            self.x_laststep = x

        return self.time_series

    def plot(self):
        from matplotlib import gridspec
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 2))
        gs = gridspec.GridSpec(1, 1)
        ax0 = plt.subplot(gs[0])

        ax0.plot(self.time_series['timesteps'], self.time_series['y'], label='Synthetic time series', alpha=1)
        ax0.set_xlabel('Day')
        ax0.legend()

    def export_to_csv(self, indices_train_val_test, observation_file_paths, datetime_file_paths, start_datetime):
        datetime_values = self._generate_datetime_values(start_datetime, len(self.time_series['timesteps']), self.time_step_interval)

        for k in range(3):
            if k == 0:
                obs_tosave = self.time_series['y'][0: indices_train_val_test[k]]
                datetime_tosave = datetime_values[0: indices_train_val_test[k]]
            else:
                obs_tosave = self.time_series['y'][indices_train_val_test[k-1]-26: indices_train_val_test[k]]
                datetime_tosave = datetime_values[indices_train_val_test[k-1]-26: indices_train_val_test[k]]
                if k == 1:
                    self.val_datetime_values = datetime_tosave

            with open(observation_file_paths[k], 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['syn_obs'])
                for i in range(len(datetime_tosave)):
                    writer.writerow([round(float(obs_tosave[i]), 3)])

            with open(datetime_file_paths[k], 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['datetime'])
                for i in range(len(datetime_tosave)):
                    writer.writerow([datetime_tosave[i]])

    def get_validation_datetime_values(self):
        return self.val_datetime_values

    def _generate_datetime_values(self, start_datetime, num_values, time_step_interval):
        datetime_values = []
        current_datetime = start_datetime
        for _ in range(num_values):
            datetime_values.append(current_datetime.strftime('%Y-%m-%d %I:%M:%S'))
            current_datetime += timedelta(days=time_step_interval)
        return datetime_values

    def _build_matrices(self, components, time_step, hyperparameters, current_time_stamp = 0):
        # Match component name strings to their index in the list
        '''
        Input: list of component names (strings)
        Output: dictionary with component names as keys and their index in the list as values
        '''
        A = np.array([])
        F = np.array([])
        Q = np.array([])
        for i, component in enumerate(components):
            if component == 'level':
                A = scipy.linalg.block_diag(A, np.array([[1.]]))
                F = np.hstack((F, np.array([1.])))
                Q = scipy.linalg.block_diag(Q, hyperparameters['level']['process_error_var'] * \
                                                    np.array([[time_step]]))
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
