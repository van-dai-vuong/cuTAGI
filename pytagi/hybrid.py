import numpy as np
from typing import Optional
from numpy.linalg import inv
from numpy.linalg import pinv

class SSM:
    """State-space models for modeling baselines:
    Define model matrices [A,Q,F] and initial hidden states zB and SzB

    Attributes:
        Define
    """

    def __init__(
        self,
        baseline: str,
        zB: Optional[np.ndarray] = None,
        SzB: Optional[np.ndarray] = None,
    ) -> None:
        self.baseline = baseline
        self.zB = zB
        self.SzB = SzB
        self.define_matrices()

    def define_matrices(self):
        if self.baseline == 'level':
            self.A = np.diag([1])
            self.Q = np.zeros((1,1))
            self.F = np.array([1,1]).reshape(1, -1)
        elif self.baseline == 'trend':
            self.A = np.array([[1,1],[0,1]])
            self.Q = np.zeros((2,2))
            self.F = np.array([1,0,1]).reshape(1, -1)
        elif self.baseline == 'acceleration':
            self.A = np.array([[1,1,0.5],[0,1,1],[0,0,1]])
            self.Q = np.zeros((3,3))
            self.F = np.array([1,0,0,1]).reshape(1, -1)

    def filter(
        self, mu_lstm: np.ndarray, var_lstm: np.ndarray,
        var_obs: Optional[np.ndarray] = None,
        mu_obs: Optional[np.ndarray] = None
    ):
        # Perform Kalman filter
        # Prediction step for baseline hs
        zB_prior  = self.A @ self.zB
        SzB_prior = self.A @ self.SzB @ self.A.T + self.Q
        nb_baseline_hs = len(zB_prior)

        # Perform Kalman filter update step
        # Assemble hidden states: baseline + lstm
        z_prior  = np.concatenate((zB_prior,np.array([mu_lstm]).reshape(-1,1)),axis=0)
        Sz_prior = np.concatenate((SzB_prior,np.zeros((nb_baseline_hs,1))),axis=1)
        Sz_prior = np.concatenate((Sz_prior,np.zeros((1,nb_baseline_hs+1))),axis=0)
        Sz_prior[-1,-1] = var_lstm
        # Predicted mean and var
        y_pred = self.F @ z_prior
        Sy_pred = self.F @ Sz_prior @ self.F.T

        if mu_obs:
            #
            cov_zy =  Sz_prior @ self.F.T
            var_y = self.F @ Sz_prior @ self.F.T + var_obs
            # delta for mean z and var Sz
            cov_= cov_zy/var_y
            delta_mean =  cov_ * (mu_obs - y_pred)
            delta_var  = - np.matmul(cov_, cov_zy.T)
            # update mean for mean z and var Sz
            z_posterior = z_prior + delta_mean
            Sz_posterior = Sz_prior + delta_var
            # detla for mean and var to update LSTM (parameters in net)
            delta_mean_lstm = delta_mean[-1,-1]/var_lstm
            delta_var_lstm  = delta_var[-1,-1]/var_lstm**2
            #
            self.zB = z_posterior[:-1,:]
            self.SzB = Sz_posterior[:-1,:-1]
        else:
            delta_mean_lstm = None
            delta_var_lstm = None
            self.zB = zB_prior
            self.SzB = SzB_prior
            z_posterior  = z_prior
            Sz_posterior = Sz_prior

        # save prior and posterior of hs for smoothing
        # self.time_idx = self.time_idx + 1
        self.mu_priors  = np.concatenate((self.mu_priors, z_prior), axis=1)
        self.cov_priors = np.concatenate((self.cov_priors, Sz_prior.reshape(-1,1)), axis=1)
        self.mu_posteriors  = np.concatenate((self.mu_posteriors, z_posterior), axis=1)
        self.cov_posteriors = np.concatenate((self.cov_posteriors, Sz_posterior.reshape(-1,1)), axis=1)

        return np.array([y_pred]).flatten(), np.array([Sy_pred]).flatten(), np.array([delta_mean_lstm]).flatten(), np.array([delta_var_lstm]).flatten()

    def smoother(self):
        nb_obs = self.mu_priors.shape[1]
        nb_hs = self.mu_priors.shape[0]
        mu_smoothed  = np.zeros((nb_hs,nb_obs))
        cov_smoothed = np.zeros((nb_hs**2,nb_obs))
        mu_smoothed[:,-1] = self.mu_posteriors[:,-1]
        cov_smoothed[:,-1] = self.cov_posteriors[:,-1]
        A = self.A
        A = np.concatenate((A,np.zeros((nb_hs-1,1))),axis=1)
        A = np.concatenate((A,np.zeros((1,nb_hs))),axis=0)

        for i in range(nb_obs-2,0,-1):
            # J = self.cov_posteriors[:,i].reshape(nb_hs,nb_hs) @ A.T \
            #     @ inv(self.cov_posteriors[:,i+1].reshape(nb_hs,nb_hs) \
            #           + 1E-8* np.eye(nb_hs))
            J = self.cov_posteriors[:,i].reshape(nb_hs,nb_hs) @ A.T \
                @ pinv(self.cov_posteriors[:,i+1].reshape(nb_hs,nb_hs),rcond=1e-12)
            mu_smoothed[:,i] = self.mu_posteriors[:,i] \
                + J @ (mu_smoothed[:,i+1] - self.mu_priors[:,i+1])
            cov_ = self.cov_posteriors[:,i].reshape(nb_hs,nb_hs) + \
                J @ (cov_smoothed[:,i+1].reshape(nb_hs,nb_hs) - self.cov_priors[:,i+1].reshape(nb_hs,nb_hs)) @ J.T
            cov_smoothed[:,i] = cov_.flatten()

        self.mu_smoothed  = mu_smoothed
        self.cov_smoothed = cov_smoothed

    def init_ssm_hs(self):
        nb_hs = len(self.zB) + 1
        self.mu_priors = np.zeros((nb_hs,1))
        self.cov_priors = np.zeros((nb_hs**2,1))
        self.mu_posteriors = np.zeros((nb_hs,1))
        self.cov_posteriors = np.zeros((nb_hs**2,1))
        if hasattr(self,'mu_smoothed'):
            self.zB  = self.mu_smoothed[:-1,1].reshape(-1,1)
            SzB_ = self.cov_smoothed[:,1].reshape(nb_hs,nb_hs)
            self.SzB = SzB_[:-1,:-1]

def process_input_ssm(
    x: np.ndarray,
    mu_preds_lstm: list,
    input_seq_len: int,
    num_features: int,
    ):
    mu_preds_lstm = np.array(mu_preds_lstm)

    if len(mu_preds_lstm)>=input_seq_len:
        x[-input_seq_len*num_features::num_features] =  mu_preds_lstm[-input_seq_len:]

    # mu_preds_lstm = np.array(mu_preds_lstm)
    # x_ = np.zeros(input_seq_len)
    # nb_replace = min(len(mu_preds_lstm),input_seq_len)
    # if nb_replace>=input_seq_len:
    #     x_[-nb_replace:] =  mu_preds_lstm[-nb_replace:]
    # x[-input_seq_len*num_features::num_features] = x_

    return x










# len_delta = len(net.input_delta_z_buffer.delta_mu)
# delta_mu_net = np.concatenate((delta_mu_net, np.zeros((len_delta-1))), axis=0)
# delta_var_net = np.concatenate((delta_var_net, np.zeros((len_delta-1))), axis=0)