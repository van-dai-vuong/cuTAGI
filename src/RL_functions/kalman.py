import numpy as np
import scipy.linalg
import scipy.stats
import math
from src.RL_functions.bounded_AR import BAR

##########################  Kalman Filter  #################################
def KF(mu_tt:np.ndarray, cov_tt:np.ndarray, observation:np.ndarray, At:np.ndarray, Ct:np.ndarray, Qt:np.ndarray, Rt:np.ndarray,\
       isBounded:bool, option_BAR):
  """ Kalman filter for one time step. This function includes prediction and filter step
  Args:
      mu_tt (np.ndarray): expected value of hidden states at time t-1, [#hidden state, 1]
      cov_tt (np.ndarray): covariance matrix of hidden states at time t-1, [#hidden state, #hidden state]
      observation (np.ndarray): observation at time t, [#observation, 1]
      At (np.ndarray): model transition matrix at time t, [#hidden state, #hidden state]
      Ct (np.ndarray): model observation matrix at time t, [#observation, #hidden state]
      Qt (np.ndarray): error covariance matrix of transition model at time t, [#hidden state, #hidden state]
      Rt (np.ndarray): observation error covariance matrix at time t, [#observation, #observation]
      isBounded (bool): Bound AR component or not, [1,]
      option_BAR (np.ndarray): phi_AR, var_AR_w, gamma_val [#BAR components, 3]

  Returns:
      mu_tt_ (np.ndarray): expected value of hidden states at time t, [#hidden state, 1]
      cov_tt_ (np.ndarray): covariance matrix of hidden states at time t, [#hidden state, #hidden state]
      mu_t_t (np.ndarray): expected value of hidden states after prediction before filter, [#hidden state, 1]
      cov_t_t (np.ndarray): covariance matrix of hidden states at time t after prediction before filter, [#hidden state, #hidden state]
      L_tt_ (np.ndarray): the likelihood of the observation given the filtered mu_tt_ and cov_tt_, [scaler]
  """

  ##Prediction step
  # Transition step
  mu_t_t=np.dot(At,mu_tt)
  cov_t_t=At.dot(cov_tt).dot(At.T)+Qt

  # Bound step
  if isBounded:
    AR_pos = -2
    BAR_pos = -1
    # Read values
    phi_AR=option_BAR[0]
    var_AR_w=option_BAR[1]
    gamma_val=option_BAR[2]

    # muBAR_t_t_, covBAR_t_t_, cov_XAR_XBAR_t_t=BAR(mu_t_t[AR_pos],cov_t_t[AR_pos][AR_pos],gamma_val, phi_AR, var_AR_w)
    muBAR_t_t_, covBAR_t_t_, cov_X_XBAR_t_t=BAR(mu_t_t[AR_pos],cov_t_t[AR_pos,0:AR_pos+1],gamma_val, phi_AR, var_AR_w)
    if np.isnan(muBAR_t_t_).any() or np.isinf(muBAR_t_t_).any():
        print("muBAR_t_t_ contains NaN or Inf")

    if np.isnan(covBAR_t_t_).any() or np.isinf(covBAR_t_t_).any():
        print("covBAR_t_t_ contains NaN or Inf")

    if np.isnan(cov_X_XBAR_t_t).any() or np.isinf(cov_X_XBAR_t_t).any():
        print("cov_X_XBAR_t_t contains NaN or Inf")

    mu_t_t[BAR_pos]=muBAR_t_t_
    cov_t_t[BAR_pos,BAR_pos]=covBAR_t_t_
    cov_t_t[0:AR_pos+1,BAR_pos]=cov_X_XBAR_t_t
    cov_t_t[BAR_pos,0:AR_pos+1]=cov_X_XBAR_t_t

  # Observation model
  mu_y_t_t=np.dot(Ct,mu_t_t)
  cov_y_t_t=Ct.dot(cov_t_t).dot(Ct.T)+Rt
  cov_xy_t_t=cov_t_t.dot(Ct.T)

  ##Update step
  if np.isnan(observation).any():
    # When no observation, skip update step
    mu_tt_=mu_t_t
    cov_tt_=cov_t_t
  else:
    if isinstance(observation, float):
      mu_tt_=mu_t_t+cov_xy_t_t*(1/cov_y_t_t)*(observation-mu_y_t_t)
      cov_tt_=cov_t_t-np.reshape(cov_xy_t_t, (-1, len(mu_tt))).T*(1/cov_y_t_t)@np.reshape(cov_xy_t_t, (-1, len(mu_tt)))
    else:
      mu_tt_=mu_t_t+cov_xy_t_t@(np.linalg.inv(cov_y_t_t))@(observation-mu_y_t_t)
      cov_tt_=cov_t_t-cov_xy_t_t@(np.linalg.inv(cov_y_t_t))@cov_xy_t_t.T

  ##Likelihood
  L_tt_=scipy.stats.multivariate_normal(cov=cov_y_t_t, mean=mu_y_t_t, allow_singular=True).pdf(observation)
  if L_tt_ == 0:
    L_tt_ = 1e-300

  return mu_tt_,cov_tt_,mu_t_t, cov_t_t, L_tt_

##########################  Kalman Smoother  #################################
def KS(mu_tt:np.ndarray, mu_t_T:np.ndarray, mu_t_t:np.ndarray, cov_tt:np.ndarray, cov_t_T:np.ndarray, cov_t_t:np.ndarray, At_:np.ndarray):
  """ Kalman smoother for one time step.
  Args:
      mu_tt (np.ndarray): expected value of hidden states at time t after Kalman filter [#hidden state, 1]
      mu_t_T (np.ndarray): expected value of hidden states at time t+1 after Kalman smoother, [#hidden state, 1]
      mu_t_t (np.ndarray): expected value of hidden states at time t+1 after prediction before filter, [#hidden state, 1]
      cov_tt (np.ndarray): covariance matrix of hidden states at time t after Kalman filter [#hidden state, #hidden state]
      cov_t_T (np.ndarray): covariance matrix of hidden states at time t+1 after Kalman smoother, [#hidden state, #hidden state]
      cov_t_t (np.ndarray): covariance matrix of hidden states at time t+1 after prediction before filter, [#hidden state, #hidden state]
      At_ (np.ndarray): model transition matrix at time t+1, [#hidden state, #hidden state]

  Returns:
      mu_tT (np.ndarray): expected value of hidden states at time t, [#hidden state, 1]
      cov_tT (np.ndarray): covariance matrix of hidden states at time t, [#hidden state, #hidden state]
  """
  J_t=cov_tt.dot(At_.T).dot(np.linalg.inv(cov_t_t))
  mu_tT=mu_tt+J_t.dot(mu_t_T-mu_t_t)
  cov_tT=cov_tt+J_t.dot(cov_t_T-cov_t_t).dot(J_t.T)

  return mu_tT,cov_tT

##########################  (merged) Switching Kalman Filter  #################################
def merge_SKF(mu:np.ndarray, cov:np.ndarray, pi:np.ndarray, L_t_old, A:np.ndarray, C:np.ndarray, Q:np.ndarray,\
              R:np.ndarray,observation,Z,isBounded:bool, option_BAR):
  '''
  Args:
      mu (np.ndarray): tuple of vectors of expected values for each regime at time t-1 [[mu1],[mu2],...], [#regime,1]
      cov (np.ndarray): tuple of covariance matrices for each regime at time t-1 [[cov1],[cov2],...], [#regime,1]
      pi (np.ndarray): vector of probability of being at regime i at t-1 [pi1,pi2,...], [#regime,1]
      A (np.ndarray): transition matrices from regime i at t-1 to regime j at t [[A11],[A12],[A21],[A22],...], [#regime, #regime]
      C (np.ndarray): observation matrices from regime i at t-1 to regime j at t [[C11],[C12],[C21],[C22],...], [#regime, #regime]
      Q (np.ndarray): error covariance matrices from regime i at t-1 to regime j at t [[Q11],[Q12],[Q21],[Q22],...], [#regime, #regime]
      R (np.ndarray): observation error covariance matrices from regime i at t-1 to regime j at t [[R11],[R12],[R21],[R22],...]
      observation (np.ndarray): observation at time t, [#observation, 1]
      Z (np.ndarray): prior knowledge of probability transiting from one state i at t-1 to another j at t, [#regime, #regime]
      isBounded (bool): Bound AR component or not, [1,]
      option_BAR (np.ndarray): phi_AR, var_AR_w, gamma_val [#BAR components, 3]

      Notes: A, C, Q, R, Z are matrices that contains parameter to learn by optimization algorithm

  Returns:
      mu_ (np.ndarray): tuple of vectors of merged expected values for each regime at time t [[mu_1],[mu_2],...]
      cov_ (np.ndarray): tuple of merged covariance matrices for each regime i at time t [[cov_1],[cov_2],...]
      pi_ (np.ndarray): tuple of merged probabilities of being at regime i at t [pi_1,pi_2,...]
      mu_merged (float): final merged mean value of X_LL from different regimes at time t, scaler
      var_merged (float): final merged variance of X_LL from different regimes at time t, scaler
  '''
  # Initialization
  n_regime=mu.shape[0]

  mu_t=np.zeros((n_regime,n_regime), dtype=tuple)
  cov_t=np.zeros((n_regime,n_regime), dtype=tuple)
  L_t=L_t_old
  M_log=np.zeros((n_regime,n_regime), dtype='float64')
  W=np.zeros((n_regime,n_regime), dtype='float64')

  mu_W=np.zeros((n_regime,n_regime), dtype=tuple)
  mu_mu=np.zeros((n_regime,n_regime), dtype=tuple)
  W_cov_mu=np.zeros((n_regime,n_regime), dtype=tuple)

  mu_=np.zeros((n_regime,), dtype=tuple)
  cov_=np.zeros((n_regime,), dtype=tuple)
  pi_=np.zeros((n_regime,), dtype=np.float32)

  # Calculate the sum of M matrix
  for i in range(n_regime):
    for j in range(n_regime):
      if np.isnan(observation).any():
        mu_t[i][j],cov_t[i][j],_,_,_=KF(mu[i],cov[i],observation,\
                                              A[i][j],C[i][j],Q[i][j],R[i][j],\
                                              isBounded, option_BAR)
      else:
        mu_t[i][j],cov_t[i][j],_,_,L_t[i][j]=KF(mu[i],cov[i],observation,\
                                              A[i][j],C[i][j],Q[i][j],R[i][j],\
                                              isBounded, option_BAR)

      ## Numerical instable
      # M[i][j]=L_t[i][j]*Z[i][j]*pi[i]

      # More stable method: likelihood -> loglikelihood
      M_log[i][j]=np.log(L_t[i][j]) + np.log(Z[i][j]) + np.log(pi[i])

  # More stable method: loglikelihood -> likelihood
  M = np.exp(M_log)
  if (M==0).any():
    M = np.exp(M_log + (299 - np.max(M_log)))
  if (M==0).any():
    M[np.argwhere(M == 0)] = 1E-100

  M_sum=np.sum(M)

  # Calculate pi_
  for i in range(n_regime):
    for j in range(n_regime):
      M[i][j]=M[i][j]/M_sum

  pi_=np.sum(M,axis=0)

  # Calculate mu_
  for i in range(n_regime):
    for j in range(n_regime):
      W[i][j]=M[i][j]/np.sum(M,axis=0)[j]
      mu_W[i][j]=mu_t[i][j]*W[i][j]

  mu_=np.sum(mu_W,axis=0)

  # Calculate cov_
  for i in range(n_regime):
    for j in range(n_regime):
      mu_mu[i][j]=mu_t[i][j]-mu_[j]
      W_cov_mu[i][j]=W[i][j]*(cov_t[i][j]+\
                              np.reshape(mu_mu[i][j], (-1, np.size(mu[i][j]))).T@np.reshape(mu_mu[i][j], (-1, np.size(mu[i][j]))))

  cov_=np.sum(W_cov_mu,axis=0)

  ##Merge different regimes at time t
  n_component=mu_[0].shape[0]
  mu_merged_=np.zeros((n_component,), dtype='float64')
  var_merged_=np.zeros((n_component,), dtype='float64')
  for i in range(n_component):
    mui_t_=np.zeros((n_regime,), dtype='float64')
    vari_t_=np.zeros((n_regime,), dtype='float64')
    for j in range(n_regime):
      mui_t_[j]=mu_[j][i]
      vari_t_[j]=cov_[j][i][i]

    mu_merged_[i]=np.sum(pi_*mui_t_)
    var_merged_[i]=np.sum(pi_*vari_t_)+pi_[0]*pi_[1]*(mui_t_[0]-mui_t_[1])**2
    # np.sum(pi_*mui_t_**2)-(np.sum(pi_*mui_t_))**2

  return mu_,cov_,pi_,L_t,mu_merged_, var_merged_

##########################  Switching Kalman Smoother  #################################
def switching_kalman_smoother(mu_tp_T:np.ndarray, cov_tp_T:np.ndarray, pi_tp_T:np.ndarray, L_t_old, A:np.ndarray, C:np.ndarray, Q:np.ndarray,\
              R:np.ndarray,observation,Z,isBounded:bool, option_BAR):
  pass

class SKS_class():
    # Switching Kalman Smoother
    def __init__(self, mu_tp_T:np.ndarray, cov_tp_T:np.ndarray, pi_tp_T:np.ndarray, mu_t_t:np.ndarray, cov_t_t:np.ndarray, pi_t_t:np.ndarray,\
                 A:np.ndarray, Q:np.ndarray, Z:np.ndarray, isBounded=False, option_BAR=[0.75, 36, 2]):
      '''
      A, C, Q, R matrices come from the time step t+1 (tP here)
      '''
      self.mu_tp_T = mu_tp_T
      self.cov_tp_T = cov_tp_T
      self.pi_tp_T = pi_tp_T
      self.mu_t_t = mu_t_t
      self.cov_t_t = cov_t_t
      self.pi_t_t = pi_t_t
      self.A = A
      self.Q = Q
      self.Z = Z
      self.isBounded = isBounded
      self.option_BAR = option_BAR
      self.num_regime = len(self.mu_tp_T)
      self.num_hidden_states = len(self.mu_tp_T[0])

    def smooth(self):
      ''' To perform Kalman Smoother from time t+1 to time t. tp means t+1
      Arguments:
      Input:  mu_tp_T:
              cov_tp_T:
              mu_t_t:
              cov_t_t:
              A: Transition matrix at time t+1
              Q: transition error matrix at time t+1
      Output: mu_t_T:
              cov_t_T:
      '''
      u_jk = np.zeros((self.num_regime,self.num_regime), dtype=tuple)
      pi_t_tp_T = np.zeros((self.num_regime,self.num_regime), dtype=tuple)
      mu_jk_t_T = np.zeros((self.num_regime,self.num_regime), dtype=tuple)
      cov_jk_t_T = np.zeros((self.num_regime,self.num_regime), dtype=tuple)
      w_jk_t_T = np.zeros((self.num_regime,self.num_regime), dtype=tuple)
      mu_j_t_T = np.zeros((self.num_regime,), dtype=tuple)
      cov_j_t_T = np.zeros((self.num_regime,), dtype=tuple)

      sum_MZ_k = self.Z.T @ self.pi_t_t

      for j in range(self.num_regime):
        for k in range(self.num_regime):
          mu_jk_t_T[j][k], cov_jk_t_T[j][k]= self.smooth_regime_ktoj(self.mu_tp_T[k], self.cov_tp_T[k], self.mu_t_t[j], self.cov_t_t[j], \
                                                                     self.A[j][k], self.Q[j][k], self.isBounded, self.option_BAR)
          u_jk[j][k] = self.pi_t_t[j] * self.Z[j][k]/sum_MZ_k[k]
          pi_t_tp_T[j][k] = u_jk[j][k] * self.pi_tp_T[k]
      pi_t_T = np.sum(pi_t_tp_T,axis=1) # Smoothed regime probability

      for j in range(self.num_regime):
        for k in range(self.num_regime):
          w_jk_t_T[j][k] = pi_t_tp_T[j][k]/pi_t_T[j]
        mu_j_t_T[j], cov_j_t_T[j] = self.collapse(mu_jk_t_T[j][:], cov_jk_t_T[j][:], w_jk_t_T[j][:])

      mu_t_T, cov_t_T = self.collapse(mu_j_t_T, cov_j_t_T, pi_t_T) # Smoothed hidden states (merged)

      return mu_t_T, cov_t_T, pi_t_T, mu_j_t_T, cov_j_t_T

    def smooth_regime_ktoj(self, mu_k_tp_T, cov_k_tp_T, mu_j_t_t, cov_j_t_t, A_jk, Q_jk, isBounded=False, option_BAR=[0.75, 36, 2]):
      ''' To perform Kalman Smoother from one regime k at time t+1 to another regime j at time t. tp means t+1
      Arguments:
      Input:  mu_k_tp_T:
              cov_k_tp_T:
              mu_j_t_t:
              cov_j_t_t:
              j:
              k:
      Output: mu_ktoj_t_T:
              cov_ktoj_t_T:
      '''
      mu_k_tp_t = A_jk @ mu_j_t_t
      cov_k_tp_t = A_jk @ cov_j_t_t @ A_jk.T + Q_jk

      # # BAR 2 steps
      # if isBounded:
      #   AR_pos = -2
      #   BAR_pos = -1
      #   phi_AR = option_BAR[0]
      #   var_AR_w = option_BAR[1]
      #   gamma_val = option_BAR[2]

      #   # Pass the AR estimation to mReLU
      #   muBAR_k_tp_t, covBAR_k_tp_t, cov_X_XBAR_k_tp_t=BAR(mu_k_tp_t[AR_pos],cov_k_tp_t[AR_pos,0:AR_pos+1],gamma_val, phi_AR, var_AR_w)

      #   # Aggregate BAR hidden state to the others
      #   mu_k_tp_t[BAR_pos] = muBAR_k_tp_t
      #   cov_k_tp_t[BAR_pos,BAR_pos] = covBAR_k_tp_t
      #   cov_k_tp_t[0:AR_pos+1,BAR_pos] = cov_X_XBAR_k_tp_t
      #   cov_k_tp_t[BAR_pos,0:AR_pos+1] = cov_X_XBAR_k_tp_t

      J_t = cov_j_t_t.dot(A_jk.T).dot(np.linalg.pinv(cov_k_tp_t, rcond=1e-3))

      mu_ktoj_t_T = mu_j_t_t + J_t.dot(mu_k_tp_T - mu_k_tp_t)
      cov_ktoj_t_T = cov_j_t_t + J_t.dot(cov_k_tp_T - cov_k_tp_t).dot(J_t.T)

      return mu_ktoj_t_T, cov_ktoj_t_T

    def collapse(self, mu_all:np.ndarray, cov_all:np.ndarray, pi_all:np.ndarray):
      ''' Collapse step based on Gaussian Mixture Reduction
      Arguments:
      Input:  mu_all:
              cov_all:
              pi_all:
      Output: mu:
              cov:
      '''
      mu = pi_all @ mu_all

      num_components = len(pi_all)

      mu_dupli = np.tile(mu, (num_components,1))
      error_mu_all_mu = np.vstack(mu_all) - mu_dupli

      cov_n = np.zeros((num_components,), dtype=tuple) # Covariance for each regime
      for n in range(num_components):
        cov_n[n] = cov_all[n] + error_mu_all_mu[n].reshape(1,self.num_hidden_states).T @ \
          error_mu_all_mu[n].reshape(1,self.num_hidden_states)

      cov = pi_all @ cov_n

      return mu, cov


##########################  Check point for SKF  #################################
# To make sure that the model is correctly defined and no anomalies are triggered by SKF in the real time series
def SKF_check(hyperparameters, x_init, timestamps, observations, time_step_interval):
  SKF_class=SKF(timestamps, observations)
  configuration_skf_bar = {
        "LT_sigma_w": np.sqrt(hyperparameters['autoregressive_acceleration']['process_error_var']),
        "LA_sigma_w": np.sqrt(hyperparameters['autoregressive_acceleration']['process_error_var']),
        "KR_p": hyperparameters['kernel']['period'],
        "KR_ell": hyperparameters['kernel']['kernel_length'],
        "KR_sigma_w": np.sqrt(hyperparameters['kernel']['sigma_KR0']),
        "KR_sigma_hw": np.sqrt(hyperparameters['kernel']['sigma_KR1']),
        "AR_phi": hyperparameters['ar']['phi'],
        "AR_sigma_w": np.sqrt(hyperparameters['ar']['process_error_var']),
        "sigma_v": np.sqrt(hyperparameters['observation']['error']),
        "X_init": x_init['mu'],
        "V_init": x_init['var'],
        "Z_11": 0.9999999,
        "Z_22": 0.9999999,
        "pi_1": 0.999,
        "pi_2": 0.001,
        "isBounded": False,
        "BAR_gamma_value": 0,
        "sigma_12": 1e-06,
        "timestep_ref": time_step_interval
        }
  SKF_class.import_configuration(configuration_skf_bar)

  prob_ns_regime, x_mu, x_var, y_mu, y_var = SKF_class.run_skf()
  return prob_ns_regime, y_mu, y_var, x_mu, x_var


from src.RL_functions.build_Matrix import *
from src.RL_functions.kalman import *
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
        self.timestep_ref = configuration["timestep_ref"]

    def run_skf(self):
        # Set BAR option
        options_BAR = [self.AR_phi, self.AR_sigma_w**2, self.BAR_gamma_value]

        timestep_ref = self.timestep_ref

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
        y_mu = np.zeros_like(self.synthetic_observations, dtype=float)
        y_var = np.zeros_like(self.synthetic_observations, dtype=float)
        x_mu = np.zeros((len(self.synthetic_observations), 15))
        x_var = np.zeros((len(self.synthetic_observations), 15))
        for i in range(len(self.synthetic_timestamps)):
            ### Model 1
            LTcA_A, LTcA_C, LTcA_Q = LA_design_matrix(self.LT_sigma_w , timestep_ref)
            LTcA_A[:,2] = 0

            ### Model 2
            LA_A, LA_C, LA_Q = LA_design_matrix(self.LA_sigma_w , timestep_ref)

            KR_A, KR_C, KR_Q = KR_design_matrix(self.KR_sigma_hw, self.KR_sigma_w, self.KR_p, self.KR_ell, self.synthetic_timestamps[i], self.synthetic_timestamps[0], len(KR_xcontrol)-1)
            AR_A, AR_C, AR_Q = AR_design_matrix(self.AR_sigma_w, self.AR_phi)

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

            n_regime=mu.shape[0]
            y_mu_sep = np.zeros((n_regime,), dtype=tuple)
            y_var_sep = np.zeros((n_regime,), dtype=tuple)
            for k in range(n_regime):
              y_mu_sep[k] = C[k][k]@mu[k]
              y_var_sep[k]= C[k][k]@cov[k]@C[k][k].T + R[k][k]

            y_mu[i] = np.sum(pi*y_mu_sep)
            y_var[i]= np.sum(pi*y_var_sep)+pi[0]*pi[1]*(y_mu_sep[0]-y_mu_sep[1])**2

            # y_mu[i] = mu_pred@C[0,0].T
            # y_var[i]= C[0][0]@var_pred
            x_mu[i] = mu_pred
            x_var[i] = var_pred

        return prob_ns_regime, x_mu, x_var, y_mu, y_var
