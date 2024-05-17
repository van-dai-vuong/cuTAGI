import numpy as np

#############################  LL  #####################################
def LL_design_matrix(sigma_LL:float):
  """ Generate Local Level component

  Args:
      sigma_LL (float): Standard deviation describing the prediction error for the locally constant model

  Return:
      A_LL (np.ndarray): model transition matrix for LL
      C_LL (np.ndarray): model observation matrix for LL
      Q_LL (np.ndarray): model error covariance matrix for LL
  """
  A_LL=np.array([1.0])
  C_LL=np.array([1.0])
  Q_LL=np.array([sigma_LL**2])

  return A_LL, C_LL, Q_LL

##############################  LT  ####################################
def LT_design_matrix(sigma_LT:float, time_step:float):
  """ Generate Local Trend component

  Args:
      sigma_LT (float): Standard deviation describing the prediction error for the locally trend model
      time_step (float): most likely time step length between observations

  Return:
      A_LT (np.ndarray): model transition matrix for LT, [2, 2]
      C_LT (np.ndarray): model observation matrix for LT, [1, 2]
      Q_LT (np.ndarray): model error covariance matrix for LT, [2, 2]
  """
  A_LT=np.array([[1.,time_step],
                 [0.,1.]])
  C_LT=np.array([1.,0.])
  Q_LT=sigma_LT**2*np.array([[time_step**4/4,time_step**3/2],
                            [time_step**3/2,time_step]])

  return A_LT, C_LT, Q_LT

##############################  LA  ####################################
def LA_design_matrix(sigma_LA:float, time_step:float):
  """ Generate Local Acceleration component

  Args:
      sigma_LA (float): Standard deviation describing the prediction error for the locally acceleration model
      time_step (float): most likely time step length between observations

  Return:
      A_LA (np.ndarray): model transition matrix for LA, [3, 3]
      C_LA (np.ndarray): model observation matrix for LA, [1, 3]
      Q_LA (np.ndarray): model error covariance matrix for LA, [3, 3]
  """
  A_LA=np.array([[1., time_step, time_step**2],
                 [0., 1.       ,time_step],
                 [0., 0.       ,1.]])
  C_LA=np.array([1.,0.,0.])
  Q_LA=sigma_LA**2*np.array([[time_step**4/4,time_step**3/2,time_step**2/2],
                             [time_step**3/2,time_step**2. ,time_step],
                             [time_step**2/2,time_step     ,1]])

  return A_LA, C_LA, Q_LA

 ##############################  Fourrier Form  ####################################
def fourrier_design_matrix(sigma_s:float, period:float, time_step:float):
  """ Generate periodic component in fourrier form

  Args:
      Model parameters:
      sigma_s (float): Standard deviation describing the prediction error for the fourrier form periodic component
      period (float): period of the Trigonometric functions

      Time series characteristic
      time_step (float): most likely time step length between observations


  Return:
      A_s (np.ndarray): model transition matrix for fourrier periodic component, [2, 2]
      C_s (np.ndarray): model observation matrix for fourrier periodic component, [1, 2]
      Q_s (np.ndarray): model error covariance matrix for fourrier periodic component, [2, 2]
  """

  w = 2*np.pi*time_step/period
  A_s=np.array([[np.cos(w) , np.sin(w)],
                [-np.sin(w), np.cos(w)]])
  C_s=np.array([1,0])
  Q_s=sigma_s**2*np.array([[1,0],
                           [0,1]])
  return A_s, C_s, Q_s


############################## Kernel Regression  ####################################
def KR_design_matrix(sigma_KR0:float, sigma_KR1:float, period: float, kernel_length:float\
                     ,time:float, control_time_begin:float, n_cp:float):
  """ Generate periodic component at time t using Kenel regression

  Args:
      Model parameters:
      sigma_KR0 (float): Standard deviation controls the time-independent process noise
                         in the hidden state variable associated with the Kernel pattern
      sigma_KR1 (float): Standard deviation controls the increase in the variance of the
                         hidden state variables associated with the control point's value
                         between successive time steps
      period (float): period of the kernel regression, p
      kernel_length (float): length of the kernel regression, l

      Time series characteristics:
      time (float): the timestamp when the target observation point happens
      control_time_begin (float): the timestamp of the first control point
      n_cp (float): number of control points

  Return:
      A_KR_t (np.ndarray): model transition matrix for kernel regression component dependent on time t, [n_cp+1, n_cp+1]
      C_KR_t (np.ndarray): model observation matrix for kernel regression component dependent on time t, [1, n_cp+1]
      Q_KR_t (np.ndarray): model error covariance matrix for kernel regression component dependent on time t, [n_cp+1, n_cp+1]

  Notes:
      Different from LL, LT, LA and Fourrier periodic components, where the A, C, Q
        matrices are fixed throughout all time series, here the A matrix is different
        at each time.
  """
  kapa=0
  #Kapa: autocorrelation between kernel pattern values at consecutive time, 0-no, 1-yes

  T_cp=np.linspace(control_time_begin,control_time_begin+period,num=n_cp,endpoint=False)
  T=np.repeat(time,len(T_cp))

  K_raw=np.exp(-2/kernel_length**2*np.sin((np.pi*(T-T_cp)/period))**2)
  K_norm=K_raw/(np.sum(K_raw)+10**(-8))

  A_KR_t=np.vstack((np.hstack(( np.array([kapa]) , K_norm )),\
                    np.hstack(( np.zeros((n_cp,1)) ,np.eye(n_cp)))))

  Q_KR_t=np.vstack((np.hstack((np.array([sigma_KR0**2]),np.zeros((n_cp)))),\
                    np.hstack((np.zeros((n_cp,1)),np.eye(n_cp)*sigma_KR1**2))))
  C_KR_t=np.concatenate((np.array([1]),np.zeros(n_cp)),axis=0)
  return A_KR_t, C_KR_t, Q_KR_t


###########################  AR  #######################################
def AR_design_matrix(sigma_AR:float,phi_AR:float):
  """ Generate Autoregressive component of 1 order

  Args:
      Model parameters:
      sigma_AR (float): Standard deviation describing the prediction error for the autoregressive model
      phi_AR (float): AR coefficient, range 0<phi_AR<1

  Return:
      A_AR (np.ndarray): model transition matrix for AR
      C_AR (np.ndarray): model observation matrix for AR
      Q_AR (np.ndarray): model error covariance matrix for AR
  """
  A_AR=np.array([phi_AR])
  C_AR=np.array([1.0])
  Q_AR=np.array([sigma_AR**2])

  return A_AR, C_AR, Q_AR

############################  BAR  ######################################
def BAR_design_matrix(sigma_AR:float,phi_AR:float):
  """ Generate Autoregressive component of 1 order

  Args:
      Model parameters:
      sigma_AR (float): Standard deviation describing the prediction error for the autoregressive model
      phi_AR (float): AR coefficient, range 0<phi_AR<1

  Return:
      A_AR (np.ndarray): model transition matrix for AR
      C_AR (np.ndarray): model observation matrix for AR
      Q_AR (np.ndarray): model error covariance matrix for AR
  """
  A_BAR=np.array([[phi_AR, 0.],
                 [0., 0.]])
  C_BAR=np.array([0.,1.])
  Q_BAR=np.array([[sigma_AR**2,0.],
                 [0., 0.]])

  return A_BAR, C_BAR, Q_BAR