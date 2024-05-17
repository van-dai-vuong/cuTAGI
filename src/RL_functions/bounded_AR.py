import numpy as np
import scipy
import scipy.stats
import math

# def stable_ratio_compute(a, b):
#     sign_b = np.sign(b)

#     # Truncate b if it's close to overflow
#     if abs(b) < 1e-300:
#         b = 1e-300
#     if abs(b) > 1e300:
#         b = 1e300

#     # Scale b if it's very small or very large
#     if abs(b) < 1.0:
#         scale_factor = -np.log10(abs(b))
#         b = sign_b
#     elif abs(b) > 1e200:
#         scale_factor = -np.log10(abs(b))
#         b = sign_b
#     else:
#         scale_factor = 0.0

#     # Scale a by the same factor to maintain the ratio of a/b
#     a *= np.power(10, scale_factor)
#     return a / b


def stable_ratio_compute(a, b):
    result = a / b
    sign_a = math.copysign(1, a)
    sign_b = math.copysign(1, b)
    if np.isnan(result):
        result = sign_a * sign_b * 1e-8
    elif np.isinf(result):
        result = sign_a * sign_b * 1e8

    return result

def replace_inf_nan(array):
    array[np.isinf(array)] = 1e8
    array[np.isnan(array)] = 1e-8
    return array

##########################  BAR  #################################
def BAR(muAR_t_t:np.ndarray, covX_t_t:np.ndarray, gamma_val:np.ndarray, phi_AR, Q_AR):
    """ Kalman filter for one time step. This function includes prediction and filter step
    Args:
        muAR_t_t (np.ndarray): expected value of AR after transition step, [1,]
        covX_t_t (np.ndarray): variance of AR after transition step, [# hidden states-1,]
        gamma_val (np.ndarray): bounding coefficient, [1,]
        A_AR (float): phi_AR, [1,]
        Q_AR (float): (sigma^AR_w)^2, [1,]

    Returns:
        muBAR_t_t_ (np.ndarray): expected value of BAR at time t after bounding, [1,]
        covBAR_t_t_ (np.ndarray): covariance matrix of hidden states at time t after bounding, [1,]
        cov_XAR_XBAR_t_t (np.ndarray): covariance between XAR and XBAR, [1,]
    """
    covAR_t_t = covX_t_t[-1]
    b_val=gamma_val*np.sqrt(Q_AR/(1-phi_AR**2))
    alpha_L=-(b_val+muAR_t_t)/np.sqrt(covAR_t_t)
    alpha_U=(b_val-muAR_t_t)/np.sqrt(covAR_t_t)

    import scipy
    cdf_alpha_U=scipy.stats.norm.cdf(alpha_U)
    cdf_alpha_L=scipy.stats.norm.cdf(alpha_L)
    # w_val=max((cdf_alpha_U-cdf_alpha_L),1e-8) # After using stable_ratio_compute, we do not need to bound the w_val
    w_val=cdf_alpha_U-cdf_alpha_L

    #   beta_val=(scipy.stats.norm.pdf(alpha_U)-scipy.stats.norm.pdf(alpha_L))/w_val
    beta_val=stable_ratio_compute((scipy.stats.norm.pdf(alpha_U)-scipy.stats.norm.pdf(alpha_L)), w_val)
    #   kappa=1-(alpha_U*scipy.stats.norm.pdf(alpha_U)-alpha_L*scipy.stats.norm.pdf(alpha_L))/w_val-beta_val**2
    kappa=1-stable_ratio_compute((alpha_U*scipy.stats.norm.pdf(alpha_U)-alpha_L*scipy.stats.norm.pdf(alpha_L)),w_val)-beta_val**2
    
    # Moments of AR_tilde, AR in standard nomal space
    mu_AR_tilde=muAR_t_t-beta_val*np.sqrt(covAR_t_t)
    var_AR_tilde=kappa*covAR_t_t
    cov_X_AR_tilde=kappa**(1/2)*covX_t_t
    
    # Moments of BAR
    muBAR_t_t_=-b_val*cdf_alpha_L+w_val*mu_AR_tilde+b_val*(1-cdf_alpha_U)
    covBAR_t_t_=w_val*var_AR_tilde+w_val*(mu_AR_tilde-muBAR_t_t_)**2+\
                    cdf_alpha_L*(b_val+muBAR_t_t_)**2+\
                    (1-cdf_alpha_U)*(b_val-muBAR_t_t_)**2

    #lambda_val=(w_val*kappa)**0.5
    #lambda_val=np.sqrt(covBAR_t_t_)/np.sqrt(covAR_t_t)
    lambda_val=stable_ratio_compute(np.sqrt(covBAR_t_t_),np.sqrt(covAR_t_t))
    cov_X_XBAR_t_t=lambda_val*cov_X_AR_tilde
    
    # To avoid numerical instability
    cov_X_XBAR_t_t=replace_inf_nan(cov_X_XBAR_t_t)
    return muBAR_t_t_, covBAR_t_t_, cov_X_XBAR_t_t