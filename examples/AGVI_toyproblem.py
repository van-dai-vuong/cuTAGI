import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats



truesig = 0.5
N=200

#initials
mu_w2hat = 2**2
sig_w2hat = 1**2

sauv=[[np.sqrt(mu_w2hat),np.sqrt(sig_w2hat)]]

for n in range(N):
    # New observation
    observe = np.random.normal(0,truesig)
    # New observation ( W|y)
    mu_wy           = observe # mu_{t|t}^{W}
    sig_wy          = 0
    # Equations 2.10 (Approximation of W^2)
    mu_w2y          = mu_wy**2 + sig_wy
    sig_w2y         = 2*sig_wy**2 + 4 * sig_wy*mu_wy**2
    # Equations 2.11 (Transition of W^2)
    mu_w2           = mu_w2hat     # mu_{t|t-1}^{W2}
    cov_w2          = 3*sig_w2hat + 2*mu_w2hat**2
    # Equations 2.12 (Update of \overline{W^2})
    Ktw              = sig_w2hat/cov_w2
    mu_w2hat        = mu_w2hat + Ktw*(mu_w2y - mu_w2) # mu_{t|t}^{V2hat}
    sig_w2hat       = sig_w2hat + Ktw**2*(sig_w2y - cov_w2)
    # Memoire
    sauv.append([np.sqrt(mu_w2hat),np.sqrt(sig_w2hat)])


plt.figure()
sauv = np.array(sauv)
n = np.arange(0,N+1,1)
plt.plot(n,np.ones([N+1,1])*truesig, '--', color = 'black',label='True sigma_W' )
plt.plot(n,sauv[:,0],label='mu_w2hat' )
plt.fill_between(n, sauv[:,0]-sauv[:,1], sauv[:,0]+sauv[:,1], facecolor='gray',  alpha=0.5,label='mu_w2hat \u00B1 sig_w2hat')
plt.fill_between(n, sauv[:,0]-2*sauv[:,1], sauv[:,0]+2*sauv[:,1], facecolor='gray',  alpha=0.3,label='mu_w2hat \u00B1 2*sig_w2hat')
plt.legend()
plt.savefig('EstimeSigmaw.png', dpi=400,)