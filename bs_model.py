import numpy as np
from scipy.stats import norm



def bs_call(S, K, T, r, sigma):
    # Validate inputs
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("Illegal argument")
    
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    C = S* norm.cdf(d1)- K* np.exp(-r*T)* norm.cdf(d2)

    return C


