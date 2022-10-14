import numpy as np
def weight_diff(w1, w2):
    """ Calculates the array of differences between the weights in arrays """
    # Expand and flatten arrays
    _w1 = np.hstack([x.flatten() for x in w1])
    _w2 = np.hstack([x.flatten() for x in w2])
    return _w1 - _w2
                 
def l2_norm(w1, w2):
    return np.linalg.norm(weight_diff(w1, w2))        
        
def estimate_KL(posterior_weights,prior_weights,sigma):
    """
    provides an estimate of the KL-div between prior and posterior
    
    We assume that we have a multivariate gaussian centred at the the weights for the prior and posterior.
    We also assume that the variance of the prior is isotropic gaussian and also that it is the same in the prior.
    """
    KL=l2_norm(posterior_weights,prior_weights)**2
    KL=KL/(2*sigma)
    return KL