def log_sum_exp_stable(z):
    """
    Computes cross entropy loss. More stable implementation.
    
    Parameters
    ----------
    z: numpy.ndarray containing outputes of linear model
    
    Returns
    -------
    float : loss
    
    Examples:
    --------
    >>> log_sum_exp_stable(np.ones(100))
    5.605170185988092
    """
    a = np.max(z)
    return a + np.log(np.sum(np.exp(z-a)))

def log_1_plus_exp_safe(z):
    """
    Computes log(1+e^z). More stable implementation.
    
    Parameters
    ----------
    z: numpy.ndarray containing numbers that should be passed to the function
    
    Returns
    -------
    numpy.ndarray: Output of the function applied to each of the numbers 
    """
    if z > 100:
        return z
    else:
        return np.log(1+np.exp(z))
