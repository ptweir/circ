import numpy as np
from scipy.stats import chi2
from scipy.stats import norm
import warnings

def circmean(alpha,axis=None):
    mean_angle = np.arctan2(np.mean(np.sin(alpha),axis),np.mean(np.cos(alpha),axis))
    return mean_angle
    
def circvar(alpha,axis=None):
    if np.ma.isMaskedArray(alpha) and alpha.mask.shape!=():
        N = np.sum(~alpha.mask,axis)
    else:
        if axis is None:
            N = alpha.size
        else:
            N = alpha.shape[axis]
    R = np.sqrt(np.sum(np.sin(alpha),axis)**2 + np.sum(np.cos(alpha),axis)**2)/N
    V = 1-R
    return V

def circdiff(alpha,beta):
    D = np.arctan2(np.sin(alpha-beta),np.cos(alpha-beta))
    return D

def watson_test(data):
    """
    Y, p_value = watson_test(data)
    
    data -- list of numpy arrays containing different samples
    Y -- test statistic
    p_value -- p_value from chi-square lookup table
    
    non-parametric test for common mean direction from Watson [(Fisher, 1993)]
    numbers in brackets correspond to equations in Fisher, 1993
    """
    r = len(data) # number of samples
    n = np.ma.masked_all(r)
    thetaBar = np.ma.masked_all(r)
    delta = np.ma.masked_all(r)
    sigma = np.ma.masked_all(r)
    for i in range(r):
        theta = data[i]
        n[i] = float(len(theta))
        C = np.sum(np.cos(theta)) # [2.7]
        S = np.sum(np.sin(theta)) # [2.7]
        R = np.sqrt(S**2 + C**2) # resultant length of sample [2.7]
        RBar = R/n[i] # mean resultant length of sample [2.10]
        thetaBar[i] = np.arctan2(S,C) # mean direction of sample [2.9]
        rho = np.sum(np.cos(2*(theta-thetaBar[i])))/n[i] # second trigonometric moment of sample [2.27]
        delta[i] = (1-rho)/(2*RBar**2) # circular dispersion of sample [2.28]
        sigma[i] = np.sqrt(delta[i]/n[i]) # circular standard error of sample [4.21]
        
    N = np.sum(n)
    if np.min(n) < 25:
        warnings.warn('too few samples, not implemented')
        Yr = np.nan
    elif np.max(delta)/np.min(delta) <= 4: # method P
        Cp = np.sum(n*np.cos(thetaBar)) # [5.10]
        Sp = np.sum(n*np.sin(thetaBar)) # [5.10]
        Rp = np.sqrt(Cp**2 + Sp**2)
        delta0 = np.sum(n*delta/N) # [5.12]
        Yr = 2*(N - Rp)/delta0 # test statistic (reject hypothesis of common mean direction if Yr is too large) [5.13]
        
    elif max(delta)/min(delta) > 4:
        Cm = np.sum(np.cos(thetaBar)/sigma**2) # [5.14]
        Sm = np.sum(np.sin(thetaBar)/sigma**2) # [5.14]
        Rm = np.sqrt(Cm**2 + Sm**2) # [5.15]
        Yr = 2*(np.sum(1/sigma**2) - Rm) # test statistic (reject hypothesis of common mean direction if Yr is too large) [5.16]
        
    p_value = 1-chi2.cdf(Yr,r-1) # look up p value in chi-square table
    return Yr, p_value

def watson_two(theta1,theta2):
    Yr, p_value = watson_test([theta1, theta2])
    return Yr, p_value
    """
    n1 = len(theta1)
    C1 = np.sum(np.cos(theta1)) # [2.7]
    S1 = np.sum(np.sin(theta1)) # [2.7]
    R1 = np.sqrt(S1**2 + C1**2) # resultant length of sample 1 [2.7]
    RBar1 = R1/n1 # mean resultant length of sample 1 [2.10]
    thetaBar1 = np.arctan2(S1,C1) # mean direction of sample 1 [2.9]
    rho1 = np.sum(np.cos(2*(theta1-thetaBar1)))/n1 # second trigonometric moment of sample 1 [2.27]
    delta1 = (1-rho1)/(2*RBar1**2) # circular dispersion of sample 1 [2.28]
    sigma1 = np.sqrt(delta1/n1) # circular standard error of sample 1 [4.21]
    
    n2 = len(theta2)
    C2 = np.sum(np.cos(theta2)) # [2.7]
    S2 = np.sum(np.sin(theta2)) # [2.7]
    R2 = np.sqrt(S2**2 + C2**2) # resultant length of sample 2 [2.7]
    RBar2 = R2/n2 # mean resultant length of sample 2 [2.10]
    thetaBar2 = np.arctan2(S2,C2) # mean direction of sample 2 [2.9]
    rho2 = np.sum(np.cos(2*(theta2-thetaBar2)))/n1 # second trigonometric moment of sample 2 [2.27]
    delta2 = (1-rho2)/(2*RBar2**2) # circular dispersion of sample 2 [2.28]
    sigma2 = np.sqrt(delta2/n2) # circular standard error of sample 2 [4.21]
    
    if n1 < 25 or n2 < 25:
        warnings.warn('too few samples, not implemented')
        Y2 = np.nan
    elif np.max((delta1,delta2))/np.min((delta1,delta2)) <= 4: # method P
        Cp = n1*np.cos(thetaBar1) + n2*np.cos(thetaBar2) # [5.10]
        Sp = n1*np.sin(thetaBar1) + n2*np.sin(thetabar2) # [5.10]
        Rp = np.sqrt(Cp**2 + Sp**2)
        N = n1 + n2 # [5.2]
        delta0 = n1*delta1/N + n2*delta2/N # [5.12]
        Y2 = 2*(N - Rp)/delta0 # test statistic (reject hypothesis of common mean direction if Yr is too large) [5.13]
        
    elif np.max((delta1,delta2))/np.min((delta1,delta2)) > 4:
        Cm = np.cos(thetaBar1)/sigma1**2 + np.cos(thetaBar2)/sigma2**2 # [5.14]
        Sm = np.sin(thetaBar1)/sigma1**2 + np.sin(thetaBar2)/sigma2**2 # [5.14]
        Rm = np.sqrt(Cm**2 + Sm**2) # [5.15]
        Y2 = 2*(1/sigma1**2 + 1/sigma2**2 - Rm) # test statistic (reject hypothesis of common mean direction if Yr is too large) [5.16]
        
    return Y2
    """
    
def confidence_interval_for_mean_direction(theta,B=200,alpha=.05):
    """
    lower_bound, upper_bound = confidence_interval_for_mean_direction(theta,B=200,alpha=.05)
    
    theta = numpy array of data
    B = number of bootstrap samples (each of same length as theta)
    alpha = desired confidence level (can be a list, in which case confidence bounds are also lists)
    
    (lower_bound, upper_bound) is 100(1-alpha)% confidence interval
    
    Returns the confidence interval for mean direction of a sample of angles using the balanced resampling bootstrap method from Fisher, 1993.
    This method should be used when len(theta)<25  and len(theta) > 7 or 8 and we cannot assume symmetry about the mean
    Fisher S4.4.4 (p. 75), S8.3.2 (pp. 205-206), S8.3.5 (pp. 210-211)
    numbers in brackets correspond to equations in Fisher, 1993
    """
    def calculate_z(phi):
        """Algorithm 1 for calculating mean vector z [8.24]"""
        x = np.cos(phi)
        y = np.sin(phi)
        n = float(len(phi))
        z1 = np.sum(x/n)
        z2 = np.sum(y/n)
        z = np.matrix([[z1],[z2]])
        return z
        
    def calculate_u(phi,z1,z2):
        """Algorithm 1 for calculating covariance matrix u [8.25-6]"""
        x = np.cos(phi)
        y = np.sin(phi)
        n = float(len(phi))
        u11 = np.sum(((x-z1)**2)/n)
        u22 = np.sum(((y-z2)**2)/n)
        u12 = np.sum((x-z1)*(y-z2)/n)
        u = np.matrix([[u11, u12],[u12,u22]])
        return u

    def calculate_v(u):
        """Algorithm 2 for calculating square root of positive definite symmetric 2x2 matrix"""
        u11 = u[0,0]
        u12 = u[0,1]
        u22 = u[1,1]
        beta = (u11 - u22)/(2*u12) - np.sqrt(((u11 - u22)**2)/(4*u12**2) + 1) # [8.27]
        t1 = np.sqrt(u11*beta**2 + 2*beta*u12 + u22)/np.sqrt(1 + beta**2) # [8.28]
        t2 = np.sqrt(u11 - 2*beta*u12 + u22*beta**2)/np.sqrt(1 + beta**2) # [8.29]
        v11 = (t1*beta**2 + t2)/(1 + beta**2) # [8.30]
        v22 = (t1 + t2*beta**2)/(1 + beta**2) # [8.30]
        v12 = beta*(t1 - t2)/(1 + beta**2) # [8.31]
        v = np.matrix([[v11, v12],[v12,v22]])
        return v
        
    def calculate_w(u):
        """Algorithm 3 for calculating inverse of square root of positive definite 2x2 matrix"""
        u11 = u[0,0]
        u12 = u[0,1]
        u22 = u[1,1]
        beta = (u11 - u22)/(2*u12) - np.sqrt(((u11 - u22)**2)/(4*u12**2) + 1) # [8.32]
        t1 = np.sqrt(1 + beta**2)/np.sqrt(u11*beta**2 + 2*beta*u12 + u22) # [8.33]
        t2 = np.sqrt(1 + beta**2)/np.sqrt(u11 - 2*beta*u12 + u22*beta**2) # [8.34]
        w11 = (t1*beta**2 + t2)/(1 + beta**2) # [8.35]
        w22 = (t1 + t2*beta**2)/(1 + beta**2) # [8.35
        w12 = beta*(t1 - t2)/(1 + beta**2) # [8.36]
        w = np.matrix([[w11, w12],[w12,w22]])
        return w
        
    def calculate_muHat(z0,v0,zb,wb):
        """Algorithm 4 for estimating the mean direction"""
        cbsb = z0 + v0*wb*(zb - z0) # [8.37]
        cb = cbsb[0,0]
        sb = cbsb[1,0]
        Cb = cb/np.sqrt(cb**2 + sb**2) # [8.38]
        Sb = sb/np.sqrt(cb**2 + sb**2) # [8.38]
        muHat = np.arctan2(Sb,Cb) # [2.9]
        return muHat
    n = len(theta)
    thetaHat = np.arctan2(np.mean(np.sin(theta)),np.mean(np.cos(theta))) # mean direction of sample [2.9]
    z0 = calculate_z(theta)
    u0 = calculate_u(theta,z0[0,0],z0[1,0])
    v0 = calculate_v(u0)
    indices = np.random.permutation(np.mod(np.arange(n*B),n)) # generate complete set of bootstrap samples (balanced resampling) [8.3]
    muHat = np.zeros(B) # initialize
    for bootstrapIndex in range(B):
        thetaStar = theta[indices[n*bootstrapIndex:n*bootstrapIndex+n]].copy() # bootstrap sample
        zb = calculate_z(thetaStar)
        ub = calculate_u(thetaStar,zb[0,0],zb[1,0])
        wb = calculate_w(ub)
        muHatb = calculate_muHat(z0,v0,zb,wb)
        muHat[bootstrapIndex] = muHatb
    gamma = np.arctan2(np.sin(muHat-thetaHat), np.cos(muHat-thetaHat)) # [8.14]
    gamma.sort()
    if type(alpha) is list:
        lower_bound = np.zeros(len(alpha))
        upper_bound = np.zeros(len(alpha))
        for alphaIndex, alph in enumerate(alpha):
            L = int(0.5*B*alph + 0.5)
            m = B - L
            lower_bound[alphaIndex] = thetaHat + gamma[L] # subtract 1 because of python's zero indexing
            upper_bound[alphaIndex] = thetaHat + gamma[m-1] # subtract 1 because of python's zero indexing
    else:
        L = int(0.5*B*alpha + 0.5)
        m = B - L
        lower_bound = thetaHat + gamma[L] # subtract 1 because of python's zero indexing
        upper_bound = thetaHat + gamma[m-1] # subtract 1 because of python's zero indexing
    return lower_bound, upper_bound
    
def rayleigh_test(theta,mu0=None):
    """
    Rayleigh test of randomness against a unimodal alternative. Useful in detecting a single modal direction in a sample of vectors.
    
    RBar, P = rayleigh_test(theta,mu0=None)
    
    Parameters
    ----------
    theta : numpy array of data
    mu0 : alternative hypothesis for mean angle of data (optional)
    
    Returns
    -------
    P : significance probability
    RBar0 : test statistic
    
    See Fisher 1993 pp.69-71 or Mardia 1972 pp. 133-135
    """
    n = float(len(theta))
    C = np.sum(np.cos(theta)) # [2.7]
    S = np.sum(np.sin(theta)) # [2.7]
    R = np.sqrt(S**2 + C**2) # resultant length of sample [2.7]
    RBar = R/n # mean resultant length of sample [2.10]
    thetaBar = np.arctan2(S,C) # mean direction of sample [2.9]
    if mu0 is not None:
        RBar0 = RBar*np.cos(thetaBar - mu0) # test statistic [4.15]
        Z0 = np.sqrt(2*n)*RBar0
        #pz = norm.ppf(1-Z0) # appendix A1 x = scipy.stats.norm.ppf(1-alpha)
        pz = 1-norm.cdf(Z0) # appendix A1 alpha = 1-norm.cdf(x)
        fz = np.exp(-0.5*Z0**2)/np.sqrt(2*np.pi)
        P = 1- (1 - pz + fz*((3*Z0 - Z0**3)/(16*n) + (15*Z0 + 305*Z0**3 - 125*Z0**5 + 9*Z0**7)/(4608*n**2))) # [4.16]
    else:
        RBar0 = RBar
        Z = n*RBar**2
        if n < 50:
            P = np.exp(-Z)*(1 + (2*Z - Z**2)/(4*n) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4)/(288*n**2)) # [4.17]
        else:
            P = np.exp(-Z) # [4.18]
            
    return RBar0, P

