import numpy as np
import scipy

def cov_from_df(df):
    #Format of the cov matrix that the other functions use
    cov = df[['x0','x1','t','y']].cov().to_numpy()
    return cov

def analytical_cyt_estimate(cov):
    c_yt = (cov[2,3]*cov[0,1] - cov[1,3]*cov[0,2]) / (cov[2,2]*cov[0,1] - cov[1,2]*cov[0,2])
    return c_yt

def analytical_cyzsquared_estimate(cov):
    c_yz_squared = cov[0,2]*cov[0,1]*(cov[2,3]*cov[1,2] - cov[2,2]*cov[1,3])**2 / \
                    (cov[1,2]*(cov[2,2]*cov[0,1] - cov[0,2]*cov[1,2])**2)
    return np.abs(c_yz_squared)

def analytical_ctsquared_estimate(cov):
    c_t_squared = cov[0,2]*cov[1,2]/cov[0,1]
    return c_t_squared

def analytical_stsquared_estimate(cov):
    s_t_squared = cov[2,2] - analytical_ctsquared_estimate(cov)
    return s_t_squared

def analytical_ctcyz_estimate(cov):
    c_t_c_yz = cov[2,3] - analytical_cyt_estimate(cov)*analytical_stsquared_estimate(cov)-\
        analytical_ctsquared_estimate(cov)*analytical_cyt_estimate(cov)
    return c_t_c_yz

def analytical_sy_estimate(cov):
    c_yt = analytical_cyt_estimate(cov)
    c_t_c_yz = analytical_ctcyz_estimate(cov)
    s_t_squared = analytical_stsquared_estimate(cov)
    c_yz_squared = analytical_cyzsquared_estimate(cov)
    c_t_squared = analytical_ctsquared_estimate(cov)
    sy_squared = cov[3,3] - c_yz_squared - c_yt**2*s_t_squared - 2*c_yt*c_t_c_yz - c_t_squared*c_yt**2
    return np.sqrt(sy_squared)

def analytical_method_AID(est_c_yz, est_c_yt, est_s_y, c_yt, c_yz, s_y, c_t, s_t, n=100, lim=6):
    t_range = np.linspace(-lim,lim,n)
    y_range = np.linspace(-lim,lim,n)
    z_range = np.linspace(-lim,lim,n)
    z_len = 2*lim/n
    y_len = 2*lim/n
    t_len = 2*lim/n
    
    #First calculate the true P(t) function
    #P(t|z)
    pt_z_mean_true = c_t*z_range
    pt_z_std_true = s_t
    #P(t)
    pt_true = (scipy.stats.norm.pdf(z_range[:,None])*scipy.stats.norm.pdf(t_range[None,:], pt_z_mean_true[:,None],pt_z_std_true)).sum(axis=0)*z_len
    
    #P(y|do(t)) for the estimate
    py_zt_mean_est = est_c_yz*z_range[:,None] + est_c_yt*t_range[None,:]
    py_zt_std_est = est_s_y
    py_dot_est = np.zeros((n,n))
    for y_index in range(n):
        py_zt_est = scipy.stats.norm.pdf(y_range[y_index], py_zt_mean_est, py_zt_std_est)#shape (z_range, t_range)
        py_dot_est[:,y_index] = (py_zt_est * scipy.stats.norm.pdf(z_range[:,None])).sum(axis=0)*z_len

    #P(y|do(t)) for the true distribution
    py_zt_mean_true = c_yz*z_range[:,None] + c_yt*t_range[None,:]
    py_zt_std_true = s_y
    py_dot_true = np.zeros((n,n))
    for y_index in range(n):
        py_zt_true = scipy.stats.norm.pdf(y_range[y_index], py_zt_mean_true, py_zt_std_true)#shape (z_range, t_range)
        py_dot_true[:,y_index] = (py_zt_true * scipy.stats.norm.pdf(z_range[:,None])).sum(axis=0)*z_len
        
    #The average distances between P_est(y|do(t)) and P_true(y|do(t))
    causal_dist = np.abs(py_dot_est - py_dot_true).sum(axis=1)*y_len#shape (t_range)
    AID = (causal_dist * pt_true).sum()*t_len
    return AID, py_dot_est, py_dot_true, y_range, t_range, pt_true
