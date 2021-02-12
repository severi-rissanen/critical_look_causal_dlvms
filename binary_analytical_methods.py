import numpy as np

def get_p2_func(df):
    def p2(var,val):
        return (df[var]==val).prod(1).sum()/len(df)
    return p2

def analytical_miao_pydot(df):
    p2 = get_p2_func(df)
    py_x0_te1 = np.array([[p2(["x0","t","y"],(1,1,1))/p2(["x0","t"],(1,1)), p2(["x0","t","y"],(0,1,1))/p2(["x0","t"],(0,1))]])
    px1_x0_te1 = np.array([[p2(["x1","x0","t"],(1,1,1))/p2(["x0","t"],(1,1)), p2(["x1","x0","t"],(1,0,1))/p2(["x0","t"],(0,1))],
                          [p2(["x1","x0","t"],(0,1,1))/p2(["x0","t"],(1,1)),p2(["x1","x0","t"],(0,0,1))/p2(["x0","t"],(0,1))]])
    px1 = np.array([[p2(["x1"],1)],[p2(["x1"],0)]])

    py_x0_te0 = np.array([[p2(["x0","t","y"],(1,0,1))/p2(["x0","t"],(1,0)), p2(["x0","t","y"],(0,0,1))/p2(["x0","t"],(0,0))]])
    px1_x0_te0 = np.array([[p2(["x1","x0","t"],(1,1,0))/p2(["x0","t"],(1,0)), p2(["x1","x0","t"],(1,0,0))/p2(["x0","t"],(0,0))],
                          [p2(["x1","x0","t"],(0,1,0))/p2(["x0","t"],(1,0)),p2(["x1","x0","t"],(0,0,0))/p2(["x0","t"],(0,0))]])

    py_dot1 = py_x0_te1.dot(np.linalg.inv(px1_x0_te1)).dot(px1)
    py_dot0 = py_x0_te0.dot(np.linalg.inv(px1_x0_te0)).dot(px1)
    return py_dot1, py_dot0

def p_x0_z_from_dist(p,y_label="y"):
    #Returns the probability of x0=0 given z=1 and z=0. The assignment to 1 and 0 is arbitrary
    t = 1
    y = 1#the valus of these don't actually matter for the final result
    P_x1_x0 = np.array([[1,p(["x0", "t"],(0,t))/p(["t"],t)],
                        [p(["x1","t"],(0,t))/p(["t"],t), p(["x1","x0","t"],(0,0,t))/p(["t"],t)]])
    Q_x1_x0 = np.array([[p([y_label,"t"],(y,t))/p(["t"],t), p([y_label,"x0","t"],(y,0,t))/p(["t"],t)],
                        [p([y_label,"x1","t"],(y,0,t))/p(["t"],t), p([y_label,"x1","x0","t"],(y,0,0,t))/p(["t"],t)]])
    e,v = np.linalg.eig(np.linalg.inv(P_x1_x0).dot(Q_x1_x0))
    e = sorted(list(enumerate(list(e))), key=lambda x: -x[1])#Just sorting the v for consistent results, probably doesn't matter
    sorted_v = np.concatenate([v[:,e[0][0]][:,None],v[:,e[1][0]][:,None]],1)
    inv = np.linalg.inv(v)
    return inv[:,1]*1/inv[:,0]

def everything_from_dist(df,y_label="y"):
    p = get_p2_func(df)
    p_x0_0_z = p_x0_z_from_dist(p,y_label)
    p_x0_z = 1-p_x0_0_z#This is ready as is
    
    #Next figure out p(z), p(t|z),p(y|z,t)
    p_y_tz = np.zeros((2,2))#p(y=1|t,z), dims (t,z)
    p_t_z = np.zeros((2,))#p(y=1|z)
    p_x1_z = np.zeros((2,))
    p_z = np.zeros(1)
    
    for t in range(2):
        for y in range(2):
            V_ty_x0 = np.array([[p([y_label,"x0","t"],(y,0,t))/p(["t"],t)],[p([y_label,"x0","t"],(y,1,t))/p(["t"],t)]])
            V_t_x0 = np.array([[p(["x0","t"],(0,t))/p(["t"],t)],[p(["x0","t"],(1,t))/p(["t"],t)]])
            V_x0 = np.array([[p(["x0"],0)],[p(["x0"],1)]])

            M_x0_z = np.array([p_x0_0_z, p_x0_z])

            V_z = np.linalg.inv(M_x0_z).dot(V_x0)#p(z)
            V_t_z = np.linalg.inv(M_x0_z).dot(V_t_x0)
            V_ty_z = np.linalg.inv(M_x0_z).dot(V_ty_x0)
            
            p_z[0] = V_z[1,0]#Prob that "z=1"
            p_y_tz[t,:] = (V_ty_z/V_t_z).squeeze() #In order z=0 and z=1
            if t==1:
                p_t_z[:] = (V_t_z*p(["t"],1)/V_z).squeeze()
    x1 = 1
    V_x1_x0 = np.array([[p(["x0","x1"],(0,x1))/p(["x1"],x1)],[p(["x0","x1"],(1,x1))/p(["x1"],x1)]])
    M_x0_z = np.array([p_x0_0_z, p_x0_z])
    V_x1_z = np.linalg.inv(M_x0_z).dot(V_x1_x0)#p(z|x1=1) (order z=0, z=1)
    p_x1_z[:] = (V_x1_z*p(["x1"],1)/V_z).squeeze()#Reusing V_z
    
    return p_y_tz, p_t_z, p_z, p_x0_z, p_x1_z