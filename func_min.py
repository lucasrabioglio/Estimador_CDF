import numpy as np

def func_min(X_CDF,CDF_inv,V):
    # Funcion a minimizar

    #----------- Puntos iniciales ---------------
    A = V[0]
    r = V[2]
    #--------------------------------------------
    
    #------ Terminos de la sumatoria ---------
    L0 = V[1]
    L1 = L0/(1 + 1/(A*r))
    L2 = L0/(1 + 2/(A*r))
    L3 = L0/(1 + 3/(A*r))
    L4 = L0/(1 + 4/(A*r))
    L5 = L0/(1 + 5/(A*r))
    L6 = L0/(1 + 6/(A*r))
    L7 = L0/(1 + 7/(A*r))
    #------------------------------------------

    x = X_CDF/10
    C = np.exp(-A)

    #------- Funcion de Costo ---------------------------
    y = np.sum((CDF_inv - C*np.exp(-L0*(10**x)) - A*C*np.exp(-L1*(10**x)) - (A**2/2)*C*np.exp(-L2*(10**x)) - (A**3/6)*C*np.exp(-L3*(10**x)) - (A**4/24)*C*np.exp(-L4*(10**x)) - (A**5/120)*C*np.exp(-L5*(10**x)) - (A**6/720)*C*np.exp(-L6*(10**x)) - (A**7/5040)*C*np.exp(-L7*(10**x)))**2)
    #----------------------------------------------------

    return y
    
