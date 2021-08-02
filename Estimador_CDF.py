import numpy as np
from math import *
from scipy.special import factorial
from func_min import func_min


def Est_Param_ClassA_CDF(env_data,Vi,num_iter,func_eval):
    """
    -------------------------------------------------------------------------------------------------------------------------
    --             Función con la cual se obtiene una estimación de los parámetros de ruido                                --                                               
    --             de Middleton de Clase A a partir de una funcion a minimizar y un vector                                 -- 
    --                         con los puntos iniciales para comenzar la iteración:                                        --
    --       A_est,gauss_noise_est,r_est = Est_Param_ClassA_propio(env_data,vec_ini,num_iter,func_eval,Sigmag2,r_org)      --
    --                                                                                                                     --
    -- Entradas:                                                                                                           --
    --         env_data: Valores de ruido de Middleton de Clase A de envolvente utilizados para la estimación              --
    --          vec_ini: Es el vector fila inicial con el cual se formará el Simplex Inicial                               --
    --         num_iter: Número de iteraciones a realizar por el método hasta alcanzar el mínimo                           --
    --        func_eval: Valor mínimo de la evaluación de la función del punto óptimo para cortar la iteración             --
    --          Sigmag2: Potencia de Ruido Gaussiano o de fondo en W.                                                      --
    --            r_org: Parámetro r(Gamma) original para desnormalizar los datos de entrada env_data                      --
    -------------------------------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------------------------------                                                                                                                     
    --                     La iteración corta si el numero de iteraciones supera num_iter                                  -- 
    --                      o si feval(func,[A_est,gauss_noise_est,r_est]) < a func_eval                                   --
    -------------------------------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------------------------------
    -- Salidas:                                                                                                            --
    --                   A_est: Es el valor del parámetro A del ruido de Middleton de Clase A estimado                     --
    --         gauss_noise_est: Es la potencia(en mW) de ruido gaussiano o de fondo estimado                               --
    --                   r_est: Es el valor del parámetro r(Gamma) del ruido de Middleton de Clase A estimado              --
    --                                                                                                                     --
    ------------------ Esta función utiliza el método de Nelder-Mead para minimizar la función objetivo ---------------------
    """

    #-- Calculo de la potencia de ruido --
    potencia = env_data**2/2
    #----------------------------------------------

    #-- Potencia en dBW(respecto a 1W) --
    potencia_log = 10*np.log10(potencia)
    #------------------------------------

    #-- Generación del vector de edges para la función histcounts --
    N = len(env_data)
    minimo = min(potencia_log)
    maximo = max(potencia_log)
    paso = (maximo - minimo)/(N-1)
    #---------------------------------------------------------------

    #----- Cálculo del histograma (CDF) de potencia_log -----
    histo, bin_edge = np.histogram(potencia_log,N+1,(minimo-2*paso,maximo+2*paso),density = True)
    #--------------------------------------------------------

    X_cdf = bin_edge[:N+1]
    CUM = np.cumsum(histo)
    CDF = CUM/max(CUM)
    CDF_inv = 1 - CDF

    V = np.zeros((4,3))
    Vi[1] = 1/Vi[1]
    V = np.array([(Vi),(Vi[0]*1.05,Vi[1],Vi[2]),(Vi[0],Vi[1]*1.05,Vi[2]),(Vi[0],Vi[1],Vi[2]*1.05)])

    size = np.shape(V)
    Y = np.zeros(size[0])


    for i in range(size[0]):
        Y[i] = func_min(X_cdf,CDF_inv,V[i])

    Aux = np.max(Y)
    Vord = np.zeros(size)
    Yord = np.zeros(size[0])

    for i in range(size[0]):
        pos_minYaux = np.where(Y == np.min(Y))
        pos_minY = pos_minYaux[0]
        Vord[i,:] = V[pos_minY[0],:]
        Yord[i] = Y[pos_minY[0]]
        Y[pos_minY[0]] = Aux + 1

    iter = 0

    n = 0

    while (iter < num_iter and Yord[0] > func_eval and n<10):
        #print('Iteracion:',iter,'Valor Yord:',Yord[0])
        M = np.array([(sum(Vord[0:3,0])/3),(sum(Vord[0:3,1])/3),(sum(Vord[0:3,2])/3)])
        R = 2*M - Vord[3,:]
        fr = func_min(X_cdf,CDF_inv,R)
        
        if (fr < Yord[1]):
            # CASO 1
            if (Yord[0] < fr):
                Vord[3,:] = R
            else:
                E = 2*R - M
                fe = func_min(X_cdf,CDF_inv,E)
                if (fe < Yord[0]):
                    Vord[3,:] = E
                else:
                    Vord[3,:] = R
        else:
            # CASO 2
            if (fr < Yord[3]):
                Vord[3,:] = R
            else:
                C = (Vord[3,:] + M)/2
                fc = func_min(X_cdf,CDF_inv,C)
                C2 = (M + R)/2
                fc2 = func_min(X_cdf,CDF_inv,C2)
                if (fc2 < fc):
                    C = C2
                    fc = fc2
                if (fc < Yord[3]):
                    Vord[3,:] = C
                else:
                    S = (Vord[0,:] + Vord[3,:])/2
                    Vord[3,:] = S
                    J = (Vord[0,:] + Vord[1,:])/2
                    Vord[1,:] = J
                    H = (Vord[0,:] + Vord[2,:])/2
                    Vord[2,:] = H
        
        for i in range(size[0]):
            Y[i] = func_min(X_cdf,CDF_inv,Vord[i,:])

        Aux = np.max(Y)
        
        for i in range(size[0]):
            posMin = np.argmin(Y)
            V[i] = Vord[posMin]
            Yord[i] = Y[posMin]
            Y[posMin] = Aux + 1

        Vord[:,:] = V[:,:]
        iter += 1
        
        if ((Yord[1] - Yord[0]) < (1E-18)):
            n = n - 1
        else:
            n = 0

        A = V[0,0]
        Sigmag2 = 1/V[0,1]
        r = V[0,2]

    return A,Sigmag2,r,iter