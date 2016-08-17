__author__ = 'jean'
import numpy as np

class heatingTank:

    def __init__(self,T0,steplen,Ts,h=0.1):

        self.Vtank = 1.13  # volume do tanque
        self.Ti = 15.0  # temperatura de entrada
        self.Q = 1100.0  # Calor
        self.ro = 1.0  # densidade
        self.cp = 4186.0  # coeficiente calorifico
        self.Vtube = 1.02
        self.T = T0
        self.Ttube = 0.99*T0
        self.steplen = steplen
        self.Ts = Ts
        self.h = h
        self.internalsteplen = int(round(steplen*self.Ts/self.h))
        self.Ttank_plot = np.empty(self.internalsteplen+int(np.floor(200*self.Ts/self.h)+1),dtype=np.float64)
        self.Ttube_plot = np.empty_like(self.Ttank_plot)
        self.step = -1
        self.internalstep = -1
        self.initialized = False
        self.delay = 0



    def StirredTank(self,T, q):
        Vtank =  self.Vtank  # volume do tanque
        Ti = self.Ti  # temperatura de entrada
        Q = self.Q  # Calor
        ro = self.ro  # densidade
        cp = self.cp  # coeficiente calorifico
        f = (q * (Ti - T) + Q / (ro * cp)) / Vtank  # Equacao diferencial em tempo continuo do balanco de energia do tanque
        return f


    def Tube(self,Ttube, Ttank):
        Tau = 29.0  # constante de tempo do tubo
        K = 0.99  # ganho do tubo
        f = K * (Ttank - Ttube) / Tau  # equacao de estados do tubo
        return f

    def RungeKutta4(self,step, f, x, u):  # runge kutta

        k1 = f(x, u)
        k2 = f(x + step * k1 / 2.0, u)
        k3 = f(x + step * k2 / 2.0, u)
        k4 = f(x + step * k3, u)

        xnext = x + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        return xnext

    def HeatingTank_iterate(self,q):

        t = np.arange(0,self.Ts,self.h)
        self.delay = int(np.floor(self.Vtube / (q * self.Ts)))
        for i in t:
            self.T = self.RungeKutta4(self.h,self.StirredTank,self.T,q)
            self.Ttube = self.RungeKutta4(self.h,self.Tube,self.Ttube,self.T)
            self.Ttank_plot[self.internalstep+1] = self.T
            self.Ttube_plot[self.internalstep+1] = self.Ttube
            self.internalstep += 1
        self.step += 1

    def get_Tout(self):
        if self.initialized is True:
            Tout = self.Ttube_plot[self.internalstep - self.delay]
            return Tout

        else:
            print "nao vai dar nao"
            return 0

    def initialize_Tank(self,q):
        for i in range(200):
            self.HeatingTank_iterate(q)

        self.initialized = True

