__author__ = 'jean'

import heatingtank_ as ht
import matplotlib.pyplot as plt
import numpy as np

from copy import *

from RNN import *

from scipy import signal

def normalize(q, mn, std):
    x = (q - mn) / std

    return x


def denormalize(x, mn, std):
    q = std * x + mn

    return q


def FindEquilibrium_q(T):
    Ti = 15.0  # temperatura de entrada
    Q = 1100.0  # Calor
    ro = 1.0  # densidade
    cp = 4186.0  # coeficiente calorifico

    invq = (T - Ti) * ro * cp / Q

    q = 1 / invq

    return q


def FindEquilibrium_T(q):
    Ti = 15.0  # temperatura de entrada
    Q = 1100.0  # Calor
    ro = 1.0  # densidade
    cp = 4186.0  # coeficiente calorifico

    return Ti + Q / (q * ro * cp)


def RandomStairRange(t, stair_step,
                min,max):  # vetor que gera uma sequencia aleatoria de valores, onde t eh um vetor de referencia, stair_step eh o numero de iteracoes que se mantem um valor
    # e values sao os valores que o degrau pode assumir
    Stair = np.empty_like(t)
    idx = 0
    for i in range(Stair.size):
        if i % stair_step == 0:
            value = min + (max-min)*np.random.random()  # escolhe o indice aleatoriamente a cada passo da escada.
        Stair[i] = value

    return Stair

def UpdateTimeWindow(A, x):  # Dado um array unidimensional A, elimina A[1] e acrescenta A[len(A)-1] = x
    B = A[1:]
    B = np.append(B, x)
    return B

def lowpass(x, u):
    w0 = 0.05
    f = w0*u - w0*x
    return f

def Euler(step, f, x, u):
    xnext = x + step*f(x, u)

    return xnext

def RedNoiseStep(x, t_step):
    white_noise = np.random.randn()
    xnext = Euler(t_step, lowpass, x, white_noise)
    return xnext

def RedNoise(numsteps,t_step):
    Noise = np.empty(numsteps)
    Noise[0] = np.random.randn()
    for i in range(1,Noise.size):
        Noise[i] = RedNoiseStep(Noise[i-1],t_step)

    return Noise


num_steps = 8000
First_step_end = 2000

q0 = 0.0167
T0 = FindEquilibrium_T(q0)  # condicao inicial

Ts = 4
delta =30
Tout_plot = np.empty(num_steps,dtype = np.float64)

Tanque = ht.heatingTank(0,num_steps+delta,Ts)

t = np.arange(0,num_steps*Ts,Ts,dtype = np.float64)
Q = 0.0167*np.sin(t)+ 0.02


Q_plot = np.empty_like(t)
qmax = 0.03
qmin = 0.005
qmn = 0.0250
qstd = 0.0175

# A Rede
neurons = 500
inputs = 2
outputs = 1
gama = 0.5
ro = 1
psi = 0.5
f_ir = 0.1
f_br = 1

CtrlNetTrainer = rNN(neurons, inputs, outputs, gama, ro, psi, f_ir, f_br)
CtrlNetController = deepcopy(CtrlNetTrainer)

Wro_plot = np.empty((len(t), neurons), dtype=float)

Control_plot = np.empty_like(t)
q_past = np.zeros_like(np.arange(delta))
T_past = np.empty_like(q_past)
Ttube_past = np.empty_like(q_past)
Tout_past = np.zeros_like(q_past)
Tfuturo = 0
Tref_plot = np.empty_like(t)
Training_Error_plot = np.empty_like(t)

stair_step_ctrl = 400
Tref_plot[:First_step_end] = 45 + 30 * RedNoise(First_step_end,Ts)
Tref_plot[First_step_end:] = RandomStairRange(Tref_plot[First_step_end:], stair_step_ctrl,30,60)

Tmn = np.mean(Tref_plot)

Tstd = np.std(Tref_plot)


Tanque.initialize_Tank(q0)

q = q0
for i in range(delta):


    Tanque.HeatingTank_iterate(q)
    Tout = Tanque.get_Tout()
    Tout_past[i] = Tout
    q_past[i] = q

for i in range(num_steps):

    #controle


    #white_noise = np.random.randn()
    CtrlNetTrainer.Update([normalize(Tout_past[0], Tmn, Tstd), normalize(Tout, Tmn, Tstd)])
    # CtrlNetTrainer.Update([(Tout_past[0]-Tmean)/Tstd,((Tout + white_noise)-Tmean)/Tstd])
    u_ctrlpast = normalize(q_past[0], qmn, qstd)
    CtrlNetTrainer.Train(u_ctrlpast)
    trainingerror = CtrlNetTrainer.trainingError(u_ctrlpast)**2
    Training_Error_plot[i] = trainingerror
    


    # etapa de teste
    CtrlNetController.CopyWeights(CtrlNetTrainer)
    uctrl = CtrlNetController.Update([normalize(Tout, Tmn, Tstd), normalize(Tref_plot[i], Tmn, Tstd)])
    # uctrl = CtrlNetController.Update([((Tout + white_noise)-Tmean)/Tstd,(Tfuturo-Tmean)/Tstd])
    q = denormalize(uctrl, qmn, qstd)
    control = q

    if q > qmax:
        q = qmax

    if q < qmin:
        q = qmin

    Q_plot[i] = q

    Wro_plot[i] = CtrlNetController.getWro(0)




    #fim controle

    Tanque.HeatingTank_iterate(q)
    Tout = Tanque.get_Tout()
    Tout_plot[i] = Tout
    q_past = UpdateTimeWindow(q_past, q)
    Tout_past = UpdateTimeWindow(Tout_past, Tout)



Wro_trueplot = np.mean(np.diff(Wro_plot)**2,1)

print np.mean(Training_Error_plot[2000:]),"Erro medio de treinamento da Rede"
print np.std(Training_Error_plot[2000:]),"Desvio padrao do erro medio"

#gravar o erro quadratico do controlador, valor esperado de temperatura vs valor real.

ErroControlador = (Tout_plot[2000+delta:] - Tref_plot[2000:-delta])**2

print np.mean(ErroControlador),"media do erro quadratico do controle"
print np.std(ErroControlador),"desvio do erro do controlador"
#plt.show()

f, sub = plt.subplots(3, sharex=True)



# p1 = sub[0].plot(t,T_plot,label = 'Temperatura do tanque')



p2 = sub[1].plot(Q_plot, label='Vazao Inlet')

# p3 = sub[0].plot(t,Ttube_plot,label = 'Temperatura Tubo')

p4 = sub[0].plot(Tout_plot, label='Temperatura Saida')

p5 = sub[0].plot(Tref_plot, label='Referencia')

p6 = sub[2].plot(Wro_trueplot, label='mean(diff(Wro)^2)')


sub[0].legend()

sub[1].legend()

sub[2].legend()

# plt.show(p1)

plt.show(p2)

 #plt.show(p3)

plt.show(p4)
