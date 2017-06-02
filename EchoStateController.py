
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from copy import *
import time
from RNN_cuda import *
class savePoint:

    def __init__(self,filename):
        self.filename = filename

        self.matFile = {}
    def save(self,a): #a = array
        self.matFile['a'] = a
        io.savemat(self.filename,self.matFile)

    def load(self): #return loaded array
        io.loadmat(self.filename, self.matFile)
        return self.matFile['a'].T


def feature_scaling(x,xmax,xmin,normalmode=True):
    if normalmode:
        y = (x - xmin)/(xmax - xmin)
    else:
        y = x
    return y

def feature_descaling(x,xmax,xmin,normalmode=True):
    if normalmode:
        y = x*(xmax-xmin) + xmin
    else:
        y = x

    return y

def normalize(q, mn, std,normalmode=True):
    if normalmode:
        x = (q - mn) / std
    else:
        x = q

    return x


def denormalize(x, mn, std,normalmode=True):
    if normalmode:
        q = std * x + mn
    else:
        q = x

    return q

#calculate equilibrium point of tank
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

# Random Repeating Sequence Stair


def RandomStairRange(t, stair_step,
                min,max):
    stair = np.empty_like(t)
    idx = 0
    for i in range(stair.size):
        if i % stair_step == 0:
            value = min + (max-min)*np.random.random()
        stair[i] = value

    return stair

#Used for the dislocation of
# timewindow to compute y[k-delta]
def UpdateTimeWindow(A, x):
    B = A[1:]
    X = np.atleast_2d(x)
    B = np.append(B, X,axis=0)

    return B

#lowepass filter
def lowpass(x, u):
    w0 = 70
    f = w0*u - w0*x
    return f

#euler integration method
def Euler(step, f, x, u):
    xnext = x + step*f(x, u)

    return xnext

#iteration of a red noise
def RedNoiseStep(x, t_step):
    white_noise = np.random.randn()
    xnext = Euler(t_step, lowpass, x, white_noise)
    return xnext

#a red noise
def RedNoise(numsteps,t_step):
    Noise = np.empty([numsteps,1],dtype = np.float64)
    Noise[0] = np.random.randn()
    for i in range(1,Noise.size):
        Noise[i] = RedNoiseStep(Noise[i-1],t_step)

    return Noise

#a PRBS(Pseudorandin Binary Signal)

def PRBS(min,max,num_steps,minimum_step):
    PRBS_sig = np.empty(num_steps)
    for i in range(num_steps):

        if i % minimum_step  == 0:
            p_val = np.random.rand()
            if p_val > 0.5:
                p_val = 1.0
            if p_val < 0.5:
                p_val = 0.0

        PRBS_sig[i] = min + p_val*(max-min)

    return PRBS_sig

def change_or_not(x,min_val,max_val):
    y = 0
    p_change = np.random.rand()
    if p_change < 0.5:
        y = x
    else:
        y = min_val + (max_val - min_val)*np.random.rand()
    return y

def RFRAS(min,max,num_steps,minimum_step):
    RFRAS_sig = np.empty(num_steps)
    val = min + (max - min)*np.random.rand()
    for i in range(num_steps):

        if i % minimum_step  == 0:
            val = change_or_not(val,min,max)


        RFRAS_sig[i] = val

    return RFRAS_sig


def principal_component(X,k):

    mean_X = np.mean(X)

    std_X = np.std(X)

    X_norm = (X - mean_X)/std_X
    Sigma = np.dot(X_norm.T,X_norm)
    #incomplete

class ESC: #Echo State Controller

    def __init__(self,num_steps,first_step_end,
                 x0,y0,Ts,neurons,gama,ro,psi,f_ir,
                 f_br,delta,xmax,xmin,ymax,ymin,
                 normalmode = True,
                 ref_filename = 'referencia',
                 loadmode_initial=False,
                 savemode_initial = False,
                 initial_filename = "initial",main = 0,training_method = "RLS",alpha = 10,noise_amplitude = 0):
        self.save_ref = savePoint(ref_filename)
        # total number of simulation steps
        self.num_steps = num_steps
        # number of steps in the exploration phase
        self.first_step_end = first_step_end
        # initical input values
        self.x0 = np.array(x0,dtype = float)
        # initial plant output balues.
        self.y0 = np.array(y0,dtype = float)

        self.x = np.array(x0,dtype = float)

        self.y = np.array(y0,dtype = float)

        self.Ts = Ts #sample time.
        # taking into account the number of outputs and desired outputs
        inputs = self.y0.size + 1

        outputs = self.x0.size

        self.CtrlNetTrainer = rNN(neurons, inputs,
                                  outputs, gama,
                                  ro, psi, f_ir, f_br,
                                  load_initial = loadmode_initial,
                                  save_initial = savemode_initial,
                                  initial_filename = initial_filename,alfa = alpha,noise_amplitude = noise_amplitude)

        self.CtrlNetController = rNN(neurons, inputs,
                                  outputs, gama,
                                  ro, psi, f_ir, f_br,
                                  load_initial = loadmode_initial,
                                  save_initial = savemode_initial,
                                  initial_filename = initial_filename,alfa = alpha,noise_amplitude = noise_amplitude)

        self.CtrlNetTrainer.CopyWeights(self.CtrlNetController)

        self.Ymean = 0 #those are defined by the refference signal

        self.Ystd = 0 #those are defined by the refference signal

        self.delta = delta

        self.t = np.arange(0, num_steps * Ts, Ts, dtype=float)

        self.y_plot = np.empty([num_steps,self.y0.size], dtype=float)
        self.forget_plot = np.empty(num_steps)
        self.Control_plot = np.empty([self.t.size,outputs],dtype = float)
        self.x_past = np.empty([delta,outputs],dtype = float)
        self.y_past = np.empty([delta,self.y0.size],dtype = float)
        self.Yref_plot = np.empty([num_steps,1],dtype = float)
        self.echostate_outputplot = np.empty_like(self.Control_plot,dtype = float)
        self.echostate_trainplot = np.empty_like(self.Control_plot,dtype = float)


        self.reference_passed = False

        self.system_initialized = False

        self.xmax = xmax

        self.xmin = xmin

        self.Training_Error_plot = np.empty_like(self.Control_plot,dtype = float)

        self.normalmode = normalmode

        self.J_error = 0

        self.J_control = 0

        self.J = 0
        self.ymax = ymax
        self.ymin = ymin
        self.trainingerror=0

        self.main = main

        self.reservoir_norm_train = np.empty(num_steps)
        self.reservoir_norm_control= np.empty(num_steps)
        self.training_method = training_method

    # Obtains reference from external source.
    def external_reference(self,ref):

        if ref.shape == self.Yref_plot.shape:
            self.Yref_plot = ref
            self.reference_passed = True
            self.Ymean = np.mean(self.Yref_plot,axis=0)
            self.Ystd = np.std(self.Yref_plot,axis=0)
        else:
            raise ValueError("wut, r u casul? wrong input vector dimension")

    def exploration_rednoise(self,stair_step_ctrl,
                             mean,amplitude,ymin,ymax):
        for i in range(self.Yref_plot[0].size):
            self.Yref_plot[:self.first_step_end] = \
                mean + \
                amplitude * \
                RedNoise(self.first_step_end, 0.01)

            self.Yref_plot[self.first_step_end:] = RandomStairRange\
                (self.Yref_plot[self.first_step_end:],
                 stair_step_ctrl,
                 ymin[i], ymax[i])

        self.reference_passed = True
        self.Ymean = np.mean(self.Yref_plot,axis=0)
        self.Ystd = np.std(self.Yref_plot,axis=0)

    def exploration_prbs(self,stair_step_ctrl,ymin,ymax,min_step):

        for i in range(self.Yref_plot[0].size):
            self.Yref_plot[:self.first_step_end,i] = \
                PRBS(ymin[i],ymax[i],self.first_step_end,min_step)
            self.Yref_plot[self.first_step_end:,i] = \
                RandomStairRange(self.Yref_plot[self.first_step_end:,i],
                             stair_step_ctrl,
                             ymin[i], ymax[i])

        self.reference_passed = True
        self.Ymean = np.mean(self.Yref_plot,axis=0)
        self.Ystd = np.std(self.Yref_plot,axis=0)

    def exploration_rfras(self,stair_step_ctrl,ymin,ymax,min_step):

        for i in range(self.Yref_plot[0].size):
            self.Yref_plot[:self.first_step_end,i] = \
                RFRAS(ymin[i],ymax[i],self.first_step_end,min_step)
            self.Yref_plot[self.first_step_end:,i] = \
                RandomStairRange(self.Yref_plot[self.first_step_end:,i],
                             stair_step_ctrl,
                             ymin[i], ymax[i])

        self.reference_passed = True
        self.Ymean = np.mean(self.Yref_plot,axis=0)
        self.Ystd = np.std(self.Yref_plot,axis=0)

    def load_previous_ref(self):
        self.Yref_plot =np.atleast_2d(self.save_ref.load()).T
        self.reference_passed = True
        self.Ymean = np.mean(self.Yref_plot,axis=0)
        self.Ystd = np.std(self.Yref_plot,axis=0)

    def load_previous_reservoir(self,filename):

        self.CtrlNetTrainer.LoadReservoir(filename)
        self.CtrlNetController.LoadReservoir(filename)

    def save_previous_ref(self):

        self.save_ref.save(self.Yref_plot)

    def save_previous_reservoir(self,filename):

        self.CtrlNetTrainer.SaveReservoir(filename)
        self.CtrlNetController.SaveReservoir(filename)

    def initialize_plant(self,x,plant):
        RFRAS_in = RFRAS(0, 1, self.delta,5)
        for i in range(self.delta):
            x_line = x
            y = plant(x_line)
            self.x_past[i] = x
            self.y_past[i] = y





    def run(self,plant): #plant is a I?O model of the system.
        # control
        start_time = time.time()
        for i in range(self.num_steps):


            self.y = np.atleast_1d(self.y)

            y_past_normal = feature_scaling(self.y_past[0], self.ymax, self.ymin,
                                  self.normalmode)
            y_present_normal = np.atleast_1d(feature_scaling(self.y[self.main], self.ymax[self.main],self.ymin[self.main],self.
                                 normalmode))
            Y = np.concatenate((y_past_normal,y_present_normal))
            u_train =self.CtrlNetTrainer.Update(Y)
            self.echostate_trainplot[i] = u_train.T
            u_ctrlpast = feature_scaling\
                (self.x_past[0], self.xmax, self.xmin,self.normalmode)

            if self.training_method == "RLS":
                self.CtrlNetTrainer.Train(u_ctrlpast.T)
                self.forget_plot[i] = self.CtrlNetTrainer.get_forgetingfactor()

            if self.training_method == "LMS":
                self.CtrlNetTrainer.trainLMS(u_ctrlpast)
            self.trainingerror = \
                (self.CtrlNetTrainer.trainingError(u_ctrlpast) ** 2 + i*self.trainingerror)/(i+1)
            self.Training_Error_plot[i] = self.trainingerror.T
            #self.reservoir_norm_train[i] = np.linalg.norm(self.CtrlNetController.a)


        # desicion of the control action:

            self.CtrlNetController.CopyWeights(self.CtrlNetTrainer)
            y_present_normal_ctrl = feature_scaling(self.y, self.ymax,
                            self.ymin,self.normalmode)
            y_future_normal = feature_scaling(self.Yref_plot[i],self.ymax[self.main],
                           self.ymin[self.main],self.normalmode)

            uctrl = self.CtrlNetController.Update\
                (np.concatenate((y_present_normal_ctrl,
                 y_future_normal)))
            self.echostate_outputplot[i] = uctrl.T

            #self.reservoir_norm_control[i] = np.linalg.norm(self.CtrlNetTrainer.a)
            self.x = feature_descaling\
                (uctrl.reshape(uctrl.size), self.xmax, self.xmin,self.normalmode)
            for j in range(self.x0.size):
                if self.x[j] > self.xmax[j]:
                    self.x[j] = self.xmax[j]

                if self.x[j] < self.xmin[j]:
                    self.x[j] = self.xmin[j]


          #  if i > self.first_step_end:
          #      self.J_control += \
         #           (self.x/(self.xmax-self.xmin)) ** 2

            if i % 1000 == 0:
                print self.J_control,\
                    "cost of the control action at timestep ",i
                print "execution time until now is",(time.time() - start_time)

            self.Control_plot[i] = self.x.T



            # fim controle
            self.y = plant(self.x)
            self.y_plot[i] = self.y
            self.x_past = UpdateTimeWindow(self.x_past, self.x)
            self.y_past = UpdateTimeWindow(self.y_past, self.y)


            if i > self.delta + self.first_step_end:
                self.J_error += \
                    ((self.y_plot[i][self.main] -
                      self.Yref_plot[i-self.delta])
                     /(self.ymax[self.main]-self.ymin[self.main]))**2

            if i % 1000 == 0:
                print self.J_error, \
                    "cost of the error at timestep ", i













