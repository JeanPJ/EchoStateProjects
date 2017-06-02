from pymodelica import compile_fmu
from pyfmi import load_fmu
from matplotlib.pyplot import *
from functools import partial

def step_fun(u,t):
    u = np.atleast_2d(u)
    return u*np.ones_like(t)

class modelica_model:

    def __init__(self,Ts,inputs = '',model_name='',file_name='',
                 maxstepsize = 0.01):
        fmu_file = compile_fmu(model_name,file_name)
        self.model = load_fmu(fmu_file)
        self.opts = self.model.simulate_options()
        self.opts['initialize'] = False
        self.opts['CVode_options']['verbosity'] = 50
        self.opts['CVode_options']['maxh'] = maxstepsize
        self.opts['CVode_options']['rtol'] = 1e-7
        self.Ts = Ts
        self.inputs = inputs
        self.First = True
        self.result = {}

    def external_timestep(self,u):
        input_object = (self.inputs, partial(step_fun,u))
        if self.First:
            self.result = self.model.simulate(
                final_time=self.Ts, input=input_object)
            self.First = False
        else:
            self.result = self.model.simulate(
                start_time = self.result['time'][-1],
                final_time=self.result['time'][-1]+self.Ts,
                input=input_object,options = self.opts)




