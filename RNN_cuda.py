__author__ = 'jean'
import numpy as np
import scipy.io as io
import cudamat as cm


def Sparcity(M,psi):
    N = cm.empty(M.shape)
    N.assign(np.random.choice([0,1],M.size,p =[psi,1-psi]))
    T = np.empty(M.shape)
    N.mult(M,target = T)

    return T


class rNN:

    def __init__(self,neu,n_in,n_out,
                 gama=0.5,ro=1,psi=0.5,in_scale=0.1,
                 bias_scale=0.5,alfa=10,forget = 1,
                 initial_filename="initial",
                 load_initial = False,save_initial = False,noise_amplitude = 0):
        #All matrixes are initialized under the normal distribution.
        cm.cublas_init()
        print "initializing reservoir"
        print n_in,"Number of inputs"
        self.neu = neu
        self.n_in = n_in
        self.n_out = n_out
        self.noise_amplitude = noise_amplitude

        # Reservoir Weight matrix.
        print "initializing reservoir matrix"
        self.Wrr0 = cm.CUDAMatrix(np.random.normal(0,1,[neu,neu],))
        print "initializing input matrix"
        # input-reservoir weight matrix
        self.Wir0 = cm.CUDAMatrix(np.random.normal(0,1,[neu,n_in]))
        # bias-reservoir weight matrix
        print "initializing bias matrix"
        self.Wbr0 = cm.CUDAMatrix(np.random.normal(0,1,[neu,1]))

        self.Wrr = cm.empty(self.Wrr0.shape)
        self.Wbr = cm.empty(self.Wbr0.shape)
        self.Wir = cm.empty(self.Wir0.shape)

        #self.Wbo = np.random.normal(0,1,[n_out,1])
        # reservoir-output weight matrix
        print "initializing Wro"
        self.Wro = cm.CUDAMatrix(np.random.normal(0,1,[n_out,neu]))

        self.leakrate = gama #the network's leak rate
        self.ro = ro #the network's desired spectral radius
        self.psi = psi #the network's sparcity, in 0 to 1 notation
        self.in_scale = in_scale #the scaling of Wir.
        self.bias_scale = bias_scale #the scaling of Wbr

        # learning rate of the Recursive Least Squares Algorithm
        self.alfa = alfa
        # forget factor of the RLS Algorithm
        self.forget = forget

        #self.a = np.random.normal(0, 1, [neu, 1])
        self.a = cm.CUDAMatrix(np.zeros([neu, 1]))
        #save if save is enabled
        if save_initial:
            self.save_initial_fun(initial_filename)

        #load if load is enabled
        if load_initial:
            self.LoedInitial(initial_filename)

        # the probability of a memeber of the Matrix Wrr being zero is psi.
        print "define sparseness"
        if psi > 0:
            self.Wrr = Sparcity(self.Wrr0,self.psi)
        else:
            self.Wrr.assign(self.Wrr0)
        #forcing Wrr to have ro as the maximum eigenvalue
        print "calculating eigenvalues"
        eigs = np.linalg.eigvals(self.Wrr.asarray())
        print "finding maximum eigenvalue"
        radius = np.abs(np.max(eigs))
        #normalize matrix
        print "normalize reservoir"
        self.Wrr.divide(np.asscalar(radius))
        #set its spectral radius to rho
        self.Wrr.mult(ro)

        #scale tbe matrices
        self.Wbr0.mult(bias_scale,target = self.Wbr)
        self.Wir0.mult(in_scale, target=self.Wir)



        #initial conditions variable forget factor.
        self.sigma_e = 0.001
        self.sigma_q = 0.001
        self.sigma_v = 0.001
        self.K_a = 6.0
        self.K_b = 3.0*self.K_a

        #covariance matrix
        self.P = cm.CUDAMatrix(np.eye(neu)/alfa)
        print "Reservoir initialization Done"


    def getWro(self,n=0): #retorna a coluna n do vetor Wro

        return self.Wro[n]

    def get_forgetingfactor(self):

        return self.forget

    def trainingError(self,ref):
        ref = np.atleast_2d(ref)
        Ref = cm.CUDAMatrix(ref)
        if self.n_out > 1:
            Ref = Ref.reshape([self.n_out,1])

        e = cm.dot(self.Wro,self.a).subtract(Ref).asarray()
        return e

    def Train(self,ref):
        #ref e o vetor de todas as sa
        # idas desejados no dado instante de tempo.
        #calcular o vetor de erros
        e = self.trainingError(ref)
        max_lambda = 0.9999
        min_lambda = 0.999
        #regularization
        mu = 1e-8
        #holder = cm.CUDAMatrix(self.P.asarray())

        for saida in range(self.n_out):
            #regularization step
            #cm.dot(self.P,self.P,target = holder)
            #holder.mult(mu)
            #self.P.subtract(holder)
            #end regularization step
            self.sigma_e = (1.0 - 1.0/(self.K_a * self.neu)) * self.sigma_e + (1.0 - (1.0 - 1.0/(self.K_a * self.neu))) * e[saida]**2
            self.sigma_q = (cm.pow(cm.dot(cm.dot(self.a.T,self.P),self.a),2).mult((1.0 - (1.0 - 1.0/(self.K_a * self.neu)))).add((1.0 - 1.0/(self.K_a * self.neu)) * float(self.sigma_q))).asarray()
            self.sigma_v = (1.0 - 1.0/(self.K_b * self.neu)) * self.sigma_v + (1.0 - (1.0 - 1.0/(self.K_b * self.neu))) * e[saida]**2
            self.forget_aux = (np.sqrt(self.sigma_q) * np.sqrt(self.sigma_v))/(1e-8 + abs(np.sqrt(self.sigma_e) - np.sqrt(self.sigma_v)))
            self.forget = np.atleast_2d(np.min([self.forget_aux,max_lambda]))
            #Transpose respective output view..
            Theta = self.Wro.asarray()[saida,:]
            Theta = Theta.reshape([self.neu,1])
            Theta = cm.CUDAMatrix(Theta)

            #MQR equations
            #the P equation step by step
            A = cm.dot(self.P,self.a)
            B = cm.dot(A,self.a.T)
            C = cm.dot(B,self.P)
            D = cm.dot(cm.dot(self.a.T,self.P),self.a).add(np.asscalar(self.forget))

            self.P.subtract(C.divide(np.asscalar(D.asarray())))
            self.P.divide(np.asscalar(self.forget))
            #final update


            #error calculation
            Theta.subtract(cm.dot(self.P,self.a).mult(np.asscalar(e[saida])))

            Theta = Theta.reshape([1,self.neu])


            self.Wro.copy_to_host()
            self.Wro.numpy_array[saida,:] = Theta.asarray()
            self.Wro.copy_to_device()





    def Update(self,input):
        # input has to have same size
        # as n_in. Returns the output as shape (2,1), so if yo
        # u want to plot the data, a buffer is mandatory.
        input = np.atleast_2d(input)
        Input = cm.CUDAMatrix(input)
        Input = Input.reshape([self.n_in,1])
        self.a = self.a.mult(1-self.leakrate).add(cm.tanh(cm.dot(self.Wrr,self.a).add(cm.dot(self.Wir,Input).add(self.Wbr))).mult(self.leakrate))
        y = cm.dot(self.Wro,self.a)
        return y.asarray()

    def CopyWeights(self, Rede):
        if self.Wro.shape == Rede.Wro.shape:
            self.Wro.assign(Rede.Wro)
            self.Wrr.assign(Rede.Wrr)
            self.Wbr.assign(Rede.Wbr)
            self.Wir.assign(Rede.Wir)
        else:
            print "shapes of the weights are not equal"

    def copy_output_weights(self, Rede):
        if self.Wro.shape == Rede.Wro.shape:
            self.Wro.assign(Rede.Wro)
        else:
            print "shapes of the weights are not equal"

    def SaveReservoir(self,fileName):
            data = {}
            data['Wrr'] = self.Wrr.asarray()
            data['Wir'] = self.Wir.asarray()
            data['Wbr'] = self.Wbr.asarray()
            data['Wro'] = self.Wro.asarray()
            data['a0'] = self.a
            io.savemat(fileName,data)

    def LoadReservoir(self,fileName):
            data = {}
            io.loadmat(fileName, data)
            self.Wrr = cm.CUDAMatrix(data['Wrr'])
            self.Wir = cm.CUDAMatrix(data['Wir'])
            self.Wbr = cm.CUDAMatrix(data['Wbr'])
            self.Wro = cm.CUDAMatrix(data['Wro'])
            self.a = cm.CUDAMatrix(data['a0'])


    def LoedInitial(self,filename):
        data = {}
        print "LOADING RESERVOIR"
        io.loadmat(filename, data)
        self.Wrr0 = cm.CUDAMatrix(data['Wrr'])
        self.Wir0 = cm.CUDAMatrix(data['Wir'])
        self.Wbr0 = cm.CUDAMatrix(data['Wbr'])
        self.Wro = cm.CUDAMatrix(data['Wro'])
        self.a = cm.CUDAMatrix(data['a0'])

    def save_initial_fun(self,filename):
        data = {}
        data['Wrr'] = self.Wrr0.asarray()
        data['Wir'] = self.Wir0.asarray()
        data['Wbr'] = self.Wbr0.asarray()
        data['Wro'] = self.Wro.asarray()
        data['a0'] = self.a.asarray()
        print "SAVING RESERVOIR"
        io.savemat(filename, data)

    def reservoir_test(self):

        return

    def trainLMS(self,ref):

        learningrate = 1

        e = self.trainingError(ref)
        for saida in range(self.n_out):
            Theta = self.Wro.asarray()[saida, :]
            Theta = Theta.reshape([self.neu, 1])
            Theta = Theta - learningrate*e*self.a.asarray()/np.dot(self.a.T,self.a.asarray())
            self.Wro.copy_to_host()
            self.Wro.numpy_array[saida, :] = Theta
            self.Wro.copy_to_device()

    def offline_train(self,X,Y,regularization): #X is a matrix in which X[i,:] is all parameters at time i. Y is a vector of desired outputs.

        return






