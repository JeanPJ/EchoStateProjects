__author__ = 'jean'
import numpy as np
import scipy.io as io


def Sparcity(M,psi):
    N = np.empty_like(M)
    for linha in range(len(N)):
        print "linha:",linha
        for coluna in range(len(N[linha])):
            prob = np.random.rand()
            if prob < psi:
                N[linha][coluna] = 0
            else:
                N[linha][coluna] = 1


    return N*M


class rNN:

    def __init__(self,neu,n_in,n_out,
                 gama=0.5,ro=1,psi=0.5,in_scale=0.1,
                 bias_scale=0.5,alfa=10,forget = 1,
                 initial_filename="initial",
                 load_initial = False,save_initial = False):
        #All matrixes are initialized under the normal distribution.
        print "initializing reservoir"
        print n_in,"Number of inputs"
        self.neu = neu
        self.n_in = n_in
        self.n_out = n_out

        # Reservoir Weight matrix.
        print "initializing reservoir matrix"
        self.Wrr0 = np.random.normal(0,1,[neu,neu],)
        print "initializing input matrix"
        # input-reservoir weight matrix
        self.Wir0 = np.random.normal(0,1,[neu,n_in])
        # bias-reservoir weight matrix
        print "initializing bias matrix"
        self.Wbr0 = np.random.normal(0,1,[neu,1])

        #self.Wbo = np.random.normal(0,1,[n_out,1])
        # reservoir-output weight matrix
        print "initializing Wro"
        self.Wro = np.random.normal(0,1,[n_out,neu])

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
        self.a = np.zeros([neu, 1],dtype=np.float64)
        #save if save is enabled
        if save_initial:
            self.save_initial_fun(initial_filename)

        #load if load is enabled
        if load_initial:
            self.LoedInitial(initial_filename)
            print self.Wrr0

        # the probability of a memeber of the Matrix Wrr being zero is psi.
        print "define sparseness"
        if psi > 0:
            self.Wrr = Sparcity(self.Wrr0,self.psi)
        else:
            self.Wrr = self.Wrr0
        #forcing Wrr to have ro as the maximum eigenvalue
        print "calculating eigenvalues"
        eigs = np.linalg.eigvals(self.Wrr)
        print "finding maximum eigenvalue"
        radius = np.abs(np.max(eigs))
        #normalize matrix
        print "normalize reservoir"
        self.Wrr = self.Wrr/radius
        #set its spectral radius to rho
        self.Wrr *= ro

        print self.Wrr,"Wrr after scaling"

        #scale tbe matrices
        self.Wbr = bias_scale*self.Wbr0
        self.Wir = in_scale*self.Wir0



        #initial conditions variable forget factor.
        self.sigma_e = 0.001
        self.sigma_q = 0.001
        self.sigma_v = 0.001
        self.K_a = 6
        self.K_b = 3*self.K_a

        #covariance matrix
        self.P = np.eye(neu)/alfa
        print "Reservoir initialization Done"

    def getWro(self,n=0): #retorna a coluna n do vetor Wro

        return self.Wro[n]

    def trainingError(self,ref):

        Ref = np.array(ref,dtype = np.float64)
        if self.n_out > 1:
            Ref = Ref.reshape(len(ref),1)

        e = np.dot(self.Wro,self.a) - Ref
        return e

    def Train(self,ref):
        #ref e o vetor de todas as sa
        # idas desejados no dado instante de tempo.
        #calcular o vetor de erros
        e = self.trainingError(ref)

        for saida in range(self.n_out):


            self.sigma_e = (1 - 1/(self.K_a*
                                   self.neu))*self.sigma_e + \
                           (1 - (1 - 1/(self.K_a*self.neu)))*\
                           e[saida]**2
            self.sigma_q = (1 - 1/(self.K_a*self.neu))*\
                           self.sigma_q + \
                           (1 - (1 - 1/(self.K_a*self.neu)))*\
                           (np.dot(np.dot(self.a.T,self.P),self.a))\
                           **2

            self.sigma_v = (1 - 1/(self.K_b*self.neu))\
                           *self.sigma_v + \
                           (1 - (1 - 1/(self.K_b*self.neu)))*e[saida]**2
            self.forget = np.min([(np.sqrt(self.sigma_q) *
                                   np.sqrt(self.sigma_v))
                                   /(10**-8 + abs(np.sqrt(self.sigma_e) -
                                   np.sqrt(self.sigma_v))),0.99999])
            #Transpose respective output view..
            Theta = self.Wro[saida,:]
            Theta = Theta.reshape([self.neu,1])

            #MQR equations

            #the P equation step by step
            A = self.P/self.forget
            B = np.dot(self.P,self.a)
            C = np.dot(B,self.a.T)
            D = np.dot(C,self.P)
            E = np.dot(self.a.T,self.P)
            G = np.dot(E,self.a)
            F = self.forget + G

            #final update
            self.P = A - D/(self.forget*F)


            #error calculation
            Theta = Theta - e[saida]*B

            Theta = Theta.reshape([1,self.neu])

            self.Wro[saida,:] = Theta




    def Update(self,input):
        # input has to have same size
        # as n_in. Returns the output as shape (2,1), so if yo
        # u want to plot the data, a buffer is mandatory.
        Input = np.array(input)
        Input = Input.reshape(Input.size,1)
        if Input.size == self.n_in:
            self.a = (1-self.leakrate)*self.a + \
                     self.leakrate*np.tanh(np.dot(self.Wrr,self.a) +
                     np.dot(self.Wir,Input) + self.Wbr)
            y = np.dot(self.Wro,self.a)
            return y
        else:
            raise ValueError("input must have size n_in")

    def CopyWeights(self, Rede):
        if self.Wro.shape == Rede.Wro.shape:
            self.Wro = np.copy(Rede.Wro)
            self.Wrr = np.copy(Rede.Wrr)
            self.Wir = np.copy(Rede.Wir)
            self.Wbr = np.copy(Rede.Wbr)
        else:
            print "shapes of the weights are not equal"

    def SaveReservoir(self,fileName):
            data = {}
            data['Wrr'] = self.Wrr
            data['Wir'] = self.Wir
            data['Wbr'] = self.Wbr
            data['Wro'] = self.Wro
            data['a0'] = self.a
            io.savemat(fileName,data)

    def LoadReservoir(self,fileName):
            data = {}
            io.loadmat(fileName, data)
            self.Wrr = data['Wrr']
            self.Wir = data['Wir']
            self.Wbr = data['Wbr']
            self.Wro = data['Wro']
            self.a = data['a0']

    def LoedInitial(self,filename):
        data = {}
        print "loading reservoir"
        io.loadmat(filename, data)
        self.Wrr0 = data['Wrr']
        self.Wir0 = data['Wir']
        self.Wbr0 = data['Wbr']
        self.Wro = data['Wro']
        self.a = data['a0']

    def save_initial_fun(self,filename):
        data = {}
        data['Wrr'] = self.Wrr0
        data['Wir'] = self.Wir0
        data['Wbr'] = self.Wbr0
        data['Wro'] = self.Wro
        data['a0'] = self.a
        print "saving reservoir"
        io.savemat(filename, data)

    def reservoir_test(self):

        return

    def trainLMS(self,ref):

        learningrate = 1

        e = self.trainingError(ref)
        for saida in range(self.n_out):
            Theta = self.Wro[saida, :]
            Theta = Theta.reshape([self.neu, 1])
            Theta = Theta - learningrate*e*self.a/np.dot(self.a.T,self.a)
            self.Wro[saida, :] = Theta.T






