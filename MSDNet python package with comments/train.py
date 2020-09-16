# The train.py defines the training algorithm and training process.

class TrainAlgorithm(abc.ABC):
    """Base class implementing a training algorithm."""
    @abc.abstractmethod
    def step(self, n, dlist):
        """Take a single algorithm step.
        
        :param n: :class:`.network.Network` to train with
        :param dlist: list of :class:`.data.DataPoint` to train with
        """
        pass
    
    @abc.abstractmethod
    def to_dict(self):
        """Save algorithm state to dictionary"""
        pass
    
    @abc.abstractmethod
    def load_dict(self, dct):
        """Load algorithm state from dictionary"""
        pass
    
    @classmethod
    @abc.abstractmethod
    def from_dict(cls, dct):
        """Load algorithm from dictionary"""
        pass
    
    @classmethod
    def from_file(cls, fn):
        """Load algorithm from file"""
        dct = store.get_dict(fn, 'trainalgorithm')
        return cls.from_dict(dct)
    
    def to_file(self, fn):
        """Save algorithm state to file"""
        store.store_dict(fn, 'trainalgorithm', self.to_dict())

class AdamAlgorithm(TrainAlgorithm):
    """Implementation of the ADAM algorithm.
    
    :param network: :class:`.network.Network` to train with
    :param a: ADAM parameter
    :param b1: ADAM parameter
    :param b2: ADAM parameter
    :param e: ADAM parameter
    """
    def __init__(self,  network, a = 0.001, b1 = 0.9, b2 = 0.999, e = 10**-8):  # b1: rou_1; b2: rou_2;
        self.a = a
        self.b1 = b1
        self.b1t = b1
        self.b2 = b2
        self.b2t = b2
        self.e = e
        if network:
            self.npars = network.getgradients().shape[0]
            self.m = np.zeros(self.npars)
            self.v = np.zeros(self.npars)
    
    def step(self, n, dlist):  # n is network; dlist is data list
        n.gradient_zero()
        tpix = 0
        for d in dlist:
            inp, tar, msk = d.getall()  # getall(self) return input image, target image, and mask image (when given).
            out = n.forward(inp)  # network forward propagation, obtaining the output channels
            err = tar - out  # compute the output error with respect to the target
            if msk is None:
                tpix += err.size
            else:
                msk = (msk == 0)
                err[:, msk] = 0
                tpix += err.size - err.shape[0]*msk.sum()
            n.backward(err)  # execute the backpropagtion step
            n.gradient()  # compute the weights' gradient, offsets(biases)' gradient and filters' gradient
        g = n.getgradients()  # stack the computed gradient values for each type of trainable parameters to be a vector
        g/=tpix  # final computed gradient (parameter vector)
        self.m *= self.b1  # m is s,i.e. estimation of expectation of g
        self.m += (1-self.b1)*g
        self.v *= self.b2  # v is r,i.e. estimation of expectation of g*g
        self.v += (1-self.b2)*(g**2)
        mhat = self.m/(1-self.b1t)  # mhat is corrected bias of expectation of g
        vhat = self.v/(1-self.b2t)  # vhat is corrected bias of expectation of g*g
        self.b1t *= self.b1  # being saved to dictionary
        self.b2t *= self.b2  # being saved to dictionary
        upd = self.a * mhat/(np.sqrt(vhat) + self.e)  # change on gradient
        n.updategradients(upd)  # update gradient iteratively
