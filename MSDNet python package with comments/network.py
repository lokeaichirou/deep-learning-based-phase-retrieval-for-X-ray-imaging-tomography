# network.py defines the structure of Mixed-scale dense network

class Network(abc.ABC):
    def normalizeinout(self, datapoints):  # this datapoints are objects of #RotateAndFlipDataPoint(OnlyAllDataPoint)
        # class
        # Normalize the input and output in case of internal covariance shift
        """Normalize input and output of network to zero mean and unit variance.

        :param datapoints: list of datapoints to compute normalization factors with.
        """
        self.normalizeinput(datapoints)
        self.normalizeoutput(datapoints)

class MSDNet(Network):
    """Main implementation of a Mixed-Scale Dense network.
    
    :param d: depth of network (width is always 1)
    :param dil: :class:`.dilations.Dilations` class defining dilations
    :param nin: number of input channels
    :param nout: number of output channels
    :param gpu: (optional) whether to use GPU or CPU
    """

    def __init__(self, d, dil, nin, nout, gpu=True):

        self.d = d  # depth of network (width is always 1), depth of network = number of hidden layers
        self.nin = nin  # number of input channels
        self.nout = nout   # number of output channels

        # Fill dilation list
        if dil:
            dil.reset()
            self.dl = np.array([dil.nextdil() for i in range(d)],dtype=np.int32)  # Creating dilatation list, it uses
            # the nextdil method is in dilations.py

        # Set up temporary images, force creation in first calls
        self.ims = np.zeros(1)  # Set up the intermediate images
        self.delta = np.zeros(1)  # Set the difference between output and target images
        self.indelta = np.zeros(1)

        self.fshape = (3,3)  # set the size of dilation convolutional filter
        self.axesslc = (slice(None), None, None)
        self.revf = (slice(None,None,-1),slice(None,None,-1))
        self.ndim = 2

        if gpu:
            from . import gpuoperations
            self.dataobject = gpuoperations.GPUImageData
        else:
            self.dataobject = operations.ImageData  # set up intermediate images

        # Set up filters
        self.f = []  # f is a list that contains the set of dilation convolution filters for each layer; each layer is
        # the concatenation result of all previous layers; each later collects all previous layers and concatenate them
        # together; since it's a regression problem, the output layer has same size as input image, each layer's
        # convolution processing doesn't change the size of intermediate images, even they (intermediate images) are
        # saved and processed as a vector in reality
        for i in range(d):
            self.f.append(np.zeros((nin+i, *self.fshape),dtype=np.float32))  # Because we set the width of MSDNet as 1,
            # so each layer we only add 1 more dilatation convolution filter

        self.fg = [np.zeros_like(k) for k in self.f]  # self.fg is filter gradient,it has identical size as self.f;
        # self.f filter was initialized to be gaussian in network initialization before

        # Set up weights
        self.w = np.zeros((nout, nin+d), dtype=np.float32)  # initialize the linear combination coefficients to be zero
        # for output layer (output image channels); since the output channels are linear combination results of all
        # intermediate images, number of intermediate images = number of input channels + depth of network
        # (number of hidden layers)
        self.wg = np.zeros_like(self.w)  # self.wg is weight gradient, it has identical size as self.w, self.w was
        # initialized to be zero in network initialization

        # Set up offsets
        self.o = np.zeros(d, dtype=np.float32)  # offset(bias) for intermediate images (feature maps that include input
        # channels and newly added feature map by each hidden layer )
        self.og = np.zeros_like(self.o)  # offset(bias) gradient
        self.oo = np.zeros(nout, dtype=np.float32)  # offset(bias) for output channels
        self.oog = np.zeros_like(self.oo)  # gradient of offset(bias) for output channels

    def forward(self, im, returnoutput=True): # forward defines the forward 
    # propagation
        if self.nin==1 and len(im.shape)==self.ndim:
            im = im[np.newaxis]
        if im.shape[0]!=self.nin:
            raise ValueError("Number of input channels ({}) does not match expected number ({}).".format(im.shape[0], self.nin))
        if im.shape[1:]!=self.ims.shape[1:]:
            self.ims = self.dataobject((self.d+self.nin, *im.shape[1:]), self.dl, self.nin)  # ims is a list, each element
            # in the list corresponds to features maps collected (intermediate images) for each layer;
            # since width is 1, each hidden layer we add one more feature map that is convolution result based on all 
            # feature maps collected at current hidden layer 
            # they are initialized to be zero
            self.out = self.dataobject((self.nout, *im.shape[1:]), self.dl, self.nin)  # out is a list, it only contains
            # the final output layer;they are initialized to be zero
        self.ims.setimages(im)  # Give the first several feature maps values as input channels' images values
        self.scaleinput()  # Normalize the input channels in self.ims to be Gaussian distribution (0,1) according to gam-in adn off-in computed
        self.ims.setscalars(self.o[self.axesslc], start=self.nin)  # initialize other intermediate images (except for 
        # input channels) to be zero(blank)
        self.ims.prepare_forw_conv(self.f)  # Prepare the dilatation convolution filter as a list, each element in the
        # list corresponds to set of dilatation convolution filters of same
        for i in range(self.d):  # this is the most important step in network forward propagation step
            self.ims.forw_conv(i, self.nin+i, self.dl[i])  # Perform forward convolutions for each layer's (only) newly
            # added intermediate image, forward convolution for each subsequent layer's newly added intermediate image
            # is based on all previous intermediate images
            self.ims.relu(self.nin+i)  # Activate each layer with ReLU
        self.out.setscalars(self.oo[self.axesslc])  # Initialize the output layer to be zero
        self.out.combine_all_all(self.ims, self.w)  # w is the linear combination coefficients, take the linear
        # combination of all previous feature maps (intermediate images) and applying an application-specific activation
        # function to the result
        self.scaleoutput()  # Normalize the output layer to be Gaussian distribution (0,1)
        if returnoutput:
            return self.out.copy()

    def backward(self, im, inputdelta=False):  # im = err
     # backward defines the backward propagation
        if im.shape[1:]!=self.delta.shape[1:]:
            self.delta = self.dataobject((self.d, *im.shape[1:]), self.dl, self.nin)  # self.delta is an object from
            # operations.ImageData class, it is a list, each element in the list corresponds to features maps
            # (intermediate images) for each layer; they are initialized to be zero
            self.delta.fill(0)  # initialize self.delta to be zero
            self.deltaout = self.dataobject(im.shape, self.dl, self.nin)  # self.deltaout is a list, it only contains
            self.delta.prepare_gradient()
        else:
            self.delta.fill(0)
        self.deltaout.setimages(im)  # Give the first several feature maps values as output channels' images values
        self.scaleoutputback()
        wt = self.w[:,self.nin:].transpose().copy()
        self.delta.combine_all_all(self.deltaout, wt)
        self.delta.relu2(self.delta.shape[0]-1, self.ims, self.ims.shape[0]-1)

        back_f = {}
        for i in reversed(range(self.d-1)):
            fb = np.zeros((self.d-i-1,*self.fshape),dtype=np.float32)
            for j in range(i+1,self.d):
                fb[j-i-1] = self.f[j][self.nin+i][self.revf]
            back_f[i] = fb
        self.delta.prepare_back_conv(back_f)

        for i in reversed(range(self.d-1)):
            self.delta.back_conv(i,self.dl)
            self.delta.relu2(i, self.ims, self.nin+i)

        if inputdelta:
            if im.shape[1:]!=self.indelta.shape[1:]:
                self.indelta = np.zeros((self.nin, *im.shape[1:]), dtype=np.float32)
            self.indelta.fill(0)
            do = self.deltaout.get()
            de = self.delta.get()
            for i in range(self.nin):
                fb = np.zeros((self.d,*self.fshape),dtype=np.float32)
                for j in range(self.d):
                    fb[j] = self.f[j][i][self.revf]
                for j in range(self.nout):
                    operations.combine(do[j], self.indelta[i], self.w[j,i])
                for j in range(self.d):
                    operations.conv2d(de[j], self.indelta[i], fb[j], self.dl[j])

    def initialize(self):  # network initialization
        """Initialize network parameters."""
        for f in self.f:
            f[:] = np.sqrt(2/(f[0].size*(self.nin+self.d-1)+self.nout))*np.random.normal(size=f.shape)
            # initialize each dilation convolution filter to be gaussian distributed, amplitude of gaussian distribution
            # depends on dilation convolution filter size (3*3), number of filters totally (in terms of all layers)
            # and number of output channels
        self.o[:]=0
        self.w[:]=0
        self.oo[:]=0
        
    def normalizeinput(self, datapoints):
        """Normalize input of network to zero mean and unit variance.

        :param datapoints: list of datapoints to compute normalization factors with.
        """
        # it is not batch normalization for batch size != 1;  it is not classical layer normalization for batch size = 1
        # because it doesn't have shift value added i.e. stochastic gradient descent or online learning

        nd = len(datapoints)
        allmeans = []  # allmeans is a list that contains list of means of input array channels for all the d in 
            # datapoints (pairs of input array and target array)
            # of input array and target array)
        allstds = []  # allstds is a list that contains list of standard deviations of input array channels for all the d
            # in datapoints (pairs of input array and target array)
        for d in datapoints:  # each d in datapoints is class RotateAndFlipDataPoint object
            inp, _, _ = d.getall()  # each time we get one input array in  dadapoints list
            means = []  # means is a list that contains mean values for each channel of input array of a datapoint(a pair
            # of input array and target array) d
            stds = []  # stds is a list that contains standard deviation values for each channel of input array of 
            # datapoint(a pair of input array and target array) d
            for im in inp:  # im is each channel of input array inp 
                mn = operations.sum(im)/im.size  # compute a channel's mean value mn 
                std = operations.std(im, mn)  # compute a channel's standard deviation value std
                means.append(mn)  
                stds.append(std)
            allmeans.append(means)
            allstds.append(stds)
        mean = np.array(allmeans).mean(0)  # mean value that considers all channels of all input arrays
        std = np.array(allstds).mean(0)   # standard deviation value that considers all channels of all input arrays

        self.gam_in = (1/std).astype(np.float32)  # scaling factor in normalization
        self.off_in = (-mean/std).astype(np.float32)  # bias added in normalization

        # The goal of function normalizeinput is to compute self.gam_in and self.off_in

    def normalizeoutput(self, datapoints):
        """Normalize output of network to zero mean and unit variance.

        :param datapoints: list of datapoints to compute normalization factors with.
        """
        # it is not batch normalization for batch size != 1;  it is not calsscal layer normalization for batch size = 1,
        # i.e. stochastic gradient descent or online learning

        nd = len(datapoints)
        allmeans = []
        allstds = []
        for d in datapoints:
            _, inp, _ = d.getall()
            means = []
            stds = []
            for im in inp:
                mn = operations.sum(im)/im.size
                std = operations.std(im, mn)
                means.append(mn)
                stds.append(std)
            allmeans.append(means)
            allstds.append(stds)
        mean = np.array(allmeans).mean(0)
        std = np.array(allstds).mean(0)

        self.gam_out = (std).astype(np.float32)
        self.off_out = (mean).astype(np.float32)

        # The goal of function normalizeoutput is to compute self.gam_in and self.off_in
