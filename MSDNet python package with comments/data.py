class DataPoint(abc.ABC):
    """Base class for a single data point (input image with corresponding target image)"""
    
    # Each time we load a pair of input and target output array,making it be like an object ArrayDataPoint, then process it by making it be like object RotateAndFlipDataPoint that further process the pair(input and output) of array

class ArrayDataPoint(DataPoint):
    """Datapoint with numpy array image data.
    
    :param inputarray: numpy array with input image (size: :math:`N_{c} \\times N_{x} \\times N_{y}`)
    :param targetarray: (optional) numpy array with target image (size: :math:`N_{c} \\times N_{x} \\times N_{y}`)
    :param maskarray: (optional) numpy array with mask image (size: :math:`N_{c} \\times N_{x} \\times N_{y}`)
    """
    def __init__(self, inputarray, targetarray=None, maskarray=None):
        self.iarr = inputarray.astype(np.float32)
        if not targetarray is None:
            self.tarr = targetarray.astype(np.float32)
        else:
            self.tarr = None
        if not maskarray is None:
            self.marr = maskarray.astype(np.float32)
        else:
            self.marr = None
    
    def getinputarray(self):  # return input array 
        return self.iarr
    
    def gettargetarray(self):  # return target output array
        return self.tarr

    def getmaskarray(self):
        return self.marr

class RotateAndFlipDataPoint(OnlyAllDataPoint):
    """Datapoint that augments input datapoint with rotations and flips.
    
    :param datapoint: input :class:`DataPoint`.
    """
    def __init__(self, datapoint, target_snr_in_db=None):
        self.dp = datapoint
        self.__resetlist()
        self.target_snr_in_db = target_snr_in_db

    def __resetlist(self):
        self.lst = list(range(8))
        random.shuffle(self.lst)
    
    def getall(self):
        inp = self.dp.input 
        # since class RotateAndFlipDataPoint follows class OnlyAllDataPoint, and class OnlyAllDataPoint follows class DataPoint, so class RotateAndFlipDataPoint follows DataPoint, since the input method in class DataPoint will return self.getinputarray().astype(np.float32), and self.dp = datapoint is an object from class ArrayDataPoint, in class ArrayDataPoint, the getinputarray() method defined in ArrayDataPoint class return input array to class ArrayDataPoint object
        
        tar = self.dp.target  
        # similar to inp above, it returns target output array
        
        msk = self.dp.mask
        c = self.lst.pop()

        inp_shape = inp.shape
        num_channel_inp = inp_shape[0]

        if self.target_snr_in_db:
            inp[num_channel_inp-1], _, std_of_noise_for_longest_distance = add_gaussian_noise_with_given_peak_to_peak_snr_for_longest_distance(
                input_array=inp[num_channel_inp-1], target_snr_db=self.target_snr_in_db, mean_noise=0)

            for channel in range(num_channel_inp-1, -1, -1):
                inp[channel] = add_gaussian_noise_with_given_peak_to_peak_snr_for_shorter_distance(
                    input_array=inp[channel], mean_noise=0, std_of_noise_for_longest_distance=std_of_noise_for_longest_distance)

        if len(self.lst)==0:
            self.__resetlist()
        if c==1:
            inp, tar = inp[:,::-1], tar[:,::-1]
        elif c==2:
            inp, tar = inp[:,:,::-1], tar[:,:,::-1]  # horizontally flip
        elif c==3:
            inp, tar = inp[:,::-1,::-1], tar[:,::-1,::-1]  # horizontally and vertically flip
        elif c==4:
            inp, tar = np.rot90(inp,1,axes=(1,2)), np.rot90(tar,1,axes=(1,2))
            # anti-clockwise rotate 90 degree
        elif c==5:
            inp, tar = np.rot90(inp,3,axes=(1,2)), np.rot90(tar,3,axes=(1,2))
            # clockwise rotate 90 degree
        elif c==6:
            inp, tar = np.rot90(inp,1,axes=(1,2))[:,::-1], np.rot90(tar,1,axes=(1,2))[:,::-1]
            # anti-clockwise rotate 90 degree and then horizontally flip 
        elif c==7:
            inp, tar = np.rot90(inp,3,axes=(1,2))[:,::-1], np.rot90(tar,3,axes=(1,2))[:,::-1]
            # clockwise rotate 90 degree and then horizontally flip
        inp = np.ascontiguousarray(inp)  # The ascontiguousarray function converts an array of discontinuously stored 
        # memory (because of flipping and rotation processing) into an array of continuously stored memory, making it
        # run faster
        tar = np.ascontiguousarray(tar)
        if not msk is None:
            if c==1:
                msk = msk[::-1]
            elif c==2:
                msk = msk[:,::-1]
            elif c==3:
                msk = msk[::-1,::-1]
            elif c==4:
                msk = np.rot90(msk,1)
            elif c==5:
                msk = np.rot90(msk,3)
            elif c==6:
                msk = np.rot90(msk,1)[::-1]
            elif c==7:
                msk = np.rot90(msk,3)[::-1]
            msk = np.ascontiguousarray(msk)
        return inp, tar, msk
        
class BatchProvider(object):
    """Object that returns small random batches of datapoints.
    
    # Since we have collected all the processed pairs(input and target output) of arrays into a list [] namely dats, in practical training, normally we do not apply batch gradient descent(each iteration of parameter update depends on all the training data) because it is too much computationally expensive, the parameter update will be too slow; so, instead of batch gradient descent, we use mini-batch or stochastic gradient descent(online learning that each iteration of parameter update depends on only one (pair) data point
    
    :param dlist: List of :class:`DataPoint`.
    :param batchsize: Number of datapoints per batch.
    :param seed: (optional) Random seed.
    """

    def __init__(self, dlist, batchsize, seed=None):
        self.d = dlist  # list that contains pair of input and target output arrays
        self.nd = len(self.d)  # nd is number of image sets(image set means a pair of input and target output arrays) in data provided
        self.rndm = np.random.RandomState(seed)  # define a random generator seed
        self.idx = np.arange(self.nd,dtype=np.int)  # idx list has same length # as nd, it contains numbers from 0,1,2,...,to nd-1
        self.rndm.shuffle(self.idx)  # randomly shuffle the idx list
        self.bsize = batchsize  # batch size for every batch
        self.i = 0
    
    def getbatch(self):
        """Return batch of datapoints."""
        batch = []  # batch list contains the image sets that was randomly selected
        while len(batch)<self.bsize:  # each time it returns a batch with size<= batch size set before
            if self.i>=self.nd:
                self.i = 0
                self.rndm.shuffle(self.idx)
            batch.append(self.d[self.idx[self.i]])  # since the self.idx list was randomly shuffled, like 
            # [1,0,23,155,888,6,...,77,998,666], we extract the element in self.d (list that contains the input and 
            # target output arrays) in random order 
            self.i+=1
        return batch  # return the list 
