from ._dragon import Net
from collections import OrderedDict

@ property
def net_params(self):
    """
    return a dict {layer_name:layer_all_param_blobs}
    """
    return OrderedDict([(name,layer.blobs) for name,layer in zip(self._layer_names,self.layers)])

@ property
def net_blobs(self):
    """
    return a dict {blob_name:blob_array}
    """
    return OrderedDict(zip(self._blob_names,self._blobs))

@ property
def net_inputs(self):
    """
    self._inputs return the Net(C++)'s input blob indices
    return a list contains all input blob_names, e.g ['data','im_info',.....]
    """
    return [list(self.blobs.keys())[i] for i in self._inputs]

@ property
def net_outputs(self):
    """
    self._outputs return the Net(C++)'s output blob indices
    return a list contains all output blob_names, e.g ['accuracy','loss',.....]
    """
    return [list(self.blobs.keys())[i] for i in self._outputs]

def net_forward(self,blobs=None,start=None,end=None,**kwargs):
    """
    forward all layers
    return all output blobs as a dict {blob_name:blob_array}
    """
    if blobs is None:blobs=[]
    # get all output blob_names as set-list
    outputs=set(self.outputs+blobs)

    if kwargs:
        for key,blob in kwargs.iteritems():
            if blob.shape[0]!=self.blobs[key].num:
                raise Exception('Input Kwargs {} batch_size is not as same as net def.'.format(key))
            # copy and replace blob
            self.blobs[key].data[...]=blob

    if start is not None:
        start = list(self._layer_names).index(start)
    else:start=0

    if end is not None:
        end = list(self._layer_names).index(end)
        outputs = set([end] + blobs)  # onyl output end blob
    else:end=len(self.layers)-1

    # Net._forward all layers
    self._forward(start,end)
    # get all output blobs as a dict
    return {out: self.blobs[out].data for out in outputs}

def net_backward(self,diffs=None,start=None,end=None,**kwargs):
    """
    backward all layers
    return all output diffs as a dict {blob_name:blob_array}
    """
    if diffs is None:diffs=[]
    outputs = set(self.inputs + diffs)

    if kwargs:
        for key,blob in kwargs.iteritems():
            if blob.shape[0]!=self.blobs[key].num:
                raise Exception('Input Kwargs {} batch_size is not as same as net def.'.format(key))
            # copy and replace blob
            self.blobs[key].diff[...]=blob

    if start is not None:
        start = list(self._layer_names).index(start)
    else:start=len(self.layers)-1

    if end is not None:
        end = list(self._layer_names).index(end)
        outputs = set([end] + diffs)  # onyl output end blob
    else:end=0

    # Net._forward all layers
    self._backward(start,end)
    # get all output blobs as a dict
    return {out: self.blobs[out].diff for out in outputs}



Net.blobs=net_blobs      # property
Net.inputs=net_inputs    # property
Net.outputs=net_outputs  # property
Net.forward=net_forward  # function
Net.backward=net_backward # function
Net.params=net_params    # property
