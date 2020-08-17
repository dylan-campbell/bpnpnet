

import torch
import torch.nn as nn
import torch.nn.functional as F

class gcn(nn.Module):
  def __init__(self, in_channel):
    super(gcn, self).__init__()
    pass
    
  def forward(self, x):
    # x: [n, c, K]
    var_eps = 1e-3
    m = torch.mean(x, 2, keepdim=True)
    v = torch.var(x, 2, keepdim=True)
    inv = 1. / torch.sqrt(v + var_eps)
    x = (x - m) * inv
    return x


def bn_act(in_channel, gcn_in_channel, perform_gcn, perform_bn, activation,
           data_format):

    """
    # Global Context normalization on the input
    """
    layers = []
    if perform_gcn:
      layers.append(gcn(gcn_in_channel))

    if perform_bn:
      layers.append(nn.BatchNorm1d(in_channel,track_running_stats=False))

    if activation == 'relu':
      layers.append(torch.nn.ReLU())

    return layers


def conv1d_layer(in_channel, gcn_in_channel,out_channel, ksize, activation, perform_bn,
                 perform_gcn, perform_kron=False,
                 padding="CYCLIC", data_format="NCHW",
                 act_pos="post"):
    
    assert act_pos == "pre" or act_pos == "post"
    pool_func = None
    self_ksize = ksize
    do_add = False
    layers = []
    # If pre activation
    if act_pos == "pre":
      new = bn_act(in_channel, gcn_in_channel, perform_gcn, perform_bn, activation,
                        data_format)
      for l in new:
        layers.append(l)

    # Normal convolution
    layers.append(torch.nn.Conv1d(in_channel, out_channel, ksize))

    # If post activation
    if act_pos == "post":
      
      new = bn_act(out_channel, gcn_in_channel, perform_gcn, perform_bn, activation,
                        data_format)
      for l in new:
        layers.append(l)

    return nn.Sequential(*layers)


class conv1d_resnet_block(nn.Module):
  def __init__(self, in_channel, gcn_in_channel, out_channel, ksize, activation,
                        midchannel=None, perform_bn=False, perform_gcn=False,
                        padding="CYCLIC", act_pos="post", data_format="NCHW"):

    super(conv1d_resnet_block, self).__init__()
    # In case we want to do a bottleneck layer
    if midchannel is None:
        midchannel = out_channel

    # don't activate conv1 in case of midact
    if activation == 'relu':
      self.activation_fn = F.relu
    self.act_pos = act_pos
    
    self.preconv = conv1d_layer(
        in_channel=in_channel, 
        gcn_in_channel = gcn_in_channel,
        out_channel=midchannel,
        ksize=1,
        activation=None,
        perform_bn=False,
        perform_gcn=False,
        padding=padding,
        data_format=data_format,
    )

    # Main convolution
    self.conv1 = conv1d_layer(
      in_channel=midchannel, 
      gcn_in_channel = gcn_in_channel,
      out_channel=midchannel,
      ksize=1,
      activation=None,
      perform_bn=False,
      perform_gcn=perform_gcn,
      padding=padding,
      data_format=data_format,
    )

    # Main convolution
    self.conv2 = conv1d_layer(
      in_channel=midchannel, 
      gcn_in_channel = gcn_in_channel,
      out_channel=out_channel,
      ksize=1,
      activation=None,
      perform_bn=False,
      perform_gcn=perform_gcn,
      padding=padding,
      data_format=data_format,
    )

  def forward(self, x):
    xorg = x
    x = self.preconv(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.activation_fn(x)
    return x + xorg
