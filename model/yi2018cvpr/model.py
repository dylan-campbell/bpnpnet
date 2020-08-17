
import torch
import torch.nn as nn
from .ops import conv1d_layer, conv1d_resnet_block

class Net(nn.Module):
  def __init__(self, config):
    super(Net, self).__init__()

    activation = 'relu'

    idx_layer = 0
    self.numlayer = config.net_depth
    ksize = 1
    in_channel = config.in_channel
    nchannel = config.net_nchannel
    gcn_in_channel = config.gcn_in_channel
    # Use resnet or simle net
    act_pos = config.net_act_pos
    conv1d_block = conv1d_resnet_block

    # First convolution
  
    self.conv_in = conv1d_layer(
        in_channel = in_channel,
        gcn_in_channel = gcn_in_channel,
        out_channel = nchannel,
        ksize=1,
        activation=None,
        perform_bn=False,
        perform_gcn=False,
        act_pos="pre",
        data_format="NHWC",
    )
    
    
    for _ksize, _nchannel in zip(
            [ksize] * self.numlayer, [nchannel] * self.numlayer):
      setattr(self, 'conv_%d' % idx_layer, conv1d_block(
          in_channel = nchannel,
          gcn_in_channel = gcn_in_channel,
          out_channel = nchannel,
          ksize=_ksize,
          activation=activation,
          perform_bn=config.net_batchnorm,
          perform_gcn=config.net_gcnorm,
          act_pos=act_pos,
          data_format="NHWC",
      ))
      idx_layer += 1

    # self.conv_out = conv1d_layer(
    #     in_channel = nchannel,
    #     gcn_in_channel = gcn_in_channel,
    #     out_channel = 1,
    #     ksize=1,
    #     activation=None,
    #     perform_bn=False,
    #     perform_gcn=False,
    #     data_format="NHWC",
    # )

  def forward(self, x):
    x_in_shp = x.shape
    x = self.conv_in(x)
    for i in range(self.numlayer):
      x = getattr(self, 'conv_%d' % i)(x)
    return x
    # x = self.conv_out(x)
    # # x = x.view(x_in_shp[0], x_in_shp[2])
    # logits = x
    # return logits
    
