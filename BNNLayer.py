import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class BNNLinear(nn.Module):
    def __init__(self, in_features, out_features, alpha=1.0, bias=True):
        super(BNNLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.map = True
        self.register_buffer('min_logvar', -8*th.ones(1))

        self.m_w = nn.Parameter(th.ones(out_features, in_features))
        self.logsig2_w = nn.Parameter(th.ones(out_features, in_features))
        if self.bias:
            self.m_b = nn.Parameter(th.ones(out_features))
        else:
            self.register_parameter("m_b", None)

        self.alpha = alpha
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.m_w.size(1))
        self.m_w.data.normal_(0, stdv)
        self.logsig2_w.data.normal_(-6, 0.001)

        if self.bias:
            self.m_b.data.normal_(0, stdv)
        else:
            self.m_b.data.zero_()

    def KL(self):
        # Note: For now there is no Log-uniform prior. See old versions of this code to include it
        # logS = self.logsig2_w.clamp(
        #     -6, 6
        # )  # This is an artefact from log-uniform priors and can probably be removed
        logS = self.min_logvar + F.softplus(self.logsig2_w - self.min_logvar)
        kl = 0.5 * (self.alpha * (self.m_w.pow(2) + logS.exp()) - logS).sum()

        return kl

    def forward(self, input):
        if self.map:
            return F.linear(input, self.m_w, self.m_b)
        else:
            Mh = F.linear(input, self.m_w, self.m_b)
            s2_w = self.logsig2_w.exp()
            Vh = F.linear(input.pow(2), s2_w)
            return Mh + th.sqrt(Vh + 1e-16) * th.randn_like(Mh)
       

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )

    def get_neuron_count(self):
        return self.m_w.numel()



class BNNConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', alpha=1.0):
        super(BNNConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.alpha = alpha

        self.map = True
        self.register_buffer('min_logvar', -8*th.ones(1))
        self.register_buffer('max_logvar', 1*th.ones(1))


        self.m_w = nn.Parameter(th.Tensor(in_channels, out_channels, self.kernel_size, self.kernel_size))
        self.m_b = nn.Parameter(th.Tensor(out_channels))
        self.logsig2_w = nn.Parameter(th.Tensor(in_channels, out_channels, self.kernel_size, self.kernel_size))

        self.register_parameter('sig2_w_b', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0/ self.m_w.flatten(start_dim=0, end_dim=2).size(0) #1.0 / math.sqrt(self.m_w.weight.flatten(start_dim=2,end_dim=3).size(2))
        self.m_w.data.normal_(0, stdv)
        self.logsig2_w.data.normal_(-6, 0.001) 

        if self.bias:
            self.m_b.data.normal_(0, stdv)
        else:
            self.m_b.data.zero_()


    def KL(self):
        # Note: For now there is no Log-uniform prior. See old versions of this code to include it
        # logS = self.logsig2_w.clamp(
        #     -6, 6
        # )  # This is an artefact from log-uniform priors and can probably be removed
        logS = self.min_logvar + F.softplus(self.logsig2_w - self.min_logvar) 
        kl = 0.5 * (self.alpha * (self.m_w.pow(2) + logS.exp()) - logS).sum()

        return kl

    def forward(self, input):

        self.sig2_w = th.exp(self.logsig2_w) 
        act_mu = F.conv_transpose2d(
            input, self.m_w, self.m_b, stride=self.stride, padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation)


        act_var = 1e-16 + F.conv_transpose2d(
            input ** 2, self.sig2_w, self.sig2_w_b, stride=self.stride, padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation)
        act_std = th.sqrt(act_var)

        if self.map:
            return act_mu
        
        else:
        # if self.training:
            eps = th.randn_like(act_mu)
            return act_mu + act_std * eps

      
    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_channels)
            + " -> "
            + str(self.out_channels)
            + ")"
        )

    def get_neuron_count(self):
        return self.m_w.numel()
