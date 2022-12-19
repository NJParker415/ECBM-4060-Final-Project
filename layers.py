# Custom nn layers, adapted from the paper and converted into pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
import numpy as np


class Attention(nn.Module):
  def __init__(self, size_in, size_out):
    super(Attention, self).__init__()

    self.size_in, self.size_out = size_in, size_out
    weights = torch.Tensor(size_out, size_in)
    self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
    bias = torch.Tensor(size_out)
    self.bias = nn.Parameter(bias)

    nn.init.xavier_uniform(self.weights)

  def forward(self, x):
    return torch.mul(self.weights, x)

class AttLayer(nn.Module):
  def __init__(self, size_in, size_out):
    super(AttLayer, self).__init__()

    self.size_in, self.size_out = size_in, size_out
    weights = torch.Tensor(size_out, size_in)
    self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
    bias = torch.Tensor(size_out)
    self.bias = nn.Parameter(bias)

    nn.init.normal(self.weights)

  def forward(self, x, mask=None):
    eij = torch.tanh(torch.dot(x, self.weights))
    ai = torch.exp(eij)
    weights = torch.div(torch.sum(ai, axis=1).randperm(weights.size()[0]), ai)
    weighted_input = torch.mul(weights.randperm(weights.size()[0]), x)
    weighted_input = weighted_input.sum(axis=1)
    return weighted_input

class DiagLayer(nn.Module):
  def __init__(self, in_features, out_features):
    super(DiagLayer, self).__init__()

    self.size_in, self.size_out = in_features, out_features
    weights = torch.Tensor(self.size_out, self.size_in)
    self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
    bias = torch.Tensor(self.size_out)
    self.bias = nn.Parameter(bias)

    nn.init.normal_(self.weights)

  def forward(self, x):
    n_features = x.size()[1]

    self.n_inputs_per_node = int(x.size()[1] / 4)

    print(x.shape)
    print(self.weights.shape)

    mult = torch.bmm(x, self.weights)
    mult = torch.reshape(mult, (-1, self.n_inputs_per_node))
    mult = torch.sum(mult, axis=1)
    output = torch.reshape(mult, (1, self.size_out))

    return F.linear(output, self.weights, self.bias)

class SparseTF(nn.Module):
  __constants__ = ['in_features', 'out_features']
  in_features: int
  out_features: int
  weight: Tensor

  def __init__(self, in_features: int, out_features: int, bias: bool = True,
               device=None, dtype=None, mask=None) -> None:
      factory_kwargs = {'device': device, 'dtype': dtype}
      super(SparseTF, self).__init__()
      self.in_features = in_features
      self.out_features = out_features
      self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
      if bias:
          self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
      else:
          self.register_parameter('bias', None)
      self.reset_parameters()

      nonzero_ind = (mask.to_numpy() != 0).T
      self.nonzero_ind = torch.from_numpy(nonzero_ind).type(torch.int64).to(device)

  def reset_parameters(self) -> None:
      nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
      if self.bias is not None:
          fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
          bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
          nn.init.uniform_(self.bias, -bound, bound)

  def forward(self, input: Tensor) -> Tensor:
      tt = torch.zeros_like(self.weight)
      tt = tt.scatter(0, self.nonzero_ind, self.weight)
      return F.linear(input, tt, self.bias)

  def extra_repr(self) -> str:
      return 'in_features={}, out_features={}, bias={}'.format(
          self.in_features, self.out_features, self.bias is not None
      )
    

    