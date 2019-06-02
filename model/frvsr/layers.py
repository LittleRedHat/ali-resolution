# -*- coding: utf-8 -*-
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-24         #
__author__ = "zookeeper"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpaceToDim(nn.Module):
  def __init__(self, scale_factor, dims=(-2, -1), dim=0):
    super(SpaceToDim, self).__init__()
    self.scale_factor = scale_factor
    self.dims = dims
    self.dim = dim

  def forward(self, x):
    _shape = list(x.shape)
    shape = _shape.copy()
    dims = [x.dim() + self.dims[0] if self.dims[0] < 0 else self.dims[0],
            x.dim() + self.dims[1] if self.dims[1] < 0 else self.dims[1]]
    dims = [max(abs(dims[0]), abs(dims[1])),
            min(abs(dims[0]), abs(dims[1]))]
    if self.dim in dims:
      raise RuntimeError("Integrate dimension can't be space dimension!")
    shape[dims[0]] //= self.scale_factor
    shape[dims[1]] //= self.scale_factor
    shape.insert(dims[0] + 1, self.scale_factor)
    shape.insert(dims[1] + 1, self.scale_factor)
    dim = self.dim if self.dim < dims[1] else self.dim + 1
    dim = dim if dim <= dims[0] else dim + 1
    x = x.reshape(*shape)
    perm = [dim, dims[1] + 1, dims[0] + 2]
    perm = [i for i in range(min(perm))] + perm
    perm.extend((i for i in range(x.dim()) if i not in perm))
    x = x.permute(*perm)
    shape = _shape
    shape[self.dim] *= self.scale_factor ** 2
    shape[self.dims[0]] //= self.scale_factor
    shape[self.dims[1]] //= self.scale_factor
    return x.reshape(*shape)

  def extra_repr(self):
    return f'scale_factor={self.scale_factor}'


class SpaceToDepth(nn.Module):
  def __init__(self, block_size):
    super(SpaceToDepth, self).__init__()
    self.body = SpaceToDim(block_size, dim=1)

  def forward(self, x):
    return self.body(x)


def nd_meshgrid(*size, permute=None):
  _error_msg = ("Permute index must match mesh dimensions, "
                "should have {} indexes but got {}")
  size = np.array(size)
  ranges = []
  for x in size:
    ranges.append(np.linspace(-1, 1, x))
  mesh = np.stack(np.meshgrid(*ranges, indexing='ij'))
  if permute is not None:
    if len(permute) != len(size):
      raise ValueError(_error_msg.format(len(size), len(permute)))
    mesh = mesh[permute]
  return mesh.transpose(*range(1, mesh.ndim), 0)


class STN(nn.Module):
  """Spatial transformer network.
    For optical flow based frame warping.
  """

  def __init__(self, mode='bilinear', padding_mode='zeros'):
    super(STN, self).__init__()
    self.mode = mode
    self.padding_mode = padding_mode

  def forward(self, inputs, u, v, normalized=True):
    batch = inputs.shape[0]
    device = inputs.device
    mesh = nd_meshgrid(*inputs.shape[-2:], permute=[1, 0])
    mesh = torch.stack([torch.Tensor(mesh)] * batch)
    # add flow to mesh
    _u, _v = u, v
    if not normalized:
      # flow needs to normalize to [-1, 1]
      h, w = inputs.shape[-2:]
      _u = u / w * 2
      _v = v / h * 2
    flow = torch.stack([_u, _v], dim=-1)
    # assert flow.shape == mesh.shape
    mesh = mesh.to(device)
    mesh += flow
    return F.grid_sample(inputs, mesh,
                         mode=self.mode, padding_mode=self.padding_mode)


class Activation(nn.Module):
  def __init__(self, name, *args, **kwargs):
    super(Activation, self).__init__()
    if name is None:
      self.f = lambda t: t
    self.name = name.lower()
    in_place = kwargs.get('in_place', True)
    if self.name == 'relu':
      self.f = nn.ReLU(in_place)
    elif self.name == 'prelu':
      self.f = nn.PReLU()
    elif self.name in ('lrelu', 'leaky', 'leakyrelu'):
      self.f = nn.LeakyReLU(*args, inplace=in_place)
    elif self.name == 'tanh':
      self.f = nn.Tanh()
    elif self.name == 'sigmoid':
      self.f = nn.Sigmoid()


class _UpsampleNearest(nn.Module):
  def __init__(self, scale):
    super(_UpsampleNearest, self).__init__()
    self.scale = scale

  def forward(self, x, scale=None):
    scale = scale or self.scale
    return F.interpolate(x, scale_factor=scale, align_corners=False)


class _UpsampleLinear(nn.Module):
  def __init__(self, scale):
    super(_UpsampleLinear, self).__init__()
    self._mode = ('linear', 'bilinear', 'trilinear')
    self.scale = scale

  def forward(self, x, scale=None):
    scale = scale or self.scale
    mode = self._mode[x.dim() - 3]
    return F.interpolate(x, scale_factor=scale, mode=mode, align_corners=False)


class Upsample(nn.Module):
  def __init__(self, channel, scale, method='ps', name='Upsample', **kwargs):
    super(Upsample, self).__init__()
    self.name = name
    self.channel = channel
    self.scale = scale
    self.method = method.lower()
    self.kernel_size = kwargs.get('kernel_size', 3)

    _allowed_methods = ('ps', 'nearest', 'deconv', 'linear')
    assert self.method in _allowed_methods
    act = kwargs.get('activation')

    samplers = []
    while scale > 1:
      if scale % 2 == 1 or scale == 2:
        samplers.append(self.upsampler(self.method, scale))
        break
      else:
        samplers.append(self.upsampler(self.method, 2, act))
        scale //= 2
    self.body = nn.Sequential(*samplers)

  def upsampler(self, method, scale, activation=None):
    body = []
    k = self.kernel_size
    if method == 'ps':
      p = k // 2  # padding
      s = 1  # strides
      body = [nn.Conv2d(self.channel, self.channel * scale * scale, k, s, p),
              nn.PixelShuffle(scale)]
      if activation:
        body.insert(1, Activation(activation))
    if method == 'deconv':
      q = k % 2  # output padding
      p = (k + q) // 2 - 1  # padding
      s = scale  # strides
      body = [nn.ConvTranspose2d(self.channel, self.channel, k, s, p, q)]
      if activation:
        body.insert(1, Activation(activation))
    if method == 'nearest':
      body = [_UpsampleNearest(scale)]
      if activation:
        body.insert(1, Activation(activation))
    if method == 'linear':
      body = [_UpsampleLinear(scale)]
      if activation:
        body.insert(1, Activation(activation))
    return nn.Sequential(*body)

  def forward(self, inputs):
    return self.body(inputs)

  def extra_repr(self):
    return f"{self.name}: scale={self.scale}"


class FNet(nn.Module):
  def __init__(self, channel, gain=32, f=32, n_layer=3):
    super(FNet, self).__init__()
    layers = []
    in_c = channel * 2
    for i in range(n_layer):
      layers += [nn.Conv2d(in_c, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
      layers += [nn.Conv2d(f, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
      layers += [nn.MaxPool2d(2)]
      in_c = f
      f *= 2
    for i in range(n_layer):
      layers += [nn.Conv2d(in_c, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
      layers += [nn.Conv2d(f, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
      layers += [Upsample(f, scale=2, method='linear')]
      in_c = f
      f //= 2
    layers += [nn.Conv2d(in_c, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
    layers += [nn.Conv2d(f, 2, 3, 1, 1), nn.Tanh()]
    self.body = nn.Sequential(*layers)
    self.gain = gain

  def forward(self, *inputs):
    x = torch.cat(inputs, dim=1)
    x = self.body(x) * self.gain
    # print(max_v)
    return x


class RB(nn.Module):
  def __init__(self, channel):
    super(RB, self).__init__()
    conv1 = nn.Conv2d(channel, channel, 3, 1, 1)
    conv2 = nn.Conv2d(channel, channel, 3, 1, 1)
    self.body = nn.Sequential(conv1, nn.ReLU(True), conv2)

  def forward(self, x):
    return x + self.body(x)


class SRNet(nn.Module):
  def __init__(self, channel, scale, n_rb=10):
    super(SRNet, self).__init__()
    rbs = [RB(64) for _ in range(n_rb)]
    entry = [nn.Conv2d(channel * (scale ** 2 + 1), 64, 3, 1, 1), nn.ReLU(True)]
    up = Upsample(64, scale, method='ps')
    out = nn.Conv2d(64, channel, 3, 1, 1)
    self.body = nn.Sequential(*entry, *rbs, up, out)

  def forward(self, *inputs):
    x = torch.cat(inputs, dim=1)
    return self.body(x)
