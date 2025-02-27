# 1) The code relevant to Score-based models are from the paper "Score-Based Generative Modeling through Stochastic Differential Equations" and tweaked by the author in this work.
# 2) The code relevant to FNO are from the paper "Fourier Neural Operator for Parametric Partial Differential Equations" and tweaked by the author in this work.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

################################
######### SDE setup ############
################################
# Set up VE SDE for diffusion process
def marginal_prob_std(t, sigma, device_):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.
      device_: The device to use.

    Returns:
      The standard deviation.
    """
    t = torch.tensor(t, device=device_)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma, device_):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.
      device_: The device to use.

    Returns:
      The vector of diffusion coefficients.
    """
    return torch.tensor(sigma ** t, device=device_)

################################
######## SBM Model setup #######
################################

# Diffusion process time step encoding
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# Dense layer for encoding time steps
class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None, None]

# 2d Fourier layer
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

# 2d FNO Score-based Models
class FNO2d(nn.Module):
    def __init__(self, marginal_prob_std, modes1, modes2, width, L = 1, embed_dim = 256):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.L = L

        self.fc0 = nn.Linear(3, self.width)
        self.fc0_sparse = nn.Linear(3, self.width)
        self.fc0_w = nn.Linear(3, self.width)

        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        self.conv0_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.conv0_w = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1_w = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2_w = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3_w = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.conv0_sparse = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1_sparse = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2_sparse = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3_sparse = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0_x = nn.Conv2d(self.width, self.width, 1)
        self.w1_x = nn.Conv2d(self.width, self.width, 1)
        self.w2_x = nn.Conv2d(self.width, self.width, 1)
        self.w3_x = nn.Conv2d(self.width, self.width, 1)

        self.w0_w = nn.Conv2d(self.width, self.width, 1)
        self.w1_w = nn.Conv2d(self.width, self.width, 1)
        self.w2_w = nn.Conv2d(self.width, self.width, 1)
        self.w3_w = nn.Conv2d(self.width, self.width, 1)

        self.w0_sparse = nn.Conv2d(self.width, self.width, 1)
        self.w1_sparse = nn.Conv2d(self.width, self.width, 1)
        self.w2_sparse = nn.Conv2d(self.width, self.width, 1)
        self.w3_sparse = nn.Conv2d(self.width, self.width, 1)

        self.dense0 = Dense(embed_dim, self.width)

        # Define a transformation network for the concatenated output
        self.transformation_net = nn.Sequential(
            nn.Conv2d(width*3, width*2, 1),
            nn.GELU(),
            nn.Conv2d(width*2, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
            nn.GELU()
        )

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, t, x, w, x_sparse):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1) # (N, X, Y) --> (N, X, Y, 1)
        x_sparse = x_sparse.reshape(x_sparse.shape[0], x_sparse.shape[1], x_sparse.shape[2], 1) # (N, X, Y) --> (N, X, Y, 1)
        w = w.reshape(w.shape[0], w.shape[1], w.shape[2], 1) # (N, X, Y) --> (N, X, Y, 1)

        grid = self.get_grid(x.shape, x.device)
        sparse_grid = self.get_grid(x_sparse.shape, x_sparse.device)

        x = torch.cat((x, grid), dim=-1)
        x_sparse = torch.cat((x_sparse, sparse_grid), dim=-1)
        w = torch.cat((w, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x_sparse = self.fc0_sparse(x_sparse)
        x_sparse = x_sparse.permute(0, 3, 1, 2)

        w = self.fc0_w(w)
        w = w.permute(0, 3, 1, 2)

        embed = self.act(self.embed(t))
        t_embed = self.dense0(embed).squeeze(-1)

        x1 = self.conv0_x(x)
        x2 = self.w0_x(x)
        x = x1 + x2 + t_embed
        x = F.gelu(x)

        x1 = self.conv1_x(x)
        x2 = self.w1_x(x)
        x = x1 + x2 + t_embed
        x = F.gelu(x)

        x1 = self.conv2_x(x)
        x2 = self.w2_x(x)
        x = x1 + x2 + t_embed
        x = F.gelu(x)

        x1 = self.conv3_x(x)
        x2 = self.w3_x(x)
        x = x1 + x2 + t_embed

        w1 = self.conv0_w(w)
        w2 = self.w0_w(w)
        w = w1 + w2
        w = F.gelu(w)

        w1 = self.conv1_w(w)
        w2 = self.w1_w(w)
        w = w1 + w2
        w = F.gelu(w)

        w1 = self.conv2_w(w)
        w2 = self.w2_w(w)
        w = w1 + w2
        w = F.gelu(w)

        w1 = self.conv3_w(w)
        w2 = self.w3_w(w)
        w = w1 + w2

        x_sparse1 = self.conv0_sparse(x_sparse)
        x_sparse2 = self.w0_sparse(x_sparse)
        x_sparse = x_sparse1 + x_sparse2
        x_sparse = F.gelu(x_sparse)

        x_sparse_1 = self.conv1_sparse(x_sparse)
        x_sparse_2 = self.w1_sparse(x_sparse)
        x_sparse = x_sparse_1 + x_sparse_2
        x_sparse = F.gelu(x_sparse)

        x_sparse_1 = self.conv2_sparse(x_sparse)
        x_sparse_2 = self.w2_sparse(x_sparse)
        x_sparse = x_sparse_1 + x_sparse_2
        x_sparse = F.gelu(x_sparse)

        x_sparse_1 = self.conv3_sparse(x_sparse)
        x_sparse_2 = self.w3_sparse(x_sparse)
        x_sparse = x_sparse_1 + x_sparse_2

        x = torch.cat((x, w, x_sparse), dim=1)
        x = self.transformation_net(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])

        return x / self.marginal_prob_std(t)[:, None, None] # (N, X, Y, 1) --> (N, X, Y)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.L, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.L, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class FNO2d_NoSparse(nn.Module):
    def __init__(self, marginal_prob_std, modes1, modes2, width, L = 1, embed_dim = 256):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.L = L

        self.fc0 = nn.Linear(3, self.width)
        self.fc0_w = nn.Linear(3, self.width)

        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        self.conv0_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3_x = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.conv0_w = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1_w = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2_w = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3_w = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0_x = nn.Conv2d(self.width, self.width, 1)
        self.w1_x = nn.Conv2d(self.width, self.width, 1)
        self.w2_x = nn.Conv2d(self.width, self.width, 1)
        self.w3_x = nn.Conv2d(self.width, self.width, 1)

        self.w0_w = nn.Conv2d(self.width, self.width, 1)
        self.w1_w = nn.Conv2d(self.width, self.width, 1)
        self.w2_w = nn.Conv2d(self.width, self.width, 1)
        self.w3_w = nn.Conv2d(self.width, self.width, 1)

        self.dense0 = Dense(embed_dim, self.width)

        # Define a transformation network for the concatenated output
        self.transformation_net = nn.Sequential(
            nn.Conv2d(width*2, width*2, 1),  # Reduce dimensionality while combining information
            nn.GELU(),
            nn.Conv2d(width*2, width, 1),  # Further compression to original width channels
            nn.GELU(),
            nn.Conv2d(width, width, 1),  # Optional: another layer to refine features
            nn.GELU()
        )

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, t, x, w):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1) # (N, X, Y) --> (N, X, Y, 1
        w = w.reshape(w.shape[0], w.shape[1], w.shape[2], 1) # (N, X, Y) --> (N, X, Y, 1)

        grid = self.get_grid(x.shape, x.device)

        x = torch.cat((x, grid), dim=-1)
        w = torch.cat((w, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        w = self.fc0_w(w)
        w = w.permute(0, 3, 1, 2)

        embed = self.act(self.embed(t))
        t_embed = self.dense0(embed).squeeze(-1)

        x1 = self.conv0_x(x)
        x2 = self.w0_x(x)
        x = x1 + x2 + t_embed
        x = F.gelu(x)

        x1 = self.conv1_x(x)
        x2 = self.w1_x(x)
        x = x1 + x2 + t_embed
        x = F.gelu(x)

        x1 = self.conv2_x(x)
        x2 = self.w2_x(x)
        x = x1 + x2 + t_embed
        x = F.gelu(x)

        x1 = self.conv3_x(x)
        x2 = self.w3_x(x)
        x = x1 + x2 + t_embed

        w1 = self.conv0_w(w)
        w2 = self.w0_w(w)
        w = w1 + w2
        w = F.gelu(w)

        w1 = self.conv1_w(w)
        w2 = self.w1_w(w)
        w = w1 + w2
        w = F.gelu(w)

        w1 = self.conv2_w(w)
        w2 = self.w2_w(w)
        w = w1 + w2
        w = F.gelu(w)

        w1 = self.conv3_w(w)
        w2 = self.w3_w(w)
        w = w1 + w2

        x = torch.cat((x, w), dim=1)
        x = self.transformation_net(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])

        return x / self.marginal_prob_std(t)[:, None, None] # (N, X, Y, 1) --> (N, X, Y)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.L, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.L, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

# Loss function
def loss_fn(model, x, w, x_sparse, marginal_prob_std, eps=1e-5, sparse=True):
  random_t = (torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps) * 2

  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_target = x + z * std[:, None, None]
  if sparse:
    score = model(random_t, perturbed_target, w, x_sparse)
  else:
    score = model(random_t, perturbed_target, w)

  loss = torch.mean(torch.sum((score * std[:, None, None] + z)**2, dim=(1, 2)))

  return loss, score