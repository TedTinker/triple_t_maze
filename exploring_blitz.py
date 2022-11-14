#%%
import torch
from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal
from torchinfo import summary as torch_summary
from blitz.modules import BayesianLinear, BayesianConv2d

import os 
os.chdir(r"/home/ted/Desktop/triple_t_maze")

from utils import args, device, ConstrainedConv2d, delete_these, \
    init_weights, shape_out, flatten_shape



class Example(nn.Module):

    def __init__(self, args):
        super(Example, self).__init__()

        self.lin_1 = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU())
        
        self.bayes_lin = BayesianLinear(32, 64, bias = False)

        self.bayes_conv = BayesianConv2d(
            in_channels = 2,
            out_channels = 4,
            kernel_size = (1,1))

        self.lin_1.apply(init_weights)
        self.bayes_lin.apply(init_weights)
        self.bayes_conv.apply(init_weights)
        self.to(device)

    def forward(self, x):
        x = x.to(device)
        x = self.lin_1(x)
        before_bayes = x
        x_1 = self.bayes_lin(x)
        x = torch.reshape(x, (x.shape[0], 2, 4, 4))
        x_2 = self.bayes_conv(x)
        return(before_bayes, x_1, x_2)



if __name__ == "__main__":

    example = Example(args)

    print("\n\n")
    print(example)
    print()
    print(torch_summary(example, (1,16)))
    
    before_bayes, _, _ = example(torch.rand(1, 16))
    
    real_lin = torch.rand((1, 64))
    real_conv = torch.rand((1, 4, 4, 4))
    
    print("\nlin_1:")
    print(example.lin_1[0].weight.shape)
    
    print("\nBayes lin:")
    print("weight mu:\t {}".format(example.bayes_lin.weight_mu.shape))
    print("weight rho:\t {}".format(example.bayes_lin.weight_rho.shape))
    print("bias mu:\t {}".format(example.bayes_lin.bias_mu.shape))
    print("bias rho:\t {}".format(example.bayes_lin.bias_rho.shape))

    print("\nweight sampler (TrainableRandomDistribution):")
    print("mu:\t {}".format(example.bayes_lin.weight_sampler.mu.shape))
    print("rho:\t {}".format(example.bayes_lin.weight_sampler.rho.shape))
    print("sigma:\t {}".format(example.bayes_lin.weight_sampler.sigma.shape))
    print("w:\t {}".format(example.bayes_lin.weight_sampler.w.shape))
    print("eps_w:\t {}".format(example.bayes_lin.weight_sampler.eps_w.shape))
    print("log_post sample:\t {}".format(example.bayes_lin.weight_sampler.log_posterior()))
    print("log_post correct:\t {}".format(example.bayes_lin.weight_sampler.log_posterior()))
    
    #print("\nbias sampler (TrainableRandomDistribution):")
    #print("mu:\t {}".format(example.bayes_lin.bias_sampler.mu.shape))
    #print("rho:\t {}".format(example.bayes_lin.bias_sampler.rho.shape))
    #print("sigma:\t {}".format(example.bayes_lin.bias_sampler.sigma.shape))
    #print("w:\t {}".format(example.bayes_lin.bias_sampler.w.shape))
    #print("eps_w:\t {}".format(example.bayes_lin.bias_sampler.eps_w.shape))
    #print("log_post:\t {}".format(example.bayes_lin.bias_sampler.log_posterior()))
    
    print("\nweight prior dist (PriorWeightDistribution):")
    print("sigma1: {}. sigma2: {}.".format(example.bayes_lin.weight_prior_dist.sigma1, example.bayes_lin.weight_prior_dist.sigma2))
    print("dist1:\t {}".format(example.bayes_lin.weight_prior_dist.dist1))
    print("dist2:\t {}".format(example.bayes_lin.weight_prior_dist.dist2))
    print("log_prior:\t {}".format(example.bayes_lin.weight_prior_dist.log_prior(example.bayes_lin.weight_sampler.w)))
    
    #print("\nbias prior dist (PriorWeightDistribution):")
    #print("sigma1: {}. sigma2: {}.".format(example.bayes_lin.bias_prior_dist.sigma1, example.bayes_lin.bias_prior_dist.sigma2))
    #print("dist1:\t {}".format(example.bayes_lin.bias_prior_dist.dist1))
    #print("dist2:\t {}".format(example.bayes_lin.bias_prior_dist.dist2))
    #print("log_prior:\t {}".format(example.bayes_lin.bias_prior_dist.log_prior(example.bayes_lin.bias_sampler.w)))
    
    print("\nBayes conv:")
    print("weight mu:\t {}".format(example.bayes_conv.weight_mu.shape))
    print("weight rho:\t {}".format(example.bayes_conv.weight_rho.shape))
    print("bias mu:\t {}".format(example.bayes_conv.bias_mu.shape))
    print("bias rho:\t {}".format(example.bayes_conv.bias_rho.shape))

    print("\nweight sampler (TrainableRandomDistribution):")
    print("mu:\t {}".format(example.bayes_conv.weight_sampler.mu.shape))
    print("rho:\t {}".format(example.bayes_conv.weight_sampler.rho.shape))
    print("sigma:\t {}".format(example.bayes_conv.weight_sampler.sigma.shape))
    print("w:\t {}".format(example.bayes_conv.weight_sampler.w.shape))
    print("eps_w:\t {}".format(example.bayes_conv.weight_sampler.eps_w.shape))
    print("log_post sample:\t {}".format(example.bayes_conv.weight_sampler.log_posterior()))
    print("log_post correct:\t {}".format(example.bayes_conv.weight_sampler.log_posterior()))

    #print("\nbias sampler (TrainableRandomDistribution):")
    #print("mu:\t {}".format(example.bayes_conv.bias_sampler.mu.shape))
    #print("rho:\t {}".format(example.bayes_conv.bias_sampler.rho.shape))
    #print("sigma:\t {}".format(example.bayes_conv.bias_sampler.sigma.shape))
    #print("w:\t {}".format(example.bayes_conv.bias_sampler.w.shape))
    #print("eps_w:\t {}".format(example.bayes_conv.bias_sampler.eps_w.shape))
    #print("log_post:\t {}".format(example.bayes_conv.bias_sampler.log_posterior()))
    
    print("\nweight prior dist (PriorWeightDistribution):")
    print("sigma1: {}. sigma2: {}.".format(example.bayes_conv.weight_prior_dist.sigma1, example.bayes_conv.weight_prior_dist.sigma2))
    print("dist1:\t {}".format(example.bayes_conv.weight_prior_dist.dist1))
    print("dist2:\t {}".format(example.bayes_conv.weight_prior_dist.dist2))
    print("log_prior:\t {}".format(example.bayes_conv.weight_prior_dist.log_prior(example.bayes_conv.weight_sampler.w)))
    
    #print("\nbias prior dist (PriorWeightDistribution):")
    #print("sigma1: {}. sigma2: {}.".format(example.bayes_conv.bias_prior_dist.sigma1, example.bayes_conv.bias_prior_dist.sigma2))
    #print("dist1:\t {}".format(example.bayes_conv.bias_prior_dist.dist1))
    #print("dist2:\t {}".format(example.bayes_conv.bias_prior_dist.dist2))
    #print("log_prior:\t {}".format(example.bayes_conv.bias_prior_dist.log_prior(example.bayes_conv.bias_sampler.w)))
    
    
# %%
import torch 

torch.tensor((10, 100))
# %%
