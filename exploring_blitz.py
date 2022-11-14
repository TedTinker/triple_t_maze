#%%
import torch
from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal
from torchinfo import summary as torch_summary
from blitz.modules import BayesianLinear

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
        
        self.bayes = BayesianLinear(32, 64)

        self.lin_2 = nn.Sequential(
            nn.Linear(64, 1)) 

        self.lin_1.apply(init_weights)
        self.bayes.apply(init_weights)
        self.lin_2.apply(init_weights)
        self.to(device)

    def forward(self, x):
        x = x.to(device)
        x = self.lin_1(x)
        x = self.bayes(x)
        x = self.lin_2(x)
        delete_these(False, x)
        return(x)



if __name__ == "__main__":

    example = Example(args)

    print("\n\n")
    print(example)
    print()
    print(torch_summary(example, (1,16)))
    
    print("\nBayes:")
    print("weight mu:\t {}".format(example.bayes.weight_mu.shape))
    print("weight rho:\t {}".format(example.bayes.weight_rho.shape))
    print("bias mu:\t {}".format(example.bayes.bias_mu.shape))
    print("bias rho:\t {}".format(example.bayes.bias_rho.shape))

    print("\nweight sampler (TrainableRandomDistribution):")
    print("mu:\t {}".format(example.bayes.weight_sampler.mu.shape))
    print("rho:\t {}".format(example.bayes.weight_sampler.rho.shape))
    print("sigma:\t {}".format(example.bayes.weight_sampler.sigma.shape))
    print("w:\t {}".format(example.bayes.weight_sampler.w.shape))
    print("eps_w:\t {}".format(example.bayes.weight_sampler.eps_w.shape))
    print("log_post:\t {}".format(example.bayes.weight_sampler.log_posterior()))
    
    print("\nbias sampler (TrainableRandomDistribution):")
    print("mu:\t {}".format(example.bayes.bias_sampler.mu.shape))
    print("rho:\t {}".format(example.bayes.bias_sampler.rho.shape))
    print("sigma:\t {}".format(example.bayes.bias_sampler.sigma.shape))
    print("w:\t {}".format(example.bayes.bias_sampler.w.shape))
    print("eps_w:\t {}".format(example.bayes.bias_sampler.eps_w.shape))
    print("log_post:\t {}".format(example.bayes.bias_sampler.log_posterior()))
    
    print("\nweight prior dist (PriorWeightDistribution):")
    print("sigma1: {}. sigma2: {}.".format(example.bayes.weight_prior_dist.sigma1, example.bayes.weight_prior_dist.sigma2))
    print("dist1:\t {}".format(example.bayes.weight_prior_dist.dist1))
    print("dist2:\t {}".format(example.bayes.weight_prior_dist.dist2))
    print("log_prior:\t {}".format(example.bayes.weight_prior_dist.log_prior(example.bayes.weight_sampler.w)))
    
    print("\nbias prior dist (PriorWeightDistribution):")
    print("sigma1: {}. sigma2: {}.".format(example.bayes.bias_prior_dist.sigma1, example.bayes.bias_prior_dist.sigma2))
    print("dist1:\t {}".format(example.bayes.bias_prior_dist.dist1))
    print("dist2:\t {}".format(example.bayes.bias_prior_dist.dist2))
    print("log_prior:\t {}".format(example.bayes.bias_prior_dist.log_prior(example.bayes.bias_sampler.w)))
    
    
# %%
import torch 

torch.tensor((10, 100))
# %%
