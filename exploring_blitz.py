# %%

import matplotlib.pyplot as plt 
from math import sin

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary as torch_summary
from blitz.modules import BayesianLinear
from blitz.losses import kl_divergence_from_nn as b_kl_loss

import os 
os.chdir(r"/home/ted/Desktop/triple_t_maze")

from utils import init_weights, device

def x_to_y(x, noise = True): 
    y = torch.cos(x) + torch.sin(.6*x)
    if(noise): y += torch.normal(torch.zeros(x.shape), .3 * torch.ones(x.shape))
    return(y)

length = 175
off_zero = 500
smooth = 50

xs       = [i/smooth for i in range(-5*length//2 + off_zero, 5*length//2  + off_zero)]
test_xs  = xs[0*len(xs)//5 : 1*len(xs)//5] + xs[2*len(xs)//5 : 3*len(xs)//5] + xs[4*len(xs)//5 : 5*len(xs)//5]
train_xs = xs[1*len(xs)//5 : 2*len(xs)//5] + xs[3*len(xs)//5 : 4*len(xs)//5]

train_xs = torch.tensor(train_xs).unsqueeze(1) ; train_ys = x_to_y(train_xs)
test_xs  = torch.tensor(test_xs).unsqueeze(1)  ; test_ys = x_to_y(test_xs)
xs = torch.tensor(xs).unsqueeze(1)             ; ys = x_to_y(xs, False)

class Example(nn.Module):

    def __init__(self):
        super(Example, self).__init__()

        self.lin = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(),
            BayesianLinear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1))
        
        self.lin.apply(init_weights)
        self.to(device)
        
        print("\n\n")
        print(self)
        print()
        print(torch_summary(self, (1,1)))
        print("\n\n")

        print("Bayes lin:")
        print("weight mu:\t {}".format(self.lin[2].weight_mu.shape))
        print("weight rho:\t {}".format(self.lin[2].weight_rho.shape))
        print("bias mu:\t {}".format(self.lin[2].bias_mu.shape))
        print("bias rho:\t {}".format(self.lin[2].bias_rho.shape))

        print("\nweight sampler (TrainableRandomDistribution):")
        print("mu:\t {}".format(self.lin[2].weight_sampler.mu.shape))
        print("rho:\t {}".format(self.lin[2].weight_sampler.rho.shape))
        print("sigma:\t {}".format(self.lin[2].weight_sampler.sigma.shape))
        print("w:\t {}".format(self.lin[2].weight_sampler.w.shape))
        print("eps_w:\t {}".format(self.lin[2].weight_sampler.eps_w.shape))
        print("log_post sample:\t {}".format(self.lin[2].weight_sampler.log_posterior()))

        print("\nbias sampler (TrainableRandomDistribution):")
        print("mu:\t {}".format(self.lin[2].bias_sampler.mu.shape))
        print("rho:\t {}".format(self.lin[2].bias_sampler.rho.shape))
        print("sigma:\t {}".format(self.lin[2].bias_sampler.sigma.shape))
        print("w:\t {}".format(self.lin[2].bias_sampler.w.shape))
        print("eps_w:\t {}".format(self.lin[2].bias_sampler.eps_w.shape))
        print("log_post:\t {}".format(self.lin[2].bias_sampler.log_posterior()))

        print("\nweight prior dist (PriorWeightDistribution):")
        print("sigma1: {}. sigma2: {}.".format(self.lin[2].weight_prior_dist.sigma1, self.lin[2].weight_prior_dist.sigma2))
        print("dist1:\t {}".format(self.lin[2].weight_prior_dist.dist1))
        print("dist2:\t {}".format(self.lin[2].weight_prior_dist.dist2))
        print("log_prior:\t {}".format(self.lin[2].weight_prior_dist.log_prior(self.lin[2].weight_sampler.w)))

        print("\nbias prior dist (PriorWeightDistribution):")
        print("sigma1: {}. sigma2: {}.".format(self.lin[2].bias_prior_dist.sigma1, self.lin[2].bias_prior_dist.sigma2))
        print("dist1:\t {}".format(self.lin[2].bias_prior_dist.dist1))
        print("dist2:\t {}".format(self.lin[2].bias_prior_dist.dist2))
        print("log_prior:\t {}".format(self.lin[2].bias_prior_dist.log_prior(self.lin[2].bias_sampler.w)))
        print("\n\n")

    def forward(self, x):
        x = x.to(device)
        x = self.lin(x)
        return(x.cpu())

example = Example()
opt = optim.Adam(params=example.parameters(), lr=.001) 


def dkl(mu_1, rho_1, mu_2, rho_2):
    sigma_1 = torch.pow(torch.log1p(torch.exp(rho_1)), 2)
    sigma_2 = torch.pow(torch.log1p(torch.exp(rho_2)), 2)
    term_1 = torch.pow(mu_2 - mu_1, 2) / sigma_2 
    term_2 = sigma_1 / sigma_2 
    term_3 = torch.log(term_2)
    return((.5 * (term_1 + term_2 - term_3 - 1)).sum())

def epoch(source):
    if(source == "test"):  example.eval()  ; Xs = test_xs  ; Ys = test_ys 
    if(source == "train"): example.train() ; Xs = train_xs ; Ys = train_ys
    if(source == "both"):  example.eval()  ; Xs = xs       ; Ys = ys
    pred = example(Xs) 
    if(source == "train"):
        mse_loss = F.mse_loss(pred, Ys)
        kl_loss  = .0001 * b_kl_loss(example)
        #print("MSE: {}. KL: {}.".format(mse_loss.item(), kl_loss.item()))
        loss = mse_loss + kl_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    return(pred)

def plot(train_pred, test_pred, title = "", means = None, stds = None):
    plt.figure(figsize=(10,10))
    plt.ylim((-3, 3))
    if(means != None):
        for s, a in [(10, .2)]:
            plt.fill_between(
                xs.squeeze(1), 
                means + s*stds,
                means - s*stds,
                color = "black", alpha = a, linewidth = 0)
    plt.plot(xs, ys, color = "black", alpha = 1)
    plt.scatter(train_xs, train_ys, color = "blue", alpha = .3, label = "Available for Training")
    plt.scatter(test_xs,  test_ys,  color = "red",  alpha = .3, label = "Unavailable")
    plt.scatter(train_xs, train_pred, color = "blue", alpha = .6)
    plt.scatter(test_xs,  test_pred,  color = "red",  alpha = .6)
    plt.title("{}".format(title))
    plt.legend()
    plt.show()
    plt.close()

epochs = 1000000
for i in range(1, epochs+1):
    train_ys = x_to_y(train_xs) ; test_ys = x_to_y(test_xs)
    
    weights_before = example.lin[2].weight_sampler.mu.clone(), example.lin[2].weight_sampler.rho.clone(), example.lin[2].bias_sampler.mu.clone(), example.lin[2].bias_sampler.rho.clone()
    train_pred = epoch("train").detach() ; test_pred = epoch("test").detach()
    weights_after = example.lin[2].weight_sampler.mu.clone(), example.lin[2].weight_sampler.rho.clone(), example.lin[2].bias_sampler.mu.clone(), example.lin[2].bias_sampler.rho.clone()

    print(dkl(weights_before[0], weights_before[1], 
              weights_after[0], weights_after[1]))

    if(i == 1 or i%1000 == 0 or i == epochs): 
        preds = []
        for _ in range(100):
            pred = epoch("both").detach() 
            preds.append(pred)
        preds = torch.cat(preds, dim = 1)
        means = torch.mean(preds, dim = 1)
        stds = torch.std(preds, dim = 1)
        plot(train_pred, test_pred, "Epoch {}".format(i), means, stds)
        print("\n\n\n\n\n")





# %%

