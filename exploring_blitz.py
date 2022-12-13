# %%

import matplotlib.pyplot as plt 
from math import sin
import numpy as np
import enlighten

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary as torch_summary
from blitz.modules import BayesianLinear
from blitz.losses import kl_divergence_from_nn as b_kl_loss
from blitz.utils import variational_estimator
from blitz.modules.base_bayesian_module import BayesianModule

import os 
os.chdir(r"/home/ted/Desktop/triple_t_maze")

from utils import init_weights, device, dkl

def x_to_y(x, kind = True, noise = True): 
    if(kind): y = torch.cos(x) + torch.sin(.6*x)
    else:     y = torch.sin(x) + torch.cos(.6*x)
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
xs = torch.tensor(xs).unsqueeze(1)             ; ys = x_to_y(xs, True, False)

@variational_estimator
class Example(nn.Module):

    def __init__(self):
        super(Example, self).__init__()

        self.lin = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU())
        
        self.bayes = BayesianLinear(64, 1)
        
        self.lin.apply(init_weights)
        self.bayes.apply(init_weights)
        self.to(device)
                
        print("\n\n")
        print(self)
        print()
        print(torch_summary(self, (1,1)))
        print("\n\n")
        
        """

        print("Bayes lin:")
        print("weight mu:\t {}".format(self.bayes.weight_mu.shape))
        print("weight rho:\t {}".format(self.bayes.weight_rho.shape))
        print("bias mu:\t {}".format(self.bayes.bias_mu.shape))
        print("bias rho:\t {}".format(self.bayes.bias_rho.shape))

        print("\nweight sampler (TrainableRandomDistribution):")
        print("mu:\t {}".format(self.bayes.weight_sampler.mu.shape))
        print("rho:\t {}".format(self.bayes.weight_sampler.rho.shape))
        print("sigma:\t {}".format(self.bayes.weight_sampler.sigma.shape))
        print("w:\t {}".format(self.bayes.weight_sampler.w.shape))
        print("eps_w:\t {}".format(self.bayes.weight_sampler.eps_w.shape))
        print("log_post sample:\t {}".format(self.bayes.weight_sampler.log_posterior()))

        print("\nbias sampler (TrainableRandomDistribution):")
        print("mu:\t {}".format(self.bayes.bias_sampler.mu.shape))
        print("rho:\t {}".format(self.bayes.bias_sampler.rho.shape))
        print("sigma:\t {}".format(self.bayes.bias_sampler.sigma.shape))
        print("w:\t {}".format(self.bayes.bias_sampler.w.shape))
        print("eps_w:\t {}".format(self.bayes.bias_sampler.eps_w.shape))
        print("log_post:\t {}".format(self.bayes.bias_sampler.log_posterior()))

        print("\nweight prior dist (PriorWeightDistribution):")
        print("sigma1: {}. sigma2: {}.".format(self.bayes.weight_prior_dist.sigma1, self.bayes.weight_prior_dist.sigma2))
        print("dist1:\t {}".format(self.bayes.weight_prior_dist.dist1))
        print("dist2:\t {}".format(self.bayes.weight_prior_dist.dist2))
        print("log_prior:\t {}".format(self.bayes.weight_prior_dist.log_prior(self.bayes.weight_sampler.w)))

        print("\nbias prior dist (PriorWeightDistribution):")
        print("sigma1: {}. sigma2: {}.".format(self.bayes.bias_prior_dist.sigma1, self.bayes.bias_prior_dist.sigma2))
        print("dist1:\t {}".format(self.bayes.bias_prior_dist.dist1))
        print("dist2:\t {}".format(self.bayes.bias_prior_dist.dist2))
        print("log_prior:\t {}".format(self.bayes.bias_prior_dist.log_prior(self.bayes.bias_sampler.w)))
        print("\n\n")
        """

    def forward(self, x):
        x = x.to(device)
        x = self.lin(x)
        x = self.bayes(x)
        return(x)
    
    def weights(self):
        weight_mu = [] ; weight_rho = []
        bias_mu = [] ;   bias_rho = []
        for module in self.modules():
            if isinstance(module, (BayesianModule)):
                weight_mu.append(module.weight_sampler.mu.clone().flatten())
                weight_rho.append(module.weight_sampler.rho.clone().flatten())
                bias_mu.append(module.bias_sampler.mu.clone().flatten()) 
                bias_rho.append(module.bias_sampler.rho.clone().flatten())
        return(
            torch.cat(weight_mu, -1),
            torch.cat(weight_rho, -1),
            torch.cat(bias_mu, -1),
            torch.cat(bias_rho, -1))
        
    def means_stds(self):
        weights = self.weights()
        return(
            torch.mean(weights[0]).item(),
            torch.log1p(torch.exp(torch.mean(weights[1]))).item(),
            torch.mean(weights[2]).item(),
            torch.log1p(torch.exp(torch.mean(weights[3]))).item())
        

example = Example()
opt = optim.Adam(params=example.parameters(), lr=.005, weight_decay=0) 



mse_losses = []
kl_losses = []
losses = []
dkls = []
weights = []
changes = []



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
    mse_loss = 0 
    kl_loss  = 0
    sample_size = 2
    for _ in range(sample_size):
        pred = example(Xs) 
        mse_loss += F.mse_loss(pred, Ys.to(device))
        kl_loss  += b_kl_loss(example) * .001
    mse_loss /= sample_size
    kl_loss  /= sample_size
    loss = mse_loss + kl_loss
    if(source == "train"):
        mse_losses.append(mse_loss.item())
        kl_losses.append(kl_loss.item())
        losses.append(loss.item())
        
        weights_before = example.weights()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        weights_after = example.weights()
        
        weight_change = dkl(weights_after[0], weights_after[1], weights_before[0], weights_before[1]) + \
            dkl(weights_after[2], weights_after[3], weights_before[2], weights_before[3])
                
        dkls.append(weight_change.item())
        weights.append(example.means_stds())
        
    return(pred)

def plot(train_pred, test_pred, title = "", means = None, stds = None):
    plt.figure(figsize=(10,10))
    plt.ylim((-3, 3))
    plt.xlim((xs[0]-3, xs[-1]+3))
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
    #plt.savefig("saves/x_y_" + title + ".png")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(10,10))
    for change in changes:
        plt.axvline(change, color = "black", alpha = .2)
    plt.plot(np.log(np.array(mse_losses)), color = "blue", alpha = .3, label = "MSE")
    plt.plot(np.log(np.array(kl_losses)), color = "red", alpha = .3, label = "KL")
    plt.plot(np.log(np.array(kl_losses) + np.array(mse_losses)), color = "black", alpha = .3, label = "Both")
    plt.legend()
    plt.title("Losses")
    #plt.savefig("saves/loss_" + title + ".png")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(10,10))
    for change in changes:
        plt.axvline(change, color = "black", alpha = .2)
    plt.plot(np.log(np.array(dkls)), color = "green", alpha = .3, label = "DKL")
    plt.legend()
    plt.title("KL(q(w|D)||q(w))")
    #plt.savefig("saves/dkl" + title + ".png")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(10,10))
    for change in changes:
        plt.axvline(change, color = "black", alpha = .2)
    plt.plot(np.log(np.array(dkls) / np.array(losses)), color = "blue", alpha = .3, label = "KL / Losses")
    plt.plot(np.log(np.array(losses) / np.array(dkls)), color = "red",  alpha = .3, label = "Losses / KL")
    plt.legend()
    plt.title("Difference")
    #plt.savefig("saves/loss_" + title + ".png")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(10,10))
    for change in changes:
        plt.axvline(change, color = "black", alpha = .2)
    plt.plot([w[0] for w in weights], color = "blue", alpha = .3, label = "Average weight mean")
    plt.plot([w[2] for w in weights], color = "red", alpha = .3, label = "Average bias mean")
    plt.legend()
    plt.title("Weights")
    #plt.savefig("saves/means_" + title + ".png")
    plt.show()
    plt.close()
    
    plt.figure(figsize=(10,10))
    for change in changes:
        plt.axvline(change, color = "black", alpha = .2)
    plt.plot([w[1] for w in weights], color = "blue", alpha = .3, label = "Average weight std")
    plt.plot([w[3] for w in weights], color = "red", alpha = .3, label = "Average bias std")
    plt.legend()
    plt.title("Weights")
    #plt.savefig("saves/stds_" + title + ".png")
    plt.show()
    plt.close()
    

    
epochs = 100000
change_time = 4333
kind = True
manager = enlighten.Manager()
#E = manager.counter(total = 1000, desc = "Epochs:", unit = "ticks", color = "blue")
for i in range(1, epochs+1):
    #E.update()
    if(i % change_time == 0): 
        kind = not kind
        changes.append(i)
        ys = x_to_y(xs, kind, False)
    train_ys = x_to_y(train_xs, kind) ; test_ys = x_to_y(test_xs, kind)
    
    train_pred = epoch("train").detach().cpu() ; test_pred = epoch("test").detach().cpu()

    if(i == 1 or i%1000 == 0 or i == epochs): 
        preds = []
        for _ in range(100):
            pred = epoch("both").detach().cpu()
            preds.append(pred)
        preds = torch.cat(preds, dim = 1)
        means = torch.mean(preds, dim = 1)
        stds = torch.std(preds, dim = 1)
        plot(train_pred, test_pred, "Epoch {}".format(str(i).zfill(10)), means, stds)
        print("\n\n\n\n\n")





# %%

