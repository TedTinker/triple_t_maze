#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from math import log
from random import choice

from utils import args, plot_curiosity, device, new_text
from buffer import RecurrentReplayBuffer
from models import Transitioner, Actor, Critic



class Agent:
    
    def __init__(self, action_prior="normal", args = args):
        
        self.args = args
        self.steps = 0
        self.action_size = 2
        
        self.target_entropy = self.args.target_entropy # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=self.args.alpha_lr) 
        self._action_prior = action_prior
        
        self.eta = 1
        self.log_eta = torch.tensor([0.0], requires_grad=True)
        self.eta_optimizer = optim.Adam(params=[self.log_eta], lr=self.args.eta_lr) 
        
        self.transitioner = Transitioner()
        self.trans_optimizer = optim.Adam(self.transitioner.parameters(), lr=self.args.trans_lr)     
                           
        self.actor = Actor()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)     
        
        self.critic1 = Critic()
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.args.critic_lr, weight_decay=0)
        self.critic1_target = Critic()
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic()
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.args.critic_lr, weight_decay=0) 
        self.critic2_target = Critic()
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.restart_memory()
        
    def restart_memory(self):
        self.memory = RecurrentReplayBuffer(self.args)

    def act(self, image, speed, hidden = None):
        encoded, hidden = self.transitioner.just_encode(image.unsqueeze(0), speed.unsqueeze(0), hidden)
        action = self.actor.get_action(encoded).detach()
        return action, hidden
    
    def learn(self, batch_size, iterations, num = -1, plot = False):
        if(iterations != 1):
            losses = []; extrinsic = []; intrinsic_curiosity = []; intrinsic_entropy = []
            for i in range(iterations): 
                l, e, ic, ie = self.learn(batch_size, 1, num = i, plot = plot)
                losses.append(l); extrinsic.append(e)
                intrinsic_curiosity.append(ic); intrinsic_entropy.append(ie)
            losses = np.concatenate(losses)
            extrinsic = [e for e in extrinsic if e != None]
            intrinsic_curiosity = [e for e in intrinsic_curiosity if e != None]
            intrinsic_entropy = [e for e in intrinsic_entropy if e != None]
            try: extrinsic = sum(extrinsic)/len(extrinsic)
            except: extrinsic = None
            try: intrinsic_curiosity = sum(intrinsic_curiosity)/len(intrinsic_curiosity)
            except: intrinsic_curiosity = None
            try: intrinsic_entropy = sum(intrinsic_entropy)/len(intrinsic_entropy)
            except: intrinsic_entropy = None
            return(losses, extrinsic, intrinsic_curiosity, intrinsic_entropy)
                
        self.steps += 1

        images, speeds, actions, rewards, dones, masks = self.memory.sample(batch_size)
        
        image_masks = torch.tile(masks.unsqueeze(-1).unsqueeze(-1), (self.args.image_size, self.args.image_size, 4))
        
        speeds = (speeds - self.args.min_speed) / (self.args.max_speed - self.args.min_speed)
        speeds = (speeds*2)-1
                            
        # Train transitioner
        pred_next_images, pred_next_speeds = self.transitioner(images[:,:-1].detach(), speeds[:,:-1].detach(), actions.detach())
        trans_loss_image = F.mse_loss(pred_next_images*image_masks.detach(), images[:,1:]*image_masks.detach())
        trans_loss_speed = F.mse_loss(pred_next_speeds*masks.detach(), speeds[:,1:]*masks.detach())
        trans_loss = trans_loss_image + trans_loss_speed
        self.trans_optimizer.zero_grad()
        trans_loss.backward()
        self.trans_optimizer.step()
                
        # Get encodings for other modules
        with torch.no_grad():
            encoded, _ = self.transitioner.just_encode(images.detach(), speeds.detach())
            next_encoded = encoded[:,1:]
            encoded = encoded[:,:-1]
            
        if(self.args.eta == None):
            curiosity = self.eta * self.transitioner.DKL(
                images[:,:-1], speeds[:,:-1], actions,
                images[:,1:], speeds[:,1:], masks)
            self.eta = self.eta * self.args.eta_rate
        else:
            curiosity = self.args.eta * self.transitioner.DKL(
                images[:,:-1], speeds[:,:-1], actions,
                images[:,1:], speeds[:,1:], masks)
            self.args.eta = self.args.eta * self.args.eta_rate
        
        plot = True if num == 0 and plot else False
        if(plot):
            """
            for _ in range(3):
                try:
                    batch_num = choice([i for i in range(self.args.batch_size)])
                    step_num =  choice([i for i in range(self.args.max_steps) if masks[batch_num, i] == 1])
                    image_plot = (images[batch_num,step_num,:,:,:-1].cpu().detach() + 1) / 2
                    pred_next_image_plot = (pred_next_images[batch_num,step_num,:,:,:-1].cpu().detach() + 1) / 2
                    next_image_plot = (images[batch_num,step_num+1,:,:,:-1].cpu().detach() + 1) / 2
                    yaw = actions[batch_num,step_num,0].item() * self.args.max_yaw_change
                    yaw = round(degrees(yaw))
                    spe = self.args.min_speed + ((actions[batch_num,step_num,1].item() + 1)/2) * \
                        (self.args.max_speed - self.args.min_speed)
                    spe = round(spe)
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                    ax1.title.set_text("Before")
                    ax1.imshow(image_plot)
                    ax1.axis('off')
                    ax2.title.set_text("Prediction")
                    ax2.imshow(pred_next_image_plot)
                    ax2.axis('off')
                    ax3.title.set_text("After")
                    ax3.imshow(next_image_plot)
                    ax3.axis('off')
                    fig.suptitle("Step {}: Action: {} degrees, {} speed".format(step_num, yaw, spe))
                    fig.tight_layout()
                    fig.subplots_adjust(top=1.2)
                    plt.show()
                    plt.close()
                except:
                    print("Something went weird trying to plot transitions.")
            """
            plot_curiosity(rewards.detach(), curiosity.detach(), masks.detach())
            
        extrinsic = torch.mean(rewards*masks.detach()).item()
        intrinsic_curiosity = torch.mean(curiosity*masks.detach()).item()
        rewards = torch.cat([rewards, curiosity], -1)
                
        # Train critics
        next_action, log_pis_next = self.actor.evaluate(next_encoded.detach())
        Q_target1_next = self.critic1_target(next_encoded.detach(), next_action.detach())
        Q_target2_next = self.critic2_target(next_encoded.detach(), next_action.detach())
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        log_pis_next = torch.mean(log_pis_next, -1).unsqueeze(-1)
        if self.args.alpha == None: Q_targets = rewards.cpu() + (self.args.gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.cpu()))
        else:                       Q_targets = rewards.cpu() + (self.args.gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.args.alpha * log_pis_next.cpu()))
        
        Q_1 = self.critic1(encoded, actions).cpu()
        critic1_loss = 0.5*F.mse_loss(Q_1*masks.detach().cpu(), Q_targets.detach()*masks.detach().cpu())
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        Q_2 = self.critic2(encoded, actions).cpu()
        critic2_loss = 0.5*F.mse_loss(Q_2*masks.detach().cpu(), Q_targets.detach()*masks.detach().cpu())
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Train alpha
        if self.args.alpha == None:
            actions_pred, log_pis = self.actor.evaluate(encoded.detach())
            alpha_loss = -(self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu())*masks.detach().cpu()
            alpha_loss = alpha_loss.sum() / masks.sum()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = torch.exp(self.log_alpha) 
            
        # Train eta
        if(self.args.eta == None):
            """
            eta_loss = "I don't know yet"
            self.eta_optimizer.zero_grad()
            eta_loss.backward()
            self.eta_optimizer.step()
            self.eta = torch.exp(self.log_eta) 
            """
            self.eta = self.eta
            
    
        # Train actor
        if self.steps % self.args.d == 0:
            if self.args.alpha == None: alpha = self.alpha 
            else:                       
                alpha = self.args.alpha
                actions_pred, log_pis = self.actor.evaluate(encoded.detach())
                alpha_loss = None

            if self._action_prior == "normal":
                loc = torch.zeros(self.action_size, dtype=torch.float64)
                scale_tril = torch.tensor([[1, 0], [1, 1]], dtype=torch.float64)
                policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
                policy_prior_log_probs = policy_prior.log_prob(actions_pred.cpu()).unsqueeze(-1)
            elif self._action_prior == "uniform":
                policy_prior_log_probs = 0.0
            Q = torch.min(
                self.critic1(encoded.detach(), actions_pred), 
                self.critic2(encoded.detach(), actions_pred)).sum(-1).unsqueeze(-1)
            intrinsic_entropy = torch.mean((alpha * log_pis.cpu())*masks.detach().cpu()).item()
            actor_loss = (alpha * log_pis.cpu() - policy_prior_log_probs - Q.cpu())*masks.detach().cpu()
            actor_loss = actor_loss.sum() / masks.sum()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.critic1, self.critic1_target, self.args.tau)
            self.soft_update(self.critic2, self.critic2_target, self.args.tau)
            
        else:
            intrinsic_entropy = None
            alpha_loss = None
            actor_loss = None
        
        if(trans_loss != None): trans_loss = log(trans_loss.item())
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(critic1_loss != None): critic1_loss = log(critic1_loss.item())
        if(critic2_loss != None): critic2_loss = log(critic2_loss.item())
        losses = np.array([[trans_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss]])
        
        try:    intrinsic_curiosity = log(intrinsic_curiosity)
        except: pass
        
        return(losses, extrinsic, intrinsic_curiosity, intrinsic_entropy)
                     
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def state_dict(self):
        return(
            self.transitioner.state_dict(),
            self.actor.state_dict(),
            self.critic1.state_dict(),
            self.critic1_target.state_dict(),
            self.critic2.state_dict(),
            self.critic2_target.state_dict())

    def load_state_dict(self, state_dict):
        self.transitioner.load_state_dict(state_dict[0])
        self.actor.load_state_dict(state_dict[1])
        self.critic1.load_state_dict(state_dict[2])
        self.critic1_target.load_state_dict(state_dict[3])
        self.critic2.load_state_dict(state_dict[4])
        self.critic2_target.load_state_dict(state_dict[5])
        self.memory = RecurrentReplayBuffer(self.args)

    def eval(self):
        self.transitioner.eval()
        self.actor.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

    def train(self):
        self.transitioner.train()
        self.actor.train()
        self.critic1.train()
        self.critic1_target.train()
        self.critic2.train()
        self.critic2_target.train()
        
new_text("agent.py loaded.")
