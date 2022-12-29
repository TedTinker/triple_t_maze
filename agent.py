#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim
from blitz.losses import kl_divergence_from_nn as b_kl_loss

import numpy as np
from math import log
from copy import deepcopy

from utils import args, device, plot_some_predictions, dkl, average_change
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
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=self.args.alpha_lr, weight_decay=0) 
        self._action_prior = action_prior
        
        self.eta = 1
        self.log_eta = torch.tensor([0.0], requires_grad=True)
        self.eta_optimizer = optim.Adam(params=[self.log_eta], lr=self.args.eta_lr, weight_decay=0) 
        
        self.transitioner = Transitioner(self.args)
        self.trans_optimizer = optim.Adam(self.transitioner.parameters(), lr=self.args.trans_lr, weight_decay=0)     
        
        clone_lr = self.args.trans_lr 
        if(self.args.dkl_change_size == "episode" or self.args.dkl_change_size == "step"):
            clone_lr /= self.args.batch_size
        if(self.args.dkl_change_size == "step"):
            clone_lr /= self.args.max_steps
        self.trans_clone = Transitioner(self.args)
        self.opt_clone = optim.Adam(self.trans_clone.parameters(), lr=clone_lr, weight_decay=0)
                           
        self.actor = Actor(self.args)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr, weight_decay=0)     
        
        self.critic1 = Critic(self.args)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.args.critic_lr, weight_decay=0)
        self.critic1_target = Critic(self.args)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(self.args)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.args.critic_lr, weight_decay=0) 
        self.critic2_target = Critic(self.args)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.restart_memory()
        
    def restart_memory(self):
        self.memory = RecurrentReplayBuffer(self.args)

    def act(self, image, speed, prev_action, hidden = None):
        encoded, hidden = self.transitioner.just_encode(
            image.unsqueeze(0), speed.unsqueeze(0), prev_action.unsqueeze(0), hidden)
        action = self.actor.get_action(encoded).detach()
        return action, hidden
    
    def learn(self, batch_size, iterations, num = -1, plot_predictions = False, epoch = 0):
        if(iterations != 1):
            losses = []; extrinsic = []; intrinsic_curiosity = []; intrinsic_entropy = [] ; dkl_changes = [] ; weight_changes = [0,0,0,0]
            for i in range(iterations): 
                l, e, ic, ie, dkl_change, weight_change = self.learn(batch_size, 1, num = i, plot_predictions = plot_predictions)
                losses.append(l); extrinsic.append(e)
                intrinsic_curiosity.append(ic); intrinsic_entropy.append(ie) ; dkl_changes.append(dkl_change)
                weight_changes = [w + nw/iterations for w, nw in zip(weight_changes, weight_change)]
            losses = np.concatenate(losses)
            extrinsic = [e for e in extrinsic if e != None]
            intrinsic_curiosity = [e for e in intrinsic_curiosity if e != None]
            intrinsic_entropy = [e for e in intrinsic_entropy if e != None]
            dkl_change = np.concatenate(dkl_changes)
            try:    extrinsic = sum(extrinsic)/len(extrinsic)
            except: extrinsic = None
            try:    intrinsic_curiosity = sum(intrinsic_curiosity)/len(intrinsic_curiosity)
            except: intrinsic_curiosity = None
            try:    intrinsic_entropy = sum(intrinsic_entropy)/len(intrinsic_entropy)
            except: intrinsic_entropy = None
            return(losses, extrinsic, intrinsic_curiosity, intrinsic_entropy, dkl_change, weight_changes)
                
        self.steps += 1

        images, speeds, actions, rewards, dones, masks = self.memory.sample(batch_size)
        image_masks = torch.tile(masks.unsqueeze(-1).unsqueeze(-1), (self.args.image_size, self.args.image_size, 4))
        speeds = (speeds - self.args.min_speed) / (self.args.max_speed - self.args.min_speed)
        speeds = (speeds*2)-1
        prev_actions = torch.cat([torch.zeros(actions.shape[0], 1, actions.shape[2]), actions], dim = 1)
        
        
                            
        # Train transitioner
        flat_images = images[:,self.args.lookahead:]*image_masks.detach()[:,self.args.lookahead-1:]
        flat_images = flat_images.flatten(2)
        flat_speeds = speeds[:,self.args.lookahead:]*masks.detach()[:,self.args.lookahead-1:]
        flat_real = torch.cat([flat_images, flat_speeds], dim = -1)
        sequential_actions = actions 
        for i in range(self.args.lookahead-1):
            next_actions = torch.cat([actions[:,i+1:], torch.zeros((actions.shape[0], i+1, 2))], dim=1)
            sequential_actions = torch.cat([sequential_actions, next_actions], dim = -1)
        sequential_actions = sequential_actions if self.args.lookahead==1 else sequential_actions[:,:-self.args.lookahead+1]
        
        trans_errors = torch.zeros(rewards.shape)
        dkl_loss = 0
        for _ in range(self.args.sample_elbo):
            pred_next_images, pred_next_speeds, _ = self.transitioner(
                images[:,:-self.args.lookahead].detach(), 
                speeds[:,:-self.args.lookahead].detach(), 
                prev_actions[:,:-self.args.lookahead].detach(), sequential_actions.detach())
            
            flat_pred_images = pred_next_images*image_masks.detach()[:,self.args.lookahead-1:]
            flat_pred_images = flat_pred_images.flatten(2)
            flat_pred_speeds = pred_next_speeds*masks.detach()[:,self.args.lookahead-1:]
            flat_pred_speeds = flat_pred_speeds
            flat_pred = torch.cat([flat_pred_images, flat_pred_speeds], dim = -1)
            
            errors = F.mse_loss(flat_pred, flat_real.detach(), reduction = "none") 
            errors = torch.sum(errors, -1).unsqueeze(-1)
            trans_errors += errors / self.args.sample_elbo
            dkl_loss += self.args.dkl_rate * b_kl_loss(self.transitioner) / self.args.sample_elbo
        mse_loss = trans_errors.sum()
        trans_loss = mse_loss + dkl_loss
        print("\nMSE: {}. KL: {}.\n".format(mse_loss.item(), dkl_loss.item()))
        
        old_state_dict = self.transitioner.state_dict() # For curiosity
        
        weights_before = self.transitioner.weights()
    
        self.trans_optimizer.zero_grad()
        trans_loss.sum().backward()
        self.trans_optimizer.step()
        
        weights_after = self.transitioner.weights()
        dkl_change = dkl(weights_after[0], weights_after[1], weights_before[0], weights_before[1]) + \
            dkl(weights_after[2], weights_after[3], weights_before[2], weights_before[3])
        dkl_changes = torch.tile(dkl_change, rewards.shape)
        
        weight_change = average_change(weights_before, weights_after)
                
    
            
        if(self.args.dkl_change_size == "episode" and self.args.naive_curiosity != "true"):
            dkl_changes = torch.zeros(rewards.shape)
            
            with torch.no_grad():
                encoding_, _ = self.trans_clone.just_encode(
                    images[:,:-self.args.lookahead].detach(), 
                    speeds[:,:-self.args.lookahead].detach(), 
                    prev_actions[:,:-self.args.lookahead].detach())
            
            for episode in range(dkl_changes.shape[0]):
                
                self.trans_clone.load_state_dict(old_state_dict)
                trans_errors_ = torch.zeros(rewards.shape)
                dkl_loss_ = 0
                
                for _ in range(self.args.sample_elbo):
                    pred_next_images_, pred_next_speeds_ = self.trans_clone.after_encode(
                        torch.clone(encoding_[episode]).unsqueeze(0), sequential_actions[episode].detach().unsqueeze(0), True)
                    
                    flat_pred_images_ = pred_next_images_*image_masks.detach()[:,self.args.lookahead-1:]
                    flat_pred_images_ = flat_pred_images_.flatten(2)
                    flat_pred_speeds_ = pred_next_speeds_*masks.detach()[:,self.args.lookahead-1:]
                    flat_pred_speeds_ = flat_pred_speeds_
                    flat_pred_ = torch.cat([flat_pred_images_, flat_pred_speeds_], dim = -1)
                    
                    errors_ = F.mse_loss(flat_pred_, flat_real.detach(), reduction = "none") 
                    errors_ = torch.sum(errors_, -1).unsqueeze(-1)
                    trans_errors_ += errors_ / self.args.sample_elbo
                    dkl_loss_ += self.args.dkl_rate * b_kl_loss(self.trans_clone) / self.args.sample_elbo
                mse_loss_ = trans_errors_.sum()
                trans_loss_ = mse_loss_ + dkl_loss_
                                
                self.opt_clone.zero_grad()
                trans_loss_.sum().backward()
                self.opt_clone.step()
            
                weights_after = self.trans_clone.weights()
                dkl_change = dkl(weights_after[0], weights_after[1], weights_before[0], weights_before[1]) + \
                    dkl(weights_after[2], weights_after[3], weights_before[2], weights_before[3])
                dkl_changes[episode] = dkl_change
            
            
        
        if(self.args.dkl_change_size == "step" and self.args.naive_curiosity != "true"):
            dkl_changes = torch.zeros(rewards.shape)
            
            with torch.no_grad():
                encoding_, _ = self.trans_clone.just_encode(
                    images[:,:-self.args.lookahead].detach(), 
                    speeds[:,:-self.args.lookahead].detach(), 
                    prev_actions[:,:-self.args.lookahead].detach())
            
            for episode in range(dkl_changes.shape[0]):
                for step in range(dkl_changes.shape[1]):
                    
                    self.trans_clone.load_state_dict(old_state_dict)
                    trans_errors_ = torch.zeros(rewards.shape)
                    dkl_loss_ = 0
                    
                    for _ in range(self.args.sample_elbo):
                        pred_next_images_, pred_next_speeds_ = self.trans_clone.after_encode(
                            torch.clone(encoding_[episode, step]).unsqueeze(0), sequential_actions[episode, step].detach().unsqueeze(0), False)
                        
                        flat_pred_images_ = pred_next_images_*image_masks.detach()[:,self.args.lookahead-1:]
                        flat_pred_images_ = flat_pred_images_.flatten(2)
                        flat_pred_speeds_ = pred_next_speeds_*masks.detach()[:,self.args.lookahead-1:]
                        flat_pred_speeds_ = flat_pred_speeds_
                        flat_pred_ = torch.cat([flat_pred_images_, flat_pred_speeds_], dim = -1)
                        
                        errors_ = F.mse_loss(flat_pred_, flat_real.detach(), reduction = "none") 
                        errors_ = torch.sum(errors_, -1).unsqueeze(-1)
                        trans_errors_ += errors_ / self.args.sample_elbo
                        dkl_loss_ += self.args.dkl_rate * b_kl_loss(self.trans_clone) / self.args.sample_elbo
                    mse_loss_ = trans_errors_.sum()
                    trans_loss_ = mse_loss_ + dkl_loss_
                    
                    self.opt_clone.zero_grad()
                    trans_loss_.sum().backward()
                    self.opt_clone.step()
                
                    weights_after = self.trans_clone.weights()

                    dkl_change = dkl(weights_after[0], weights_after[1], weights_before[0], weights_before[1]) + \
                        dkl(weights_after[2], weights_after[3], weights_before[2], weights_before[3])
                    dkl_changes[episode,step] = dkl_change    
        
        dkl_change = log(dkl_changes.sum().item())    
        
        print("\n\n{}\n\n".format(dkl_changes))
        
        
        
        if(self.args.naive_curiosity == "true"):
            if(self.args.eta == None):
                curiosity = self.eta * trans_errors
                self.eta = self.eta * self.args.eta_rate
            else:
                curiosity = self.args.eta * trans_errors
                self.args.eta = self.args.eta * self.args.eta_rate
                
            print("\nMSE curiosity: {}, {}.\n".format(curiosity.shape, torch.sum(curiosity)))
        
        else:
            if(self.args.eta == None):
                curiosity = self.eta * dkl_changes
                self.eta = self.eta * self.args.eta_rate
            else:
                curiosity = self.args.eta * dkl_changes
                self.args.eta = self.args.eta * self.args.eta_rate
                
            print("\nFEB curiosity: {}, {}.\n".format(curiosity.shape, torch.sum(curiosity)))
            
            
    
        # Get encodings for other modules
        with torch.no_grad():
            encoded, _ = self.transitioner.just_encode(images.detach(), speeds.detach(), prev_actions.detach())
            next_encoded = encoded[:,1:]
            encoded = encoded[:,:-1]
            
        plot_predictions = True if num in (0, -1) and plot_predictions else False
        if(plot_predictions): plot_some_predictions(self.args, images, speeds, pred_next_images, pred_next_speeds, actions, masks, self.steps, epoch)
            
        extrinsic = torch.mean(rewards*masks.detach()).item()
        intrinsic_curiosity = torch.mean(curiosity*masks.detach()[:,self.args.lookahead-1:]).item()
        curiosity = torch.cat([curiosity, torch.zeros([curiosity.shape[0], self.args.lookahead-1, 1]).to(device)], dim = 1)
        rewards += curiosity
        
        
                
        # Train critics
        next_action, log_pis_next = self.actor.evaluate(next_encoded.detach())
        Q_target1_next = self.critic1_target(next_encoded.detach(), next_action.detach())
        Q_target2_next = self.critic2_target(next_encoded.detach(), next_action.detach())
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        if self.args.alpha == None: Q_targets = rewards.cpu() + (self.args.GAMMA * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.cpu()))
        else:                       Q_targets = rewards.cpu() + (self.args.GAMMA * (1 - dones.cpu()) * (Q_target_next.cpu() - self.args.alpha * log_pis_next.cpu()))
        
        Q_1 = self.critic1(encoded.detach(), actions.detach()).cpu()
        critic1_loss = 0.5*F.mse_loss(Q_1*masks.detach().cpu(), Q_targets.detach()*masks.detach().cpu())
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        Q_2 = self.critic2(encoded.detach(), actions.detach()).cpu()
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
        else:
            alpha_loss = None
            
        # Train eta
        if(self.args.eta == None):
            """
            eta_loss = "If testing a loss-funciton for curiosity's eta, put it here"
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
            actor_loss = None
        
        if(mse_loss != None): mse_loss = log(mse_loss.item())
        if(dkl_loss != None): 
            try: dkl_loss = log(dkl_loss.item())
            except: dkl_loss = 0
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(critic1_loss != None): critic1_loss = log(critic1_loss.item())
        if(critic2_loss != None): critic2_loss = log(critic2_loss.item())
        losses = np.array([[mse_loss, dkl_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss]])
        
        try:    intrinsic_entropy = (1 if intrinsic_entropy >= 0 else -1) * abs(intrinsic_entropy)**.5
        except: pass
        try:    intrinsic_curiosity = log(intrinsic_curiosity)
        except: pass
        
        return(losses, extrinsic, intrinsic_curiosity, intrinsic_entropy, dkl_change, weight_change)
                     
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
        
print("agent.py loaded.")