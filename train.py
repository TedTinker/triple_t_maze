#%%

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
import enlighten
from copy import deepcopy
from time import sleep
from itertools import accumulate

from utils import device, args, save_agent, load_agent, get_rolling_average, reset_start_time, folder
from env import Env
from agent import Agent

def episode(env, agent, push = True, delay = False):
    env.reset()  
    done = False
    positions = []
    with torch.no_grad():
        while(done == False):
            done, exit, which, pos = env.step(agent)
            positions.append(pos)
            if(delay): sleep(.5)
            if(device == "cuda"): torch.cuda.synchronize(device=device)
    env.body.to_push.finalize_rewards()
    rewards = deepcopy(env.body.to_push.rew)
    if(push): env.body.to_push.push(agent.memory, agent)
    else:     env.body.to_push.empty()
    env.close()
    return(exit, which, rewards, positions)



q = False



class Trainer():
    def __init__(
            self, args = args,
            load_folder = None, 
            load_name = None):
        
        self.current_arena = 0
        self.arena_names = ["1", "2", "3"]
        self.args = args
        self.load_folder = load_folder; self.load_name = load_name
                
        self.env     = Env(self.arena_names[self.current_arena], self.args, GUI = False)
        self.env_gui = None
        self.restart()
        torch.save(self, folder + "/trainer.pt")
        
    def new_load_name(self, folder, load_name):
        self.load_folder = folder
        self.load_name = load_name 
        self.restart()
    
    def restart(self):
        reset_start_time()
        self.e = 0
        self.agent = Agent(args = self.args)
        if(self.load_folder != None):
            self.agent = load_agent(
                self.agent, suf = self.load_name, folder = self.load_folder)
            #if(self.load_folder != self.save_folder):
            #    save_agent(self.agent, suf = self.e )
        else:
            save_agent(self.agent, suf = self.e )
        self.exits = []; self.exits_rolled = []; self.which = []
        self.ext= []; self.int_cur = []; self.int_ent = []
        self.rewards = []; self.punishments = []
        self.losses = np.array([[None]*6])
        
    def get_GUI(self):
        if(self.env_gui == None):
            self.env_gui = Env(self.arena_names[self.current_arena], self.args, GUI = True)
        return(self.env_gui)
    
    def close_env(self, forever = False):
        self.env.close(forever)
        if(self.env_gui != None):
            self.env_gui.close(forever)
            self.env_gui = None
        
    def one_episode(self, push = True, GUI = False, delay = False):     
        if(GUI == False): GUI = q
        if(GUI): env = self.get_GUI()
        else:    env = self.env
        win, which, rewards, positions = \
            episode(env, self.agent, push, delay)
        return(int(win), which, rewards, positions)

    def epoch(self, plot_predictions = False, append = True):
        for _ in range(self.args.episodes_per_epoch):
            exit, which, rewards, _ = self.one_episode()
            if(append):
                self.exits.append(exit)
                self.exits_rolled.append(get_rolling_average(self.exits))
                self.which.append(which)
                rewards = sum(rewards)
                if(rewards > 0): self.rewards.append(rewards); self.punishments.append(0)
                else:            self.punishments.append(rewards); self.rewards.append(0)
        
        losses, extrinsic, intrinsic_curiosity, intrinsic_entropy = \
            self.agent.learn(batch_size = self.args.batch_size, iterations = self.args.iterations, plot_predictions = plot_predictions, epoch = self.e)
        if(append):
            self.ext.append(extrinsic)
            self.int_cur.append(intrinsic_curiosity)
            self.int_ent.append(intrinsic_entropy)

            self.losses = np.concatenate([self.losses, losses])

    def train(self):
        self.agent.train()
        manager = enlighten.Manager()
        E = manager.counter(total = sum(self.args.epochs_per_arena), desc = "Epochs:", unit = "ticks", color = "blue")
        if(self.args.fill_memory):
            for e in range(self.args.batch_size):
                self.epoch(plot_predictions = False, append = False)
        epoch_sum = list(accumulate(self.args.epochs_per_arena))

        while(True):
            E.update()
            self.e += 1
            self.epoch(plot_predictions = self.e % self.args.show_and_save_pred == 0)
            if(self.e % self.args.show_and_save == 0): 
                save_agent(self.agent, suf = self.e)
                
            if(self.e >= epoch_sum[self.current_arena] and self.current_arena+1 < len(self.arena_names)):
                self.current_arena += 1
                self.close_env(True)
                if(self.args.discard_memory): self.agent.restart_memory()
                self.env = Env(self.arena_names[self.current_arena], self.args, GUI = False)
                if(self.args.fill_memory):
                    for e_ in range(self.args.batch_size):
                        self.epoch(plot_predictions = False, append = False)

            if(self.e >= epoch_sum[-1]):
                save_agent(self.agent, suf = self.e)
                plot_dict = {
                    "args"        : self.args,
                    "folder"      : folder,
                    "exits_rolled" : self.exits_rolled,
                    "which"       : self.which,
                    "rew"         : self.rewards,
                    "pun"         : self.punishments, 
                    "ext"         : self.ext, 
                    "cur"         : self.int_cur,
                    "ent"         : self.int_ent,
                    "mse"         : self.losses[:,0],
                    "dkl"         : self.losses[:,1],
                    "alpha"       : self.losses[:,2],
                    "actor"       : self.losses[:,3],
                    "crit1"       : self.losses[:,4],
                    "crit2"       : self.losses[:,5]}
                torch.save(plot_dict, folder + "/plot_dict.pt")
                self.close_env(True)
                break
    
    def test(self, size = 100):
        self.agent.eval()
        exits = 0
        for i in range(size):
            exit, which, rewards, positions = self.one_episode(push = False, GUI = True, delay = False)
            exits += exit
        print("Agent exits {} out of {} games ({}%).".format(exits, size, round(100*(exits/size))))
        
    def get_positions(self, size, arena_name = None):
        self.agent.eval()
        positions_list = []
        self.close_env(True)
        
        if(arena_name == None):
            arena_name = self.arena_names[self.current_arena]
        
        self.env = Env(arena_name, self.args, GUI = False)
            
        for i in range(size):
            w, which, rewards, positions = self.one_episode(push = False)
            positions_list.append(positions)
        self.close_env(True)
        return(positions_list, arena_name)
    
print("train.py loaded.")
# %%