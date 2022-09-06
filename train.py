#%%

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
import enlighten
from copy import deepcopy
from time import sleep

from utils import device, args, save_agent, load_agent, get_rolling_average,reset_start_time, folder, \
    delete_with_name, plot_rewards, plot_losses, plot_wins, plot_extrinsic_intrinsic, plot_which, plot_cumulative_rewards, new_text
from env import Env
from agent import Agent

def episode(env, agent, push = True, delay = False):
    env.reset()  
    done = False
    positions = []
    with torch.no_grad():
        while(done == False):
            done, win, which, pos = env.step(agent)
            positions.append(pos)
            if(delay): sleep(.5)
            torch.cuda.synchronize(device=device)
    env.body.to_push.finalize_rewards()
    rewards = deepcopy(env.body.to_push.rew)
    if(push): env.body.to_push.push(agent.memory)
    else:     env.body.to_push.empty()
    env.close()
    return(win, which, rewards, positions)



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
        self.wins = []; self.wins_rolled = []; self.which = []
        self.extrinsics = []; self.intrinsic_curiosities = []; self.intrinsic_entropies = []
        self.rewards = []; self.punishments = []
        self.losses = np.array([[None]*5])
        
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
        if(q): plot_rewards(rewards)
        return(int(win), which, rewards, positions)

    def epoch(self, plot = False, append = True):
        for _ in range(self.args.episodes_per_epoch):
            win, which, rewards, _ = self.one_episode()
            if(append):
                self.wins.append(win)
                self.wins_rolled.append(get_rolling_average(self.wins))
                self.which.append(which)
                rewards = sum(rewards)
                if(rewards > 0): self.rewards.append(rewards); self.punishments.append(0)
                else:            self.punishments.append(rewards); self.rewards.append(0)
                    
        losses, extrinsic, intrinsic_curiosity, intrinsic_entropy = \
            self.agent.learn(batch_size = self.args.batch_size, iterations = self.args.iterations, plot = plot)
        if(append):
            self.extrinsics.append(extrinsic)
            self.intrinsic_curiosities.append(intrinsic_curiosity)
            self.intrinsic_entropies.append(intrinsic_entropy)

            self.losses = np.concatenate([self.losses, losses])

        if(q): plot_losses(self.losses, too_long = self.args.too_long, d = self.args.d)

    def train(self):
        self.agent.train()
        manager = enlighten.Manager()
        E = manager.counter(total = 3*args.epochs_per_arena, desc = "Epochs:", unit = "ticks", color = "blue")
        if(self.args.fill_memory):
            for e in range(self.args.batch_size):
                self.epoch(plot = False, append = False)
        prob = range(len(self.arena_names) * self.args.epochs_per_arena)
        for e in prob:
            E.update()
            self.e += 1
            self.epoch(plot = self.e % self.args.show_and_save == 0)
            if(self.e % self.args.show_and_save == 0): 
                save_agent(self.agent, suf = self.e)
                plot_wins(self.wins_rolled, name = str(self.e).zfill(5))
                plot_which(self.which, name = str(self.e).zfill(5))                
                plot_cumulative_rewards(self.rewards, self.punishments, name = str(self.e).zfill(5))
                plot_extrinsic_intrinsic(self.extrinsics, self.intrinsic_curiosities, self.intrinsic_entropies, name = str(self.e).zfill(5))
                plot_losses(self.losses, too_long = self.args.too_long, d = self.args.d, name = str(self.e).zfill(5))
            if(self.e >= len(self.arena_names) * self.args.epochs_per_arena):
                save_agent(self.agent, suf = self.e)
                for string in ["wins", "which", "cumulative", "ext_int", "loss"]:
                    delete_with_name(string, subfolder = "plots")
                plot_wins(self.wins_rolled, name = "")
                plot_which(self.which, name = "")
                plot_cumulative_rewards(self.rewards, self.punishments, name = "")
                plot_extrinsic_intrinsic(self.extrinsics, self.intrinsic_curiosities, self.intrinsic_entropies, name = "")
                plot_losses(self.losses, too_long = None, d = self.args.d, name = "")
                self.close_env(True)
                break
            if(self.e >= (self.current_arena+1) * self.args.epochs_per_arena):
                self.current_arena += 1
                self.close_env(True)
                if(self.args.discard_memory): self.agent.restart_memory()
                self.env = Env(self.arena_names[self.current_arena], self.args, GUI = False)
                if(self.args.fill_memory):
                    for e_ in range(self.args.batch_size):
                        self.epoch(plot = False, append = False)
    
    def test(self, size = 100):
        self.agent.eval()
        wins = 0
        for i in range(size):
            w, which, rewards, positions = self.one_episode(push = False, GUI = True, delay = False)
            wins += w
        print("Agent wins {} out of {} games ({}%).".format(wins, size, round(100*(wins/size))))
        
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
    
new_text("train.py loaded.")
# %%
