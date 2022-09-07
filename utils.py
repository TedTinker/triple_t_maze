#%%

import argparse
from math import pi
import numpy as np

parser = argparse.ArgumentParser()

# Meta 
parser.add_argument("--explore_type",       type=str,   default = "POST_MAIN") 
parser.add_argument("--id",                 type=int,   default = 0)

# Environment 
parser.add_argument('--boxes_per_cube',     type=int,   default = 2)  
parser.add_argument('--bigger_cube',        type=float, default = 1.4)    
parser.add_argument('--wall_punishment',    type=float, default = .1)
parser.add_argument('--reward_scaling',     type=float, default = .999)    
parser.add_argument("--gamma",              type=float, default = .99)  # For discounting reward

# Agent
parser.add_argument('--body_size',          type=float, default = 2)    
parser.add_argument('--image_size',         type=int,   default = 8)
parser.add_argument('--max_steps',          type=int,   default = 30)
parser.add_argument('--min_speed',          type=float, default = 25)
parser.add_argument('--max_speed',          type=float, default = 100)
parser.add_argument('--max_yaw_change',     type=float, default = pi/2)

# Module 
parser.add_argument('--batch_size',         type=int,   default = 128)
parser.add_argument('--hidden_size',        type=int,   default = 128)
parser.add_argument('--encode_size',        type=int,   default = 128)
parser.add_argument('--lstm_size',          type=int,   default = 256)
parser.add_argument('--trans_lr',           type=float, default = .001)
parser.add_argument('--actor_lr',           type=float, default = .001) 
parser.add_argument('--critic_lr',          type=float, default = .001) 
parser.add_argument('--alpha_lr',           type=float, default = .005) 

# Memory buffer
parser.add_argument('--capacity',           type=int,   default = 500)
parser.add_argument('--power',              type=float, default = 2)
parser.add_argument('--discard_memory',     type=bool,  default = False)
parser.add_argument('--fill_memory',        type=bool,  default = False)

# Training
parser.add_argument('--epochs_per_arena',   type=int,   default = 1500)
parser.add_argument('--episodes_per_epoch', type=int,   default = 1)
parser.add_argument('--iterations',         type=int,   default = 1)
parser.add_argument("--d",                  type=int,   default = 2)    # Delay to train actors
parser.add_argument("--alpha",              type=float, default = None) # Soft-Actor-Critic entropy aim
parser.add_argument("--target_entropy",     type=float, default = -2)   # Soft-Actor-Critic entropy aim
parser.add_argument("--eta",                type=float, default = 5)    # Scale curiosity
parser.add_argument("--eta_rate",           type=float, default = 1)    # Scale eta
parser.add_argument("--tau",                type=float, default = 1e-2) # For soft-updating target critics

# Plotting and saving
parser.add_argument('--too_long',           type=int,   default = None)
parser.add_argument('--show_and_save',      type=int,   default = 50)

args = parser.parse_args()

#%%

better_expectation = ((.5, .5),(.5, 3.5))
arena_dict = {
    "1.png" : ((2,2),                         # Start (Y, X)
               {(1,1) : 1,                    # This reward here
               (1,3) : better_expectation}),  # This reward here
    "2.png" : ((3,3),
               {(1,1) : 1,
                (1,5) : 1,
                (3,1) : better_expectation,
                (3,5) : 1}),
    "3.png" : ((4, 4),
                {(1,1) : 1,
                (1,3) : 1,
                (1,5) : better_expectation,
                (1,7) : 1,
                (5,1) : 1,
                (5,3) : 1,
                (5,5) : 1,
                (5,7) : 1})}



import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal

already_done = False 
os.chdir("triple_t_maze")
folder = "saves/{}_{}".format(args.explore_type, str(args.id).zfill(3))
if args.id != 0:
    try:
        os.mkdir(folder)
        os.mkdir(folder + "/agents")
        os.mkdir(folder + "/plots")
    except:
        already_done = True

def new_text(string):
    print(string + "\n")
    
new_text("\nID: {}_{}.\nDevice: {}.".format(args.explore_type, str(args.id).zfill(3), device))



# Monitor GPU memory.
def get_free_mem(string = ""):
    #r = torch.cuda.memory_reserved(0)
    #a = torch.cuda.memory_allocated(0)
    #f = r-a  # free inside reserved
    #print("\n{}: {}.\n".format(string, f))
    pass 

# Remove from GPU memory.
def delete_these(verbose = False, *args):
    if(verbose): get_free_mem("Before deleting")
    del args
    #torch.cuda.empty_cache()
    if(verbose): get_free_mem("After deleting")
    
def shape_out(layer, shape_in):
    example = torch.zeros(shape_in)
    example = layer(example)
    return(example.shape)

def multiply_these(l):
    product = 1
    for l_ in l: product*=l_ 
    return(product)

def flatten_shape(shape, num):
    new_shape = tuple(s for i,s in enumerate(shape) if i < num)
    new_shape += (multiply_these(shape[num:]),)
    return(new_shape)

def reshape_shape(shape, new_shape):
    assert(multiply_these(shape) == multiply_these(new_shape))
    return(new_shape)

def cat_shape(shape_1, shape_2, dim):
    assert(len(shape_1) == len(shape_2))
    new_shape = ()
    for (s1, s2, d) in zip(shape_1, shape_2, range(len(shape_1))):
        if(d != dim): 
            assert(s1 == s2)
            new_shape += (s1,)
        else:
            new_shape += (s1+s2,)
    return(new_shape)

class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
    
def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass

            
            
# How to get rolling average.
def get_rolling_average(wins, roll = 100):
    if(len(wins) < roll):
        return(sum(wins)/len(wins))
    return(sum(wins[-roll:])/roll)       


# How to add discount to a list.
def add_discount(rewards, GAMMA = .99):
    d = rewards[-1]
    for i, r in enumerate(rewards[:-1]):
        rewards[i] += d*(GAMMA)**(len(rewards) - i)
    return(rewards)



# Track seconds starting right now. 
import datetime
start_time = datetime.datetime.now()
def reset_start_time():
    global start_time
    start_time = datetime.datetime.now()
def duration():
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)
  
  

# How to save plots.
import matplotlib.pyplot as plt
import shutil

def remove_folder(folder):
    files = os.listdir("saves")
    if(folder not in files): return
    shutil.rmtree("saves/" + folder)
    
def save_plot(name):
    plt.savefig(folder + "/plots/"+name+".png") #, bbox_inches='tight')
  
def delete_with_name(name, subfolder = "plots"):
    files = os.listdir(folder + "/{}".format(subfolder))
    for file in files:
        if(file.startswith(name)):
            os.remove(folder + "/{}/{}".format(subfolder, file))
            

def divide_arenas(epochs, here = plt):
    x = [e for e in epochs if e%args.epochs_per_arena == 0 and e != 0]
    for x_ in x:
        here.axvline(x=x_, color = (0,0,0,.2))
    

# How to plot an episode's rewards.
def plot_rewards(rewards):
    total_length = len(rewards)
    x = [i for i in range(1, total_length + 1)]
    plt.plot(x, [0 for _ in range(total_length)], "--", color = "black", alpha = .5)
    plt.plot(x, rewards, color = "turquoise")
    plt.title("Rewards")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    save_plot("rewards")
    plt.close()
    
# How to plot cumulative rewards.
def plot_cumulative_rewards(rewards, punishments, name = ""):
    total_length = len(rewards)
    x = [i for i in range(1, total_length + 1)]
    rewards = np.cumsum(rewards)
    punishments = np.cumsum(punishments)
    divide_arenas(x)
    plt.title("Cumulative Rewards and Punishments")
    plt.xlabel("Time")
    plt.ylabel("Rewards/Punishments")

    plt.plot(x, [0 for _ in range(total_length)], "--", color = "black", alpha = .5)
    plt.plot(x, rewards, color = "turquoise")
    plt.plot(x, punishments, color = "pink")

    save_plot("cumulative" + ("_{}".format(name) if name != "" else ""))
    plt.close()
    
    
def get_x_y(losses, too_long = None):
    x = [i for i in range(len(losses)) if losses[i] != None]
    y = [l for l in losses if l != None]
    if(too_long != None and len(x) > too_long):
        x = x[-too_long:]; y = y[-too_long:]
    return(x, y)

def normalize(this):
    if(all(i == 0 for i in this) or min(this) == max(this)): pass
    else:
        this = [2*((i - min(this)) / (max(this) - min(this)))-1 for i in this]
    return(this)


# How to plot extrinsic vs intrinsic.
def plot_extrinsic_intrinsic(extrinsic, intrinsic_curiosity, intrinsic_entropy, name = ""):
    
    ex, ey       = get_x_y(extrinsic)
    icx, icy     = get_x_y(intrinsic_curiosity)
    iex, iey     = get_x_y(intrinsic_entropy)
    
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    
    divide_arenas([x+1 for x in ex])

    plt.axhline(y = 0, color = 'gray', linestyle = '--')
    if(not all(i == 0 for i in icy)):
        plt.plot(icx, icy, color = "green", label = "Curiosity")
    if(not all(i == 0 for i in iey)):
        plt.plot(iex, iey, color = "blue",  label = "Entropy")
    plt.plot(ex,  ey,  color = "red",   label = "Extrinsic", alpha = .5)
    plt.legend(loc = 'upper left')
    
    plt.title("Average Extrinsic vs Intrinsic Rewards")
    save_plot("ext_int" + ("_{}".format(name) if name != "" else ""))
    plt.close()
    
    ey = normalize(ey)
    icy = normalize(icy)
    iey = normalize(iey)
    
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    
    divide_arenas([x+1 for x in ex])

    plt.axhline(y = 0, color = 'gray', linestyle = '--')
    if(not all(i == 0 for i in icy)):
        plt.plot(icx, icy, color = "green", label = "Curiosity")
    if(not all(i == 0 for i in iey)):
        plt.plot(iex, iey, color = "blue",  label = "Entropy")
    plt.plot(ex,  ey,  color = "red",   label = "Extrinsic", alpha = .5)
    plt.legend()
    
    plt.title("Normalized average Extrinsic vs Intrinsic Rewards")
    save_plot("ext_int_normalized" + ("_{}".format(name) if name != "" else ""))
    plt.close()
    
# Compare rewards to curiosity.
def plot_curiosity(rewards, curiosity, masks):
    fig, ax1 = plt.subplots()    
    ax2 = ax1.twinx()
    for i in range(len(rewards)):
        r = rewards[i].squeeze(-1).tolist()
        r = [r_ for j, r_ in enumerate(r) if masks[i,j] == 1]
        c = curiosity[i].squeeze(-1).tolist()
        c = [c_ for j, c_ in enumerate(c) if masks[i,j] == 1]
        x = [i for i in range(len(r))]
        ax1.plot(x, r, color = "blue", alpha = .5, label = "Reward" if i == 0 else "")
        ax2.plot(x, c, color = "green", alpha = .5, label = "Curiosity" if i == 0 else "")
    plt.title("Value of rewards vs curiosity")
    ax1.set_ylabel("Rewards")
    ax1.legend(loc = 'upper left')
    ax2.set_ylabel("Curiosity")
    ax2.legend(loc = 'lower left')
    plt.close()
    
            

# How to plot losses.
def plot_losses(losses, too_long, d, name = ""):
    trans_losses   = losses[:,0]
    alpha_losses   = losses[:,1]
    actor_losses   = losses[:,2]
    critic1_losses = losses[:,3]
    critic2_losses = losses[:,4]
    
    if(len(alpha_losses) == len([a for a in alpha_losses if a == None])):
        no_alpha = True 
    else:
        no_alpha = False
    
    trans_x, trans_y     = get_x_y(trans_losses, too_long)
    alpha_x, alpha_y     = get_x_y(alpha_losses, too_long)
    actor_x, actor_y     = get_x_y(actor_losses, too_long)
    critic1_x, critic1_y = get_x_y(critic1_losses, None if too_long == None else too_long * d)
    critic2_x, critic2_y = get_x_y(critic2_losses, None if too_long == None else too_long * d)
    
    trans_x = [x/args.iterations for x in trans_x]
    alpha_x = [x/args.iterations for x in alpha_x]
    actor_x = [x/args.iterations for x in actor_x]
    critic1_x = [x/args.iterations for x in critic1_x]
    critic2_x = [x/args.iterations for x in critic2_x]
    
    divide_arenas(trans_x)
    
    # Plot trans_loss
    plt.xlabel("Epochs")
    plt.plot(trans_x, trans_y, color = "green", label = "Trans")
    plt.ylabel("Trans losses")
    plt.legend(loc = 'upper left')
    plt.title("Transitioner loss")
    save_plot("loss_trans" + ("_{}".format(name) if name != "" else ""))
    plt.close()
    
    # Plot losses for actor, critics, and alpha
    fig, ax1 = plt.subplots()
    plt.xlabel("Epochs")

    ax1.plot(actor_x, actor_y, color='red', label = "Actor")
    ax1.set_ylabel("Actor losses")
    ax1.legend(loc = 'upper left')

    ax2 = ax1.twinx()
    divide_arenas(trans_x)
    ax2.plot(critic1_x, critic1_y, color='blue', linestyle = "--", label = "Critic")
    ax2.plot(critic2_x, critic2_y, color='blue', linestyle = ":", label = "Critic")
    ax2.set_ylabel("Critic losses")
    ax2.legend(loc = 'lower left')
    
    if(not no_alpha):
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.2))
        ax3.plot(alpha_x, alpha_y, color = (0,0,0,.5), label = "Alpha")
        ax3.set_ylabel("Alpha losses")
        ax3.legend(loc = 'upper right')
    
    plt.title("Agent losses")
    fig.tight_layout()
    save_plot("loss_agent" + ("_{}".format(name) if name != "" else ""))
    plt.close()
    
    
  
# How to plot victory-rates.
def plot_wins(wins, name = ""):
    x = [i for i in range(1, len(wins)+1)]
    divide_arenas(x)
    plt.plot(x, wins, color = "gray")
    plt.ylim([0, 1])
    plt.title("Win-rates")
    plt.xlabel("Episodes")
    plt.ylabel("Win-rate")
    save_plot("wins" + ("_{}".format(name) if name != "" else ""))
    plt.close()
    
# How to plot kinds of victory.
def plot_which(which, name = ""):
    which = [(w, r) if type(r) in [int, float] else (w, sum([w_*r_ for (w_, r_) in r])) for (w, r) in which]
    which = [w[0] + ", " + str(w[1]) for w in which]
    kinds = list(set(which))
    kinds.sort()
    kinds.insert(0, kinds.pop(-1))
    kinds.reverse()
    # Screw that, do it manually 
    kinds = ["FAIL, -1", 
             "1A, 1", "1B, 2.0", 
             "2A, 1", "2B, 1", "2C, 2.0", "2D, 1",
             "3A, 1", "3B, 1", "3C, 2.0", "3D, 1", "3E, 1", "3F, 1", "3G, 1", "3H, 1"]
    kinds.reverse()
    plt.scatter([0 for _ in kinds], kinds, color = (0,0,0,0))
    plt.axhline(y = 13.5, color = (0, 0, 0, .2))
    plt.axhline(y = 11.5, color = (0, 0, 0, .2))
    plt.axhline(y = 7.5,  color = (0, 0, 0, .2))
    
    x = [i for i in range(1, len(which)+1)]
    divide_arenas(x)
    plt.scatter(x, which, color = "gray")
    plt.title("Kind of Win")
    plt.xlabel("Episodes")
    plt.ylabel("Which Victory")
    save_plot("which" + ("_{}".format(name) if name != "" else ""))
    plt.close()





      
  
  
# How to save/load agent

def save_agent(agent, suf = ""):
    if(folder == None): return
    if(type(suf) == int): suf = str(suf).zfill(5)
    torch.save(agent.state_dict(), folder + "/agents/agent_{}.pt".format(suf))

def load_agent(agent, folder, suf = "last"):
    if(type(suf) == int): suf = str(suf).zfill(5)
    agent.load_state_dict(torch.load(folder + "/agents/agent_{}.pt".format(suf)))
    return(agent)

if args.id != 0:
    new_text("units.py loaded.")
# %%
