#%%

import argparse
from math import pi
import numpy as np
import pandas as pd
import datetime
from itertools import accumulate
from matplotlib import pyplot as plt, font_manager as fm
import shutil
from PIL import Image
from random import choice
from math import degrees
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal

import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

# Meta 
parser.add_argument("--explore_type",       type=str,   default = "POST_MAIN") 
parser.add_argument("--id",                 type=int,   default = 0)

# Environment 
parser.add_argument('--boxes_per_cube',     type=int,   default = 2)  
parser.add_argument('--bigger_cube',        type=float, default = 1.2)    
parser.add_argument('--wall_punishment',    type=float, default = .1)
parser.add_argument('--reward_scaling',     type=float, default = .999)    
parser.add_argument("--gamma",              type=float, default = .9)  # For discounting reward
parser.add_argument("--default_reward",     type=float, default = -1)#1)
parser.add_argument("--better_reward",      type=float, default = 1)#((.5, .5),(.5, 3.5)))

# Agent
parser.add_argument('--body_size',          type=float, default = 2)    
parser.add_argument('--image_size',         type=int,   default = 8)
parser.add_argument('--min_speed',          type=float, default = 40)
parser.add_argument('--max_speed',          type=float, default = 50)
parser.add_argument('--max_steps',          type=int,   default = 30)
parser.add_argument('--max_yaw_change',     type=float, default = pi/2)

# Module 
parser.add_argument('--lookahead',          type=int,   default = 1)
parser.add_argument('--batch_size',         type=int,   default = 128)
parser.add_argument('--hidden_size',        type=int,   default = 128)
parser.add_argument('--encode_size',        type=int,   default = 128)
parser.add_argument('--lstm_size',          type=int,   default = 256)
parser.add_argument('--trans_lr',           type=float, default = .001)
parser.add_argument('--actor_lr',           type=float, default = .001) 
parser.add_argument('--critic_lr',          type=float, default = .001) 
parser.add_argument('--alpha_lr',           type=float, default = .005) 
parser.add_argument('--eta_lr',             type=float, default = .005)     # Not yet implemented

# Memory buffer
parser.add_argument('--capacity',           type=int,   default = 300)
parser.add_argument('--replacement',        type=str,   default = "index")
parser.add_argument('--selection',          type=str,   default = "uniform")
parser.add_argument('--power',              type=float, default = 1)
parser.add_argument('--discard_memory',     type=bool,  default = False)
parser.add_argument('--fill_memory',        type=bool,  default = False)

# Training
parser.add_argument('--epochs_per_arena',   type=int,   default = (1000, 2000, 4000))
parser.add_argument('--episodes_per_epoch', type=int,   default = 1)
parser.add_argument('--iterations',         type=int,   default = 1)
parser.add_argument("--d",                  type=int,   default = 2)    # Delay to train actors
parser.add_argument("--alpha",              type=float, default = None) # Soft-Actor-Critic entropy aim
parser.add_argument("--target_entropy",     type=float, default = -2)   # Soft-Actor-Critic entropy aim
parser.add_argument("--eta",                type=float, default = None) # Scale curiosity
parser.add_argument("--eta_rate",           type=float, default = 1)    # Scale eta
parser.add_argument("--tau",                type=float, default = .05)  # For soft-updating target critics

# Plotting and saving
parser.add_argument('--too_long',           type=int,   default = None)
parser.add_argument('--show_and_save',      type=int,   default = 250)
parser.add_argument('--show_and_save_pred', type=int,   default = 250)
parser.add_argument('--predictions_to_plot',type=int,   default = 1)

try:    args = parser.parse_args()
except: args, _ = parser.parse_known_args()



class Exit:
    def __init__(self, name, pos, rew):     # Position (Y, X)
        self.name = name ; self.pos = pos ; self.rew = rew

class Arena_Dict:
    def __init__(self, start, exits):
        self.start = start 
        self.exits = pd.DataFrame(
            data = [[exit.name, exit.pos, exit.rew] for exit in exits],
            columns = ["Name", "Position", "Reward"])
        
arena_dict = {
    "1.png" : Arena_Dict(
        (2,2), 
        [Exit(  "L",    (1,0), args.default_reward),
        Exit(   "R",    (1,4), args.better_reward)]),
    "2.png" : Arena_Dict(
        (3,3), 
        [Exit(  "LL",   (4,1), args.better_reward),
        Exit(   "LR",   (0,1), args.default_reward),
        Exit(   "RL",   (0,5), args.default_reward),
        Exit(   "RR",   (4,5), args.default_reward)]),
    "3.png" : Arena_Dict(
        (4,4), 
        [Exit(  "LLL",  (6,3), args.default_reward),
        Exit(   "LLR",  (6,1), args.default_reward),
        Exit(   "LRL",  (0,1), args.default_reward),
        Exit(   "LRR",  (0,3), args.default_reward),
        Exit(   "RLL",  (0,5), args.better_reward),
        Exit(   "RLR",  (0,7), args.default_reward),
        Exit(   "RRL",  (6,7), args.default_reward),
        Exit(   "RRR",  (6,5), args.default_reward)])}



already_done = False 
try:    os.chdir("triple_t_maze")
except: pass
folder = "saves/{}_{}".format(args.explore_type, str(args.id).zfill(3))
if args.id != 0:
    try:
        os.mkdir(folder)
        os.mkdir(folder + "/agents")
        os.mkdir(folder + "/plots")
        os.mkdir(folder + "/predictions")
    except:
        already_done = True
        
print("\nID: {}_{}.\nDevice: {}.\n".format(args.explore_type, str(args.id).zfill(3), device))



# Monitor GPU memory.
def get_free_mem(string = ""):
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print("\n{}: {}.\n".format(string, f))
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
def get_rolling_average(exits, roll = 100):
    if(len(exits) < roll):
        return(sum(exits)/len(exits))
    return(sum(exits[-roll:])/roll)       


# How to add discount to a list.
def add_discount(rewards, GAMMA = .99):
    d = rewards[-1]
    for i, r in enumerate(rewards[:-1]):
        rewards[i] += d*(GAMMA)**(len(rewards) - i)
    return(rewards)



# Track seconds starting right now. 
start_time = datetime.datetime.now()
def reset_start_time():
    global start_time
    start_time = datetime.datetime.now()
def duration():
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)
  
  

# How to save plots.
def remove_folder(folder):
    files = os.listdir("saves")
    if(folder not in files): return
    shutil.rmtree("saves/" + folder)
    
def save_plot(name, folder = folder):
    plt.savefig(folder + "/plots/"+name+".png") #, bbox_inches='tight')
  
def delete_with_name(name, subfolder = "plots"):
    files = os.listdir(folder + "/{}".format(subfolder))
    for file in files:
        if(file.startswith(name)):
            os.remove(folder + "/{}/{}".format(subfolder, file))
            
def divide_arenas(epochs, here = plt):
    sums = list(accumulate(args.epochs_per_arena))
    x = [e for e in epochs if e in sums]
    for x_ in x:
        here.axvline(x=x_, color = (0,0,0,.2))



# Plot random predictions
def norm_speed_to_speed(args, s):
    s = args.min_speed + ((s + 1)/2) * \
        (args.max_speed - args.min_speed)
    s = round(s)
    return(s)
    
def plot_some_predictions(args, images, speeds, pred_next_images, pred_next_speeds, actions, masks, steps):
    pred_images = []
    for ex in range(args.predictions_to_plot):
        batch_num = choice([i for i in range(actions.shape[0])])
        step_num =  choice([i for i in range(actions.shape[1] - args.lookahead - 1) if masks[batch_num, i] == 1])
        image_plot = (images[batch_num,step_num,:,:,:-1].cpu().detach() + 1) / 2
        next_image_plot = (images[batch_num,step_num+args.lookahead,:,:,:-1].cpu().detach() + 1) / 2
        next_speed = norm_speed_to_speed(args, speeds[batch_num, step_num+args.lookahead].cpu().detach().item())
        
        pred_next_image_plot = (pred_next_images[batch_num,step_num,:,:,:-1].cpu().detach() + 1) / 2
        pred_next_speed = norm_speed_to_speed(args, pred_next_speeds[batch_num, step_num].cpu().detach().item())
        
        yaws = [] ; spes = []
        for i in range(args.lookahead):
            yaw = actions[batch_num,step_num+i,0].item() * args.max_yaw_change
            yaw = round(degrees(yaw))
            spe = norm_speed_to_speed(args, actions[batch_num,step_num+i,1].item() )
            yaws.append(yaw)
            spes.append(spe)
            
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.title.set_text("Before")
        ax1.imshow(image_plot)
        ax1.axis('off')
        ax2.title.set_text("Prediction: {} speed".format(pred_next_speed))
        ax2.imshow(pred_next_image_plot)
        ax2.axis('off')
        ax3.title.set_text("After: {} speed".format(next_speed))
        ax3.imshow(next_image_plot)
        ax3.axis('off')
        title = ""
        for i in range(args.lookahead):
            title += "Step {} action: {} degrees, {} speed".format(step_num+i, yaws[i], spes[i])
            if(i < args.lookahead): title += "\n"
        fig.suptitle(title)
        fig.tight_layout()
        fig.subplots_adjust(top=1.2)
        plt.savefig(folder + "/plots/{}.png".format(ex), bbox_inches='tight',pad_inches = .3)
        plt.close()
        pred_images.append(Image.open(folder + "/plots/{}.png".format(ex)))
    new_image = Image.new("RGB", (args.predictions_to_plot*pred_images[0].size[0], pred_images[0].size[1]))
    for i, image in enumerate(pred_images):
        new_image.paste(image, (i*image.size[0], 0))
    new_image.save("{}/predictions/{}.png".format(folder, str(steps).zfill(6)))



# How to plot an episode's rewards.
def plot_rewards(rewards):
    total_length = len(rewards)
    x = [i for i in range(1, total_length + 1)]
    plt.plot(x, [0 for _ in range(total_length)], "--", color = "black", alpha = .5)
    plt.plot(x, rewards, color = "turquoise")
    plt.title("Rewards")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.show()
    #save_plot("rewards")
    plt.close()
    
    
    
# How to plot cumulative rewards.
def plot_cumulative_rewards(rewards, punishments, folder = folder, name = "", min_max = (0,0)):
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
    plt.ylim(min_max)

    save_plot("cumulative" + ("_{}".format(name) if name != "" else ""), folder)
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
def plot_extrinsic_intrinsic(extrinsic, intrinsic_curiosity, intrinsic_entropy, folder = folder, name = "", min_max = (0,0)):
    
    ex, ey       = get_x_y(extrinsic)
    icx, icy     = get_x_y(intrinsic_curiosity)
    iex, iey     = get_x_y(intrinsic_entropy)
    
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    
    divide_arenas([x+1 for x in ex])

    plt.axhline(y = 0, color = 'gray', linestyle = '--')
    if(not all(i == 0 for i in icy)):
        plt.plot(icx, icy, color = "green", label = "ln Curiosity")
    if(not all(i == 0 for i in iey)):
        plt.plot(iex, iey, color = "blue",  label = "sq Entropy")
    plt.plot(ex,  ey,  color = "red",   label = "Extrinsic", alpha = .5)
    plt.legend(loc = 'upper left')
    plt.ylim(min_max)
    
    plt.title("Average Extrinsic vs Intrinsic Rewards")
    save_plot("ext_int" + ("_{}".format(name) if name != "" else ""), folder)
    plt.close()
    
    ey = normalize(ey)
    icy = normalize(icy)
    iey = normalize(iey)
    
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    
    divide_arenas([x+1 for x in ex])

    plt.axhline(y = 0, color = 'gray', linestyle = '--')
    if(not all(i == 0 for i in icy)):
        plt.plot(icx, icy, color = "green", label = "ln Curiosity")
    if(not all(i == 0 for i in iey)):
        plt.plot(iex, iey, color = "blue",  label = "sq Entropy")
    plt.plot(ex,  ey,  color = "red",   label = "Extrinsic", alpha = .5)
    plt.legend(loc = 'upper left')
    
    plt.title("Normalized average Extrinsic vs Intrinsic Rewards")
    save_plot("ext_int_normalized" + ("_{}".format(name) if name != "" else ""), folder)
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
def plot_losses(losses, too_long, d, folder = folder, name = "", trans_min_max = (0,0), alpha_min_max = (0,0), actor_min_max = (0,0), critic_min_max = (0,0)):
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
    plt.ylabel("ln Trans losses")
    plt.legend(loc = 'upper left')
    plt.title("ln Transitioner loss")
    plt.ylim(trans_min_max)
    save_plot("loss_trans" + ("_{}".format(name) if name != "" else ""), folder)
    plt.close()
    
    # Plot losses for actor, critics, and alpha
    fig, ax1 = plt.subplots()
    plt.xlabel("Epochs")

    ax1.plot(actor_x, actor_y, color='red', label = "Actor")
    ax1.set_ylabel("Actor losses")
    ax1.legend(loc = 'upper left')
    ax1.set_ylim(actor_min_max)

    ax2 = ax1.twinx()
    divide_arenas(trans_x)
    ax2.plot(critic1_x, critic1_y, color='blue', linestyle = "--", label = "Critic")
    ax2.plot(critic2_x, critic2_y, color='blue', linestyle = ":",  label = "Critic")
    ax2.set_ylabel("ln Critic losses")
    ax2.legend(loc = 'lower left')
    ax2.set_ylim(critic_min_max)
    
    if(not no_alpha):
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.2))
        ax3.plot(alpha_x, alpha_y, color = (0,0,0,.5), label = "Alpha")
        ax3.set_ylabel("Alpha losses")
        ax3.legend(loc = 'upper right')
        ax3.set_ylim(alpha_min_max)
    
    plt.title("Agent losses")
    #fig.tight_layout()
    save_plot("loss_agent" + ("_{}".format(name) if name != "" else ""), folder)
    plt.close()
    
    
  
# How to plot exit-rates.
def plot_exits(exits, folder = folder, name = "", min_max = (0,0)):
    x = [i for i in range(1, len(exits)+1)]
    divide_arenas(x)
    plt.plot(x, exits, color = "gray")
    plt.ylim([0, 1])
    plt.title("Exit-rates")
    plt.xlabel("Episodes")
    plt.ylabel("Exit-rate")
    save_plot("exits" + ("_{}".format(name) if name != "" else ""), folder)
    plt.close()
    
    
    
# How to plot kinds of victory.
def plot_which(which, folder = folder, name = ""):
    which = [(w, r) if type(r) in [int, float] else (w, sum([w_*r_ for (w_, r_) in r])) for (w, r) in which]
    #which = [w[0] + ", " + str(w[1]) for w in which]
    
    which = [r"$\bf{(" + w[0] + ")}$" if w[0] in ["R", "LL", "RLL"] else w[0] for w in which]
    kinds = ["FAIL", 
             "L", "R",
             "LL", "LR", "RL", "RR",
             "LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR"]
    kinds = [r"$\bf{(" + k + ")}$" if k in ["R", "LL", "RLL"] else k for k in kinds]
    
    kinds.reverse()
    plt.scatter([0 for _ in kinds], kinds, color = (0,0,0,0))
    plt.axhline(y = 13.5, color = (0, 0, 0, .2))
    plt.axhline(y = 11.5, color = (0, 0, 0, .2))
    plt.axhline(y = 7.5,  color = (0, 0, 0, .2))
    
    x = [i for i in range(1, len(which)+1)]
    divide_arenas(x)
    plt.scatter(x, which, color = "gray")
    plt.title("Kind of Exit")
    plt.xlabel("Episodes")
    plt.ylabel("Which Victory")
    save_plot("which" + ("_{}".format(name) if name != "" else ""), folder)
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
    print("units.py loaded.")
# %%
