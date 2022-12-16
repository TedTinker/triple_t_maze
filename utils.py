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
parser.add_argument('--map',                type=tuple, default = ("1", "2", "3"))
parser.add_argument('--boxes_per_cube',     type=int,   default = 1)  
parser.add_argument('--bigger_cube',        type=float, default = 1.05)    
parser.add_argument('--wall_punishment',    type=float, default = .1)
parser.add_argument('--reward_scaling',     type=float, default = .999)    
parser.add_argument("--GAMMA",              type=float, default = .9)  # For discounting reward in training
parser.add_argument("--gamma",              type=float, default = .9)  # For discounting reward
parser.add_argument("--default_reward",     type=float, default = -1)#1)
parser.add_argument("--better_reward",      type=float, default = 1)#((.5, .5),(.5, 3.5)))

# Agent
parser.add_argument('--body_size',          type=float, default = 2)    
parser.add_argument('--image_size',         type=int,   default = 8)
parser.add_argument('--min_speed',          type=float, default = 50)
parser.add_argument('--max_speed',          type=float, default = 100)
parser.add_argument('--steps_per_step',     type=int,   default = 5)
parser.add_argument('--max_steps',          type=int,   default = 16)
parser.add_argument('--max_yaw_change',     type=float, default = pi/2)

# Module 
parser.add_argument('--lookahead',          type=int,   default = 1)
parser.add_argument('--batch_size',         type=int,   default = 16)
parser.add_argument('--hidden_size',        type=int,   default = 128)
parser.add_argument('--encode_size',        type=int,   default = 128)
parser.add_argument('--lstm_size',          type=int,   default = 256)
parser.add_argument('--trans_lr',           type=float, default = .005)
parser.add_argument('--actor_lr',           type=float, default = .005) 
parser.add_argument('--critic_lr',          type=float, default = .005) 
parser.add_argument('--alpha_lr',           type=float, default = .01) 
parser.add_argument('--eta_lr',             type=float, default = .01)     # Not implemented

# Memory buffer
parser.add_argument('--capacity',           type=int,   default = 300)
parser.add_argument('--replacement',        type=str,   default = "index")
parser.add_argument('--selection',          type=str,   default = "uniform")
parser.add_argument('--power',              type=float, default = 1)
parser.add_argument('--discard_memory',     type=bool,  default = False)
parser.add_argument('--fill_memory',        type=bool,  default = False)

# Training
parser.add_argument('--epochs_per_arena',   type=tuple, default = (500, 1500, 3000))
parser.add_argument('--episodes_per_epoch', type=int,   default = 1)
parser.add_argument('--iterations',         type=int,   default = 1)
parser.add_argument("--d",                  type=int,   default = 2)    # Delay to train actors
parser.add_argument("--alpha",              type=float, default = None) # Soft-Actor-Critic entropy aim
parser.add_argument("--target_entropy",     type=float, default = -2)   # Soft-Actor-Critic entropy aim
parser.add_argument("--eta",                type=float, default = None) # Scale curiosity
parser.add_argument("--eta_rate",           type=float, default = 1)    # Scale eta
parser.add_argument("--tau",                type=float, default = .05)  # For soft-updating target critics
parser.add_argument("--dkl_rate",           type=float, default = .1)   # Scale bayesian dkl
parser.add_argument("--sample_elbo",        type=int,   default = 30)    # Samples for elbo
parser.add_argument("--naive_curiosity",    type=str,   default = "true") # Which kind of curiosity
parser.add_argument("--dkl_change_size",    type=str,   default = "batch")  # "batch", "episode", "step"

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
    "t.png" : Arena_Dict(
        (3, 2),
        [Exit(  "L",    (2,0), 1),
        Exit(   "R",    (2,7), ((.5, .5), (.5, 3.5)))]),
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



def dkl(mu_1, sigma_1, mu_2, sigma_2):
    sigma_1 = torch.pow(sigma_1, 2)
    sigma_2 = torch.pow(sigma_2, 2)
    term_1 = torch.pow(mu_2 - mu_1, 2) / sigma_2 
    term_2 = sigma_1 / sigma_2 
    term_3 = torch.log(term_2)
    return((.5 * (term_1 + term_2 - term_3 - 1)).sum())

def average_change(before, after):
    change = []
    for b, a in zip(before, after):
        change.append(torch.mean(b-a).item())
    return(change)



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
    
# Track seconds starting right now. 
start_time = datetime.datetime.now()

def reset_start_time():
    global start_time
    start_time = datetime.datetime.now()
    
def duration():
    global start_time
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)
    
    
    
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




# How to save/load agent

def save_agent(agent, suf = ""):
    if(folder == None): return
    if(type(suf) == int): suf = str(suf).zfill(5)
    torch.save(agent.state_dict(), folder + "/agents/agent_{}.pt".format(suf))

def load_agent(agent, folder, suf = "last"):
    if(type(suf) == int): suf = str(suf).zfill(5)
    agent.load_state_dict(torch.load(folder + "/agents/agent_{}.pt".format(suf)))
    return(agent)
  
  

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
    
def plot_some_predictions(args, images, speeds, pred_next_images, pred_next_speeds, actions, masks, steps, epoch):
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
            title += "Epoch {}. Step {} action: {} degrees, {} speed".format(epoch, step_num+i, yaws[i], round(spes[i]/args.steps_per_step))
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

    
    
def get_x_y(losses, too_long = None):
    x = [i for i in range(len(losses)) if losses[i] != None]
    y = [l for l in losses if l != None]
    if(too_long != None and len(x) > too_long):
        x = x[-too_long:]; y = y[-too_long:]
    return(x, y)

def normalize(this):
    if(all(i == 0 for i in this) or min(this) == max(this)): pass
    else:
        minimum = min(this)
        maximum = max(this)
        this = np.array(this)
        this = 2*((this - minimum) / (maximum - minimum)) - 1
    return(this)

def get_quantiles(plot_dict_list, name):
    xs = [i for i, l in enumerate(plot_dict_list[0][name]) if l != None]
    lists = np.array([[i for i in plot_dict[name] if i != None] for plot_dict in plot_dict_list])    
    q05 = np.quantile(lists, .05, 0)
    med = np.quantile(lists, .50, 0)
    q95 = np.quantile(lists, .95, 0)
    return(xs, q05, med, q95)
    
    
line_transparency = .5 ; fill_transparency = .1
def plots(plot_dict, mins_maxs, folder = folder, name = ""):    
    if(type(plot_dict) == list): many = True  ; epochs = len(plot_dict[0]["rew"])
    else:                        many = False ; epochs = len(plot_dict["rew"])
    
    fig, axs = plt.subplots(9, 1, figsize = (7, 45))
    xs = [i for i in range(epochs)]
    
    
    
    # Cumulative rewards
    ax = axs[0]
    if(many): 
        rew_xs, low_rew, rew, high_rew = get_quantiles(plot_dict, "rew")
        pun_xs, low_pun, pun, high_pun = get_quantiles(plot_dict, "pun")
        low_rew = np.cumsum(low_rew) ; rew = np.cumsum(rew) ; high_rew = np.cumsum(high_rew)
        low_pun = np.cumsum(low_pun) ; pun = np.cumsum(pun) ; high_pun = np.cumsum(high_pun)
        ax.fill_between(rew_xs, low_rew, high_rew, color = "turquoise", alpha = 2*fill_transparency, linewidth = 0)
        ax.fill_between(pun_xs, low_pun, high_pun, color = "pink", alpha = 2*fill_transparency, linewidth = 0)
    else:
        rew = plot_dict["rew"] ; rew = np.cumsum(rew)
        pun = plot_dict["pun"] ; pun = np.cumsum(pun)
    ax.axhline(y = 0, color = 'gray', linestyle = '--')
    ax.plot([x for x in range(len(rew))], rew, color = "turquoise", alpha = 2*line_transparency, label = "Reward")
    ax.plot([x for x in range(len(pun))], pun, color = "pink", alpha = 2*line_transparency,      label = "Punishment")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Rewards/Punishments")
    ax.title.set_text("Cumulative Rewards and Punishments")
    divide_arenas(xs, ax)
    ax.legend(loc = 'lower left')
    ax.set_ylim(mins_maxs[0])
    
    
    
    # Extrinsic, intrinsic
    ax = axs[1]
    if(many):
        ext_xs, low_ext, ext, high_ext = get_quantiles(plot_dict, "ext")
        cur_xs, low_cur, cur, high_cur = get_quantiles(plot_dict, "cur")
        ent_xs, low_ent, ent, high_ent = get_quantiles(plot_dict, "ent")
        _, low_ext_y = get_x_y(low_ext) ; _, high_ext_y = get_x_y(high_ext)
        _, low_cur_y = get_x_y(low_cur) ; _, high_cur_y = get_x_y(high_cur)
        _, low_ent_y = get_x_y(low_ent) ; _, high_ent_y = get_x_y(high_ent)
        ax.fill_between(ext_xs, low_ext_y, high_ext_y, color = "red", alpha = fill_transparency, linewidth = 0)
        ax.fill_between(cur_xs, low_cur_y, high_cur_y, color = "green", alpha = fill_transparency, linewidth = 0)
        ax.fill_between(ent_xs, low_ent_y, high_ent_y, color = "blue", alpha = fill_transparency, linewidth = 0)
    else:
        ext = plot_dict["ext"] ; cur = plot_dict["cur"] ; ent = plot_dict["ent"]
        ext_xs = None ; cur_xs = None ; ent_xs = None
        
    ex, ey       = get_x_y(ext)
    icx, icy     = get_x_y(cur)
    iex, iey     = get_x_y(ent)

    ax.axhline(y = 0, color = 'gray', linestyle = '--')
    ax.plot(ext_xs if ext_xs != None else ex,  ey,  color = "red", alpha = line_transparency, label = "Extrinsic")
    if(not all(i == 0 for i in icy)):
        ax.plot(cur_xs if cur_xs != None else icx, icy, color = "green", alpha = line_transparency, label = "ln Curiosity")
    if(not all(i == 0 for i in iey)):
        ax.plot(ent_xs if ent_xs != None else iex, iey, color = "blue", alpha = line_transparency, label = "sq Entropy")
    ax.legend(loc = 'upper left')
    ax.set_ylim(mins_maxs[1])
    
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Value")
    ax.title.set_text("Extrinsic vs Intrinsic Rewards")
    divide_arenas(xs, ax)



    # Extrinsic, intrinsic normalized
    ax = axs[2]
    
    if(many):
        ney = normalize(low_ext_y + ey + high_ext_y)
        nicy = normalize(low_cur_y + icy + high_cur_y)
        niey = normalize(low_ent_y + iey + high_ent_y)  
        low_ext_y = ney[:len(ey)]    ; high_ext_y = ney[2*len(ey):]   ; ney = ney[len(ey):2*len(ey)]
        low_cur_y = nicy[:len(icy)]  ; high_cur_y = nicy[2*len(icy):] ; nicy = nicy[len(ey):2*len(icy)]
        low_ent_y = niey[:len(iey)]  ; high_ent_y = niey[2*len(iey):] ; niey = niey[len(iey):2*len(iey)]
        ax.fill_between(ex, low_ext_y, high_ext_y, color = "red", alpha = fill_transparency, linewidth = 0)
        ax.fill_between(icx, low_cur_y, high_cur_y, color = "green", alpha = fill_transparency, linewidth = 0)
        #ax.fill_between(iex, low_ent_y, high_ent_y, color = "blue", alpha = fill_transparency, linewidth = 0)
    else:
        ney = normalize(ey)
        nicy = normalize(icy)
        niey = normalize(iey)    
    
    ax.axhline(y = 0, color = 'gray', linestyle = '--')
    ax.plot(ex,  ney,  color = "red", alpha = line_transparency, label = "Extrinsic")
    if(not all(i == 0 for i in icy)):
        ax.plot(icx, nicy, color = "green", alpha = line_transparency, label = "ln Curiosity")
    #if(not all(i == 0 for i in iey)):
    #    ax.plot(iex, niey, color = "blue", alpha = line_transparency, label = "sq Entropy")
    ax.legend(loc = 'upper left')
    
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Value")
    ax.title.set_text("Normalized Extrinsic vs Intrinsic Rewards")    
    divide_arenas(xs, ax)
        
    # Agent losses
    if(many):
        mse_xs, low_mse, mse, high_mse = get_quantiles(plot_dict, "mse")
        dkl_xs, low_dkl, dkl, high_dkl = get_quantiles(plot_dict, "dkl")
        alpha_xs, low_alpha, alpha, high_alpha = get_quantiles(plot_dict, "alpha")
        actor_xs, low_actor, actor, high_actor = get_quantiles(plot_dict, "actor")
        crit1_xs, low_crit1, crit1, high_crit1 = get_quantiles(plot_dict, "crit1")
        crit2_xs, low_crit2, crit2, high_crit2 = get_quantiles(plot_dict, "crit2")
        
        _, mse_y = get_x_y(mse) ; _, low_mse_y = get_x_y(low_mse)  ; _, high_mse_y = get_x_y(high_mse)
        _, dkl_y = get_x_y(dkl) ; _, low_dkl_y = get_x_y(low_dkl)  ; _, high_dkl_y = get_x_y(high_dkl)
        _, alpha_y = get_x_y(alpha) ; _, low_alpha_y = get_x_y(low_alpha)  ; _, high_alpha_y = get_x_y(high_alpha)
        _, actor_y = get_x_y(actor) ; _, low_actor_y = get_x_y(low_actor)  ; _, high_actor_y = get_x_y(high_actor)
        _, crit1_y = get_x_y(crit1) ; _, low_crit1_y = get_x_y(low_crit1)  ; _, high_crit1_y = get_x_y(high_crit1)
        _, crit2_y = get_x_y(crit2) ; _, low_crit2_y = get_x_y(low_crit2)  ; _, high_crit2_y = get_x_y(high_crit2)
        
    else:
        mse = plot_dict["mse"]
        dkl = plot_dict["dkl"]
        alpha = plot_dict["alpha"]
        actor = plot_dict["actor"]
        crit1 = plot_dict["crit1"]
        crit2 = plot_dict["crit2"]
        
        mse_xs, mse_y = get_x_y(mse)
        dkl_xs, dkl_y = get_x_y(dkl)
        alpha_xs, alpha_y = get_x_y(alpha)
        actor_xs, actor_y = get_x_y(actor)
        crit1_xs, crit1_y = get_x_y(crit1)
        crit2_xs, crit2_y = get_x_y(crit2)
    
    if(len(alpha) == len([a for a in alpha if a == None])):
        no_alpha = True 
    else:
        no_alpha = False
    
    # Trans losses
    ax = axs[3]
    if(many): 
        ax.fill_between(mse_xs, low_mse_y, high_mse_y, color = "green", alpha = fill_transparency, linewidth = 0)
        ax.fill_between(dkl_xs, low_dkl_y, high_dkl_y, color = "red", alpha = fill_transparency, linewidth = 0)
    ax.plot(mse_xs, mse_y, color = "green", alpha = line_transparency, label = "ln mse")
    ax.plot(dkl_xs, dkl_y, color = "red", alpha = line_transparency, label = "ln dkl")
    ax.legend(loc = 'upper left')
    divide_arenas(dkl_xs, ax)
    ax.set_ylim(mins_maxs[2])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("ln Trans losses")
    ax.title.set_text("Transitioner losses")
    
    # Plot losses for actor, critics, and alpha
    ax1 = axs[4]
    if(many): ax1.fill_between(actor_xs, low_actor_y, high_actor_y, color = "red", alpha = fill_transparency, linewidth = 0)
    ax1.plot(actor_xs, actor_y, color='red', alpha = line_transparency, label = "Actor")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Actor losses")
    ax1.legend(loc = 'upper left')
    ax1.set_ylim(mins_maxs[3])

    ax2 = ax1.twinx()
    if(many): 
        ax2.fill_between(crit1_xs, low_crit1_y, high_crit1_y, color = "blue", alpha = fill_transparency, linewidth = 0)
        ax2.fill_between(crit2_xs, low_crit2_y, high_crit2_y, color = "blue", alpha = fill_transparency, linewidth = 0)
    ax2.plot(crit1_xs, crit1_y, color='blue', alpha = line_transparency, linestyle = "--", label = "Critic")
    ax2.plot(crit2_xs, crit2_y, color='blue', alpha = line_transparency, linestyle = ":",  label = "Critic")
    ax2.set_ylabel("ln Critic losses")
    ax2.legend(loc = 'lower left')
    ax2.set_ylim(mins_maxs[4])
    
    if(not no_alpha):
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.2))
        if(many): ax3.fill_between(alpha_xs, low_alpha_y, high_alpha_y, color = (0,0,0,fill_transparency), linewidth = 0)
        ax3.plot(alpha_xs, alpha_y, color = (0,0,0,line_transparency), label = "Alpha")
        ax3.set_ylabel("Alpha losses")
        ax3.legend(loc = 'upper right')
        ax3.set_ylim(mins_maxs[5])
        
    divide_arenas(xs, ax1)
    
    ax1.title.set_text("Agent losses")
    
    
    
    # Plot which exits taken
    ax = axs[5]
    kinds = ["FAIL", 
             "L", "R",
             "LL", "LR", "RL", "RR",
             "LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR"]
    kinds = [r"$\bf{(" + k + ")}$" if k in ["R", "LL", "RLL"] else k for k in kinds]
    kinds.reverse()
    ax.scatter([0 for _ in kinds], kinds, color = (0,0,0,0))
    ax.axhline(y = 13.5, color = (0, 0, 0, .2))
    ax.axhline(y = 11.5, color = (0, 0, 0, .2))
    ax.axhline(y = 7.5,  color = (0, 0, 0, .2))
    divide_arenas(xs, ax)
    
    if(many):
        for dict in plot_dict:
            which = dict["which"]
            which = [(w, r) if type(r) != tuple else (w, sum([w_*r_ for (w_, r_) in r])) for (w, r) in which]
            which = [r"$\bf{(" + w[0] + ")}$" if w[0] in ["R", "LL", "RLL"] else w[0] for w in which]
            ax.scatter([x for x in range(len(which))], which, color = "gray", alpha = 1/len(plot_dict))
    else:
        which = plot_dict["which"]
        which = [(w, r) if type(r) != tuple else (w, sum([w_*r_ for (w_, r_) in r])) for (w, r) in which]
        which = [r"$\bf{(" + w[0] + ")}$" if w[0] in ["R", "LL", "RLL"] else w[0] for w in which]
        ax.scatter([x for x in range(len(which))], which, color = "gray")
        
    ax.title.set_text("Kind of Exit")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Which Exit")
    
    
    
    # Weight means
    ax1 = axs[6]
    ax2 = ax1.twinx()
    if(many): 
        weight_mean_xs, low_weight_mean, weight_mean, high_weight_mean = get_quantiles(plot_dict, "weight_mean")
        bias_mean_xs, low_bias_mean, bias_mean, high_bias_mean = get_quantiles(plot_dict, "bias_mean")
        ax1.fill_between(weight_mean_xs, low_weight_mean, high_weight_mean, color = "red", alpha = 2*fill_transparency, linewidth = 0)
        ax2.fill_between(bias_mean_xs, low_bias_mean, high_bias_mean, color = "blue", alpha = 2*fill_transparency, linewidth = 0)
    else:
        weight_mean = plot_dict["weight_mean"]
        bias_mean = plot_dict["bias_mean"]
    ax1.plot(xs, weight_mean, color = "red", alpha = 2*line_transparency, label = "Weight Mean")
    ax2.plot(xs, bias_mean, color = "blue", alpha = 2*line_transparency, label = "Bias Mean")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Weight Mean")
    ax2.set_ylabel("Bias Mean")
    ax1.title.set_text("Change in Weight Means")
    divide_arenas(xs, ax)
    ax1.legend(loc = 'upper left')
    ax2.legend(loc = 'lower left')
    ax1.set_ylim(mins_maxs[6])
    ax2.set_ylim(mins_maxs[7])
    
    # Weight stds
    ax1 = axs[7]
    ax2 = ax1.twinx()
    if(many): 
        weight_std_xs, low_weight_std, weight_std, high_weight_std = get_quantiles(plot_dict, "weight_std")
        bias_std_xs, low_bias_std, bias_std, high_bias_std = get_quantiles(plot_dict, "bias_std")
        ax1.fill_between(weight_std_xs, low_weight_std, high_weight_std, color = "red", alpha = fill_transparency, linewidth = 0)
        ax2.fill_between(bias_std_xs, low_bias_std, high_bias_std, color = "blue", alpha = fill_transparency, linewidth = 0)
    else:
        weight_std = plot_dict["weight_std"]
        bias_std = plot_dict["bias_std"]
    ax1.plot(xs, weight_std, color = "red", alpha = line_transparency, label = "Weight STD")
    ax2.plot(xs, bias_std, color = "blue", alpha = line_transparency, label = "Bias STD")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Weight STD")
    ax2.set_ylabel("Bias STD")
    ax1.title.set_text("Change in Weight STDs")
    divide_arenas(xs, ax)
    ax1.legend(loc = 'upper left')
    ax2.legend(loc = 'lower left')
    ax1.set_ylim(mins_maxs[8])
    ax2.set_ylim(mins_maxs[9])
    
    # Changes in DKL
    ax = axs[8]
    if(many): 
        dkl_xs, low_dkl, dkl, high_dkl = get_quantiles(plot_dict, "dkl_change")
        ax.fill_between(dkl_xs, low_dkl, high_dkl, color = "green", alpha = fill_transparency, linewidth = 0)
    else:
        dkl = plot_dict["dkl_change"]
    ax.plot(xs, dkl, color = "green", alpha = line_transparency, label = "ln DKL change")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("ln DKL change")
    ax.title.set_text("Change in DKL")
    divide_arenas(xs, ax)
    #ax.legend(loc = 'lower left')
    ax.set_ylim(mins_maxs[10])
    


    # Save
    plt.savefig(folder + "/plots" + ("_{}".format(name) if name != "" else "") + ".png")
    plt.close()
    


if args.id != 0:
    print("units.py loaded.")