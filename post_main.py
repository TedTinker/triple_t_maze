import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pos_count",    type=int,   default = 10)
parser.add_argument("--explore_type", type=str,   default = "ALL")
args = parser.parse_args()


import os
from PIL import Image
from itertools import product 
from math import floor
import matplotlib.pyplot as plt
import numpy as np 
import enlighten

import torch

from utils import arena_dict, new_text #When I import this, it tries using provided parameters for utils' args.




from colorsys import hsv_to_rgb
def plot_positions(positions_lists, arena_name, folder, load_name):
    arena_map = plt.imread("arenas/" + arena_name + ".png")
    arena_map = np.flip(arena_map, 0)    
    
    h, w, _ = arena_map.shape
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    extent = [-.5, w-.5, -h+.5, .5]
    ax.imshow(arena_map, extent=extent, zorder = 1, origin='lower') 
    
    _, this_arena_dict = arena_dict[arena_name + ".png"]
    rewards = this_arena_dict.values() 
    rewards = list(set(rewards))
    rewards_ints = [r if type(r) in [int, float] else sum([w*r_ for (w, r_) in r]) for r in rewards]
    rewards_ints.sort()

    reward_colors = []
    for i in range(len(rewards_ints)):
        green = i/(len(rewards_ints)-1)
        reward_colors.append((1-green, green, 0))
    
    for k, v in this_arena_dict.items():
        ax.text(k[1], -k[0], 'E', bbox={'facecolor': reward_colors[rewards.index(v)], 'alpha': 0.5, 'pad': 2})
            
    colors = []
    for i in range(len(positions_lists)):
        hue = i/len(positions_lists)
        r, g, b = hsv_to_rgb(hue, 1, 1)
        colors.append((r, g, b))
    for i, positions_list in enumerate(positions_lists):
        for j, positions in enumerate(positions_list):
            x = [p[1] for p in positions]
            y = [-p[0] for p in positions]
            ax.plot(x, y, zorder = 2, color = colors[i], alpha = .5)
            ax.scatter(x[-1], y[-1], s = 100, color = "black", alpha = .5, marker = "*", zorder = 3)
            ax.scatter(x[-1], y[-1], s = 75, color = colors[i], alpha = .5, marker = "*", zorder = 4)
    plt.title("{}: Tracks of agents {}, arena {}".format(folder, load_name, arena_name)) 
    
    files = os.listdir("saves")
    if(folder in files): pass
    else: os.mkdir("saves/"+folder)
    plt.savefig("saves/"+folder+"/arena_{}_tracks_{}".format(arena_name, load_name)+".png")
    
    plt.close()


    
def positions(trainer):
    positions_list, arena_name = trainer.get_positions(size = args.pos_count)
    return(positions_list, arena_name)

def get_positions(training_name):
    folders = []
    f = os.listdir("saves")
    for folder in f:
        if(folder[:-4] == training_name):
            folders.append(folder)
    folders.sort()
        
    load_names = os.listdir("saves/" + folders[0] + "/agents")
    load_names.sort()
    load_names = [l[6:-3] for l in load_names]
        
    pos_dict = {(l, a) : [] for l, a in product(load_names, ["1", "2", "3"])}
    
    manager = enlighten.Manager()
    P = manager.counter(total = len(load_names), desc = "{} positions:".format(training_name), unit = "ticks", color = "blue")

    for load_name in load_names:
        positions_lists = []
        for i, folder in enumerate(folders):            
            trainer = torch.load("saves/" + folder + "/trainer.pt")
            trainer.new_load_name("saves/" + folder, load_name)
            which_arena = int(load_name)/trainer.args.epochs_per_arena
            if(which_arena == floor(which_arena)): 
                if(which_arena == 0): pass 
                elif(which_arena == 3): which_arena = 2
                else:
                    trainer.current_arena = floor(which_arena) - 1
                    positions_list, arena_name = positions(trainer)
                    pos_dict[(load_name, arena_name)].append(positions_list)
            trainer.current_arena = floor(which_arena)
            positions_list, arena_name = positions(trainer)
            pos_dict[(load_name, arena_name)].append(positions_list)
        P.update()
    
    for (load_name, arena_name), positions_lists in pos_dict.items():
        if(positions_lists == []): pass 
        else:
            plot_positions(positions_lists = positions_lists, 
                           arena_name = arena_name, 
                           folder = training_name + "_positions",
                           load_name = load_name)
            
def make_gif(training_name, duration = 500):
    files = []
    folder = "saves/{}_positions".format(training_name)
    for file in os.listdir(folder):
        if(file[-4:] == ".png"):
            files.append(file)
    files.sort()
    frames = []
    for file in files:
        new_frame = Image.open(folder + "/" +file)
        frames.append(new_frame)
    frames[0].save(folder + "/animation.gif", format="GIF",
                append_images=frames[1:],
                save_all=True,
                duration=duration, loop=0)
    
def make_mega_gif(duration = 500):
    files = {"none"    : [], 
             "entropy" : [], 
             "curious" : []}
    for k in files.keys():
        folder = "saves/{}_positions".format(k)
        for file in os.listdir(folder):
            if(file[-4:] == ".png"):
                files[k].append(file)
        files[k].sort()
    
    bigger_images = []
    
    for i in range(len(files["none"])):
        images = []
        for kind in ["none", "entropy", "curious"]:
            images.append(Image.open("saves/{}_positions/{}".format(kind, files[kind][i])))
        new_image = Image.new("RGB", (3*images[0].size[0], images[0].size[1]))
        for j, image in enumerate(images):
            new_image.paste(image, (j*image.size[0],0))
        bigger_images.append(new_image)
    bigger_images[0].save("saves/all_animations.gif", format="GIF",
                append_images=bigger_images[1:],
                save_all=True,
                duration=duration, loop=0)
    
def make_end_pics(training_name):
    folders = []
    for folder in os.listdir("saves"):
        if(folder.split('_')[0] == training_name and folder.split('_')[1] != "positions"):
            folders.append(folder)

    new_folder = "saves/{}_done".format(training_name)
    if(new_folder in folders): pass
    else: os.mkdir(new_folder)
    
    plot_names = ["cumulative", "ext_int", "ext_int_normalized", "loss_agent", "loss_trans", "which", "wins"]
    for name in plot_names:
        images = []
        for folder in folders:
            images.append(Image.open("saves/{}/plots/{}".format(folder, name+".png")))
        new_image = Image.new("RGB", (len(folders)*images[0].size[0], images[0].size[1]))
        for i, image in enumerate(images):
            new_image.paste(image, (i*image.size[0],0))
        new_image.save(new_folder + "/{}.png".format(name))
        
    images = []    
    files = os.listdir(new_folder); files.sort()
    for file in files:
        images.append(Image.open(new_folder + "/" + file))
        os.remove(new_folder + "/" + file)
    new_image = Image.new("RGB", (images[0].size[0], len(plot_names)*images[0].size[1]))
    for i, image in enumerate(images):
        new_image.paste(image, (0, i*image.size[1]))
    new_image.save("saves/all_{}_plots.png".format(training_name))
    os.rmdir(new_folder)
    
    
#%%

if(args.explore_type != "ALL"):
    make_end_pics(args.explore_type)
    get_positions(args.explore_type)
    make_gif(args.explore_type)
    new_text("\n\nDone with {}.".format(args.explore_type))
else:
    make_mega_gif()
    new_text("\n\nDone!")


