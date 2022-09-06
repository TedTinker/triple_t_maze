import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--explore_type", type=str)
args = parser.parse_args()


import os
import shutil
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np 
from itertools import chain

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
    plt.title("{}: {} epochs, arena {}".format("_".join(folder.split("_")[:-1]), load_name, arena_name)) 
    
    files = os.listdir("saves")
    if(folder in files): pass
    else: os.mkdir("saves/"+folder)
    plt.savefig("saves/"+folder+"/arena_{}_tracks_{}".format(arena_name, load_name)+".png", bbox_inches='tight')
    
    plt.close()


    


def plot_all_positions(training_name):
    pos_dicts = []
    f = os.listdir("saves")
    for folder in f:
        breaks = folder.split("_")
        if(breaks[-1] in ["positions", "done"] or folder[-4:] == ".png"): pass 
        elif("_".join(breaks[:-1]) == training_name):
            _, pos_dict = torch.load("saves/" + folder + "/pos_dict.pt")
            pos_dicts.append(pos_dict)
        
    pos_dict = pos_dicts[0]
    for k in pos_dict.keys():
        for pd in pos_dicts[1:]:
            pos_dict[k] += pd[k]
    
    for (load_name, arena_name), positions_lists in pos_dict.items():
        if(positions_lists == []): pass 
        else:
            plot_positions(positions_lists = positions_lists, 
                           arena_name = arena_name, 
                           folder = training_name + "_positions",
                           load_name = load_name)
            

    
def make_vid(training_name, fps = 1):
    files = []
    folder = "saves/{}_positions".format(training_name)
    for file in os.listdir(folder):
        if(file[-4:] == ".png"):
            files.append(file)
    files.sort()
    
    frame = cv2.imread(folder + "/" + files[0]); height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
    video = cv2.VideoWriter("saves/{}_video.avi".format(training_name), fourcc, fps, (width, height))
    for file in files:
        video.write(cv2.imread(folder + "/" + file))
    cv2.destroyAllWindows()
    video.release()
    
def make_mega_vid(order, fps = 1):
    types = {}
    for row in order:
        for t in row:
            types[t] = []
        
    for k in types.keys():
        if(k != "empty_space"):
            folder = "saves/{}_positions".format(k)
            for file in os.listdir(folder):
                if(file[-4:] == ".png"):
                    types[k].append(file)
            types[k].sort()
    
    folders = []
    for folder in os.listdir("saves"):
        folders.append(folder)
    print(folders)
    if("all_positions" in folders): shutil.rmtree("saves/all_positions")
    os.mkdir("saves/all_positions")
    
    length = len(types[list(types.keys())[0]])
    rows = len(order) 
    columns = max([len(order[i]) for i in range(rows)])
    positions = []
    for i, row in enumerate(order):
        for j, column in enumerate(row):
            positions.append((i,j))
            
    xs = [] ; ys = []
    for i in range(length):
        images = []
        for kind in list(types.keys()):
            if(kind != "empty_space"):
                images.append(Image.open("saves/{}_positions/{}".format(kind, types[kind][i])))
        for image in images:
            xs.append(image.size[0]) ; ys.append(image.size[1])
    x = max(xs) ; y = max(ys)
                    
    for i in range(length):
        images = []
        for kind in list(chain(*order)):
            if(kind != "empty_space"):
                images.append(Image.open("saves/{}_positions/{}".format(kind, types[kind][i])))
            else:
                images.append(None)
            #print(images[-1].shape)
        new_image = Image.new("RGB", (columns*x, rows*y))
        for j, image in enumerate(images):
            row, column = positions[j]
            if(image != None): new_image.paste(image, (column*x,row*y))
            #print("Together:", new_image.shape)
            #print()
        new_image.save("saves/all_positions/{}.png".format(str(i).zfill(5)), format="PNG")
        
    make_vid("all", fps)
    
    
    
def make_end_pics(training_name):
    folders = []
    for folder in os.listdir("saves"):
        if("_".join(folder.split("_")[:-1]) == training_name and folder.split('_')[-1] != "positions"):
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

if(args.explore_type[0] != "("):
    make_end_pics(args.explore_type)
    plot_all_positions(args.explore_type)
    new_text("\n\nDone with {}!".format(args.explore_type))
else:
    order = args.explore_type[1:-1]
    order = order.split("+")
    if(order[-1] != "break"): order.append("break")
    row = [] ; rows = [] 
    for i, job in enumerate(order):
        if(job != "break"):
            row.append(job)
        else:
            rows.append(row) ; row = []
    order = rows
    make_mega_vid(order)
    new_text("\n\nDone!")



