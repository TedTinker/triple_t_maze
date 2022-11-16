import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--explore_type", type=str)
args = parser.parse_args()


import os
import shutil
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import cv2
import matplotlib.pyplot as plt
import numpy as np 
from itertools import chain

import torch

# When I import this, it tries using provided parameters for utils' args.
from utils import arena_dict, plots

import datetime 
start_time = datetime.datetime.now()

def reset_start_time():
    global start_time
    start_time = datetime.datetime.now()
    
def duration():
    global start_time
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)



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
    
    exits = arena_dict[arena_name + ".png"].exits
    rewards = exits["Reward"].values.tolist() 
    rewards = list(set(rewards))
    rewards_ints = [r if type(r) in [int, float] else sum([w*r_ for (w, r_) in r]) for r in rewards]
    rewards_ints.sort()
    reward_color = {rewards_ints[0] : (1, 0, 0), rewards_ints[-1] : (0, 1, 0)}
    
    for n, k, v in zip(exits["Name"].values.tolist(), exits["Position"].values.tolist(), exits["Reward"].values.tolist()):
        r = v if type(v) in [int, float] else sum([w*v_ for (w, v_) in v])
        ax.text(k[1], -k[0], n, size = 10, bbox={'facecolor': reward_color[r], 'alpha': 0.5, 'pad': 5})
            
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
                           folder = training_name + "_shared",
                           load_name = load_name)
            

    
def make_vid(training_name, fps = 1):
    files = []
    folder = "saves/{}_shared".format(training_name)
    for file in os.listdir(folder):
        if(file[-4:] == ".png" and file[:5] != "plots"):
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
            folder = "saves/{}_shared".format(k)
            for file in os.listdir(folder):
                if(file[-4:] == ".png" and file[:5] != "plots"):
                    types[k].append(file)
            types[k].sort()
    
    folders = []
    for folder in os.listdir("saves"):
        folders.append(folder)
    if("all_positions" in folders): shutil.rmtree("saves/all_positions")
    os.mkdir("saves/all_shared")
    
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
                images.append(Image.open("saves/{}_shared/{}".format(kind, types[kind][i])))
        for image in images:
            xs.append(image.size[0]) ; ys.append(image.size[1])
    x = max(xs) ; y = max(ys)
                    
    for i in range(length):
        images = []
        for kind in list(chain(*order)):
            if(kind != "empty_space"):
                images.append(Image.open("saves/{}_shared/{}".format(kind, types[kind][i])))
            else:
                images.append(None)
            #print(images[-1].shape)
        new_image = Image.new("RGB", (columns*x, rows*y), color="white")
        for j, image in enumerate(images):
            row, column = positions[j]
            if(image != None): new_image.paste(image, (column*x,row*y))
            #print("Together:", new_image.shape)
            #print()
        new_image.save("saves/all_shared/{}.png".format(str(i).zfill(5)), format="PNG")
        
    make_vid("all", fps)
    
    
    
def tuple_min_max(min_max_list):
    mins = [min_max[0] for min_max in min_max_list]
    maxs = [min_max[1] for min_max in min_max_list]
    return((min(mins), max(maxs)))
    
def get_min_max(this, plot_dict_dict, cumulative = False):
    plot_dict_list = [] 
    for key in plot_dict_dict.keys(): plot_dict_list += plot_dict_dict[key]
    these = [[i for i in plot_dict[this] if i != None] for plot_dict in plot_dict_list]
    these = [t for t in these if t != []]
    these = np.array(these)
    if(cumulative):
        these = np.cumsum(these, -1)
    return(np.amin(these), np.amax(these))



def make_end_pics(order):
    real_order = [] 
    for o in order: real_order += [o_ for o_ in o if o_ != "empty_space"]
    order = real_order

    all_folders = []
    for folder in os.listdir("saves"):
        if("_".join(folder.split("_")[:-1]) in order and folder.split('_')[-1] != "shared"):
            all_folders.append(folder)
    all_folders.sort()
                
    plot_dict_dict = {training_name : [] for training_name in order}
    for folder in all_folders:
        training_name = "_".join(folder.split("_")[:-1])
        plot_dict_dict[training_name].append(torch.load("saves/" + folder + "/plot_dict.pt"))
                        
    rew_min_max = get_min_max("rew", plot_dict_dict, True)
    pun_min_max = get_min_max("pun", plot_dict_dict, True)
    ext_min_max = get_min_max("ext", plot_dict_dict)
    cur_min_max = get_min_max("cur", plot_dict_dict)
    ent_min_max = get_min_max("ent", plot_dict_dict)
    trans_min_max = get_min_max("trans", plot_dict_dict)
    alpha_min_max = get_min_max("alpha", plot_dict_dict)
    actor_min_max = get_min_max("actor", plot_dict_dict)
    critic1_min_max = get_min_max("crit1", plot_dict_dict)
    critic2_min_max = get_min_max("crit2", plot_dict_dict)
        
    critic_min_max = tuple_min_max([critic1_min_max, critic2_min_max])
    rew_min_max = tuple_min_max([rew_min_max, pun_min_max]) 
    ext_min_max = tuple_min_max([ext_min_max, cur_min_max, ent_min_max]) 
    
    mins_maxs = [rew_min_max, ext_min_max, trans_min_max, actor_min_max, critic_min_max, alpha_min_max]
    
    print("\n\nStarting plots.\n{}\n\n".format(duration()))
        
    for training_name, plot_dict_list in plot_dict_dict.items():
        for i, plot_dict in enumerate(plot_dict_list):
            plots(plot_dict, mins_maxs, folder = plot_dict["folder"] + "/plots")  
            print("\n\nPlot done.\n{}\n\n".format(duration()))
        plots(plot_dict_list, mins_maxs, folder = "saves/" + training_name + "_shared")
        print("\n\nMany plots done.\n{}\n\n".format(duration()))

    for training_name in order:
        folders = []
        for folder in os.listdir("saves"):
            if("_".join(folder.split("_")[:-1]) == training_name):
                folders.append("saves/" + folder)
        folders.sort()
            
        images = []    
        for folder in folders:
            if(folder.split("_")[-1] == "shared"):
                images.append(Image.open(folder + "/plots.png"))
            else:
                images.append(Image.open(folder + "/plots/plots.png"))
        new_image = Image.new("RGB", (len(folders)*images[0].size[0], images[0].size[1]))
        for i, image in enumerate(images):
            new_image.paste(image, (i*image.size[0], 0))
        new_image.save("saves/all_{}_plots.png".format(training_name))
            
    # Predictions
    for training_name in order:
        os.mkdir("saves/{}_predictions".format(training_name))
        
        folders = []
        for folder in os.listdir("saves"):
            if("_".join(folder.split("_")[:-1]) == training_name and not folder.split('_')[-1] in ("shared", "predictions")):
                folders.append(folder)
        
        for folder in folders:
            images = []
            files = os.listdir("saves/{}/predictions".format(folder)) ; files.sort()
            for f in files:
                images.append(Image.open("saves/{}/predictions/{}".format(folder, f)))
            new_image = Image.new("RGB", (images[0].size[0], len(files)*images[0].size[1]))
            for i, image in enumerate(images):
                new_image.paste(image, (0, i*image.size[1]))
            new_image.save("saves/{}_predictions/predictions_{}.png".format(training_name, folder.split("_")[-1]))
            shutil.rmtree("saves/{}/predictions".format(folder))
            
        images = []
        files = os.listdir("saves/{}_predictions".format(training_name)) ; files.sort()
        for f in files:
            images.append(Image.open("saves/{}_predictions/{}".format(training_name, f)))
            
        new_image = Image.new("RGB", (len(images)*(10+images[0].size[0]), images[0].size[1]))
        for i, image in enumerate(images):
            new_image.paste(image, (i*(10+image.size[0]), 0))
        new_image.save("saves/{}_predictions.png".format(training_name))
        shutil.rmtree("saves/{}_predictions".format(training_name))
        
def make_together_pic(order):
    real_order = [] 
    for o in order: real_order += [o_ for o_ in o if o_ != "empty_space"]
    order = real_order
    
    images = [] ; names = []
    for f in os.listdir("saves"):
        if f[-4:] == ".png" and f[:4] == "all_":
            name = f[4:-10]
            if name in order:
                images.append(Image.open("saves/{}".format(f)))
                names.append(name)
            
    indices = [names.index(name) for name in order]
    names = [names[index] for index in indices]
    images = [images[index] for index in indices]
    
    width = 0
    for f in os.listdir("saves/{}_001/plots".format(order[0])):
        image = Image.open("saves/{}_001/plots/{}".format(order[0], f))
        width = max(width, image.size[0])
    w, h = images[0].size
    images = [image.crop((w-width, 0, w, h)) for image in images]
    
    font = ImageFont.truetype('arial.ttf', 50)
    new_image = Image.new("RGB", ((len(order))*width, images[0].size[1]+150), color="white")
    for i, image in enumerate(images):
        new_image.paste(image, (i*image.size[0], 30))
        I1 = ImageDraw.Draw(new_image)
        I1.text((i*image.size[0]+50,5), names[i], font = font, fill = (0,0,0))
    new_image.save("saves/together_plots.png")
    
    
#%%

if(args.explore_type[0] != "("):
    plot_all_positions(args.explore_type)
    print("\n\nDone with {}!".format(args.explore_type))
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
    make_end_pics(order)
    make_together_pic(order)
    make_mega_vid(order)



