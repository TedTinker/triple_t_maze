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
from utils import arena_dict, new_text, \
    plot_rewards, plot_losses, plot_exits, plot_extrinsic_intrinsic, plot_which, plot_cumulative_rewards



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
        new_image = Image.new("RGB", (columns*x, rows*y), color="white")
        for j, image in enumerate(images):
            row, column = positions[j]
            if(image != None): new_image.paste(image, (column*x,row*y))
            #print("Together:", new_image.shape)
            #print()
        new_image.save("saves/all_positions/{}.png".format(str(i).zfill(5)), format="PNG")
        
    make_vid("all", fps)
    
    
    
def tuple_min_max(min_max_list):
    mins = [min_max[0] for min_max in min_max_list]
    maxs = [min_max[1] for min_max in min_max_list]
    return((min(mins), max(maxs)))
    
def get_min_max(this, plot_dict_list, cumulative = False):
    these = [plot_dict[this] for plot_dict in plot_dict_list]
    if(cumulative):
        these = [[t for t in this if t != None] for this in these]
        these = [sum(this) for this in these]
        return((min(these), max(these)))
    if(type(these[0]) == list):
        these = [[t for t in this if t != None] for this in these]
        mins = [min(this) for this in these]
        maxs = [max(this) for this in these]
        return((min(mins), max(maxs)))
    min_max_lists = [] 
    for this in these:
        this_min_max_list = []
        for column in this.T:
            that = [t for t in column.tolist() if t != None]
            if(len(that) == 0): this_min_max_list.append((0,0))
            else:               this_min_max_list.append((min(that), max(that)))
        min_max_lists.append(this_min_max_list)
    min_max_list = []
    for i in range(len(min_max_lists[0])):
        min_max_list.append(tuple_min_max([min_max_lists[j][i] for j in range(len(min_max_lists))]))
    return(min_max_list)



def make_end_pics(order):
    real_order = [] 
    for o in order: real_order += [o_ for o_ in o if o_ != "empty_space"]
    order = real_order

    all_folders = []
    for folder in os.listdir("saves"):
        if("_".join(folder.split("_")[:-1]) in order and folder.split('_')[-1] != "positions"):
            all_folders.append(folder)
                
    plot_dict_list = []
    for folder in all_folders:
        plot_dict_list.append(torch.load("saves/" + folder + "/plot_dict.pt"))
        
    exits_rolled_min_max = get_min_max("exits_rolled", plot_dict_list)
    rew_min_max = get_min_max("rew", plot_dict_list, True)
    pun_min_max = get_min_max("pun", plot_dict_list, True)
    ext_min_max = get_min_max("ext", plot_dict_list)
    cur_min_max = get_min_max("cur", plot_dict_list)
    ent_min_max = get_min_max("ent", plot_dict_list)
    trans_min_max, alpha_min_max, actor_min_max, critic1_min_max, critic2_min_max = get_min_max("losses", plot_dict_list)
    
    critic_min_max = tuple_min_max([critic1_min_max, critic2_min_max])
    rew_min_max = tuple_min_max([rew_min_max, pun_min_max]) 
    ext_min_max = tuple_min_max([ext_min_max, cur_min_max, ent_min_max]) 
    
    for plot_dict in plot_dict_list:
        plot_exits(plot_dict["exits_rolled"], folder = plot_dict["folder"], name = "", min_max = exits_rolled_min_max)
        plot_which(plot_dict["which"], folder = plot_dict["folder"], name = "")
        plot_cumulative_rewards(plot_dict["rew"], plot_dict["pun"], folder = plot_dict["folder"], name = "", min_max = rew_min_max)
        plot_extrinsic_intrinsic(plot_dict["ext"], plot_dict["cur"], plot_dict["ent"], folder = plot_dict["folder"], name = "", min_max = ext_min_max)
        plot_losses(plot_dict["losses"], too_long = None, d = plot_dict["args"].d, folder = plot_dict["folder"], name = "", trans_min_max = trans_min_max, alpha_min_max = alpha_min_max, actor_min_max = actor_min_max, critic_min_max = critic_min_max)

    for training_name in order:
        new_folder = "saves/{}_done".format(training_name)
        if("{}_done".format(training_name) in all_folders): pass
        else: os.mkdir(new_folder)
        
        folders = [folder for folder in all_folders if "_".join(folder.split("_")[:-1]) == training_name]
        plot_names = ["cumulative", "ext_int", "ext_int_normalized", "loss_agent", "loss_trans", "which", "exits"]
        for name in plot_names:
            images = []
            for folder in folders:
                images.append(Image.open("saves/{}/plots/{}".format(folder, name+".png")))
            new_image = Image.new("RGB", ((len(folders)+1)*images[0].size[0], images[0].size[1]))
            w,h = images[0].size
            arr = np.zeros((h,w,4), float)
            for i, image in enumerate(images):
                new_image.paste(image, (i*image.size[0],0))
                imarr = np.array(image, dtype=np.uint8)
                arr += imarr / len(images)
            arr = np.array(np.round(arr), dtype=np.uint8)
            new_image.paste(Image.fromarray(arr, mode="RGBA"), (len(images)*image.size[0],0))
            new_image.save(new_folder + "/{}.png".format(name))
            
        images = []    
        files = [n + ".png" for n in plot_names]
        for file in files:
            images.append(Image.open(new_folder + "/" + file))
            os.remove(new_folder + "/" + file)
        new_image = Image.new("RGB", (images[0].size[0], len(plot_names)*images[0].size[1]))
        for i, image in enumerate(images):
            new_image.paste(image, (0, i*image.size[1]))
        new_image.save("saves/all_{}_plots.png".format(training_name))
        os.rmdir(new_folder)
    
    # Predictions
    for training_name in order:
        os.mkdir("saves/{}_predictions".format(training_name))
        
        folders = []
        for folder in os.listdir("saves"):
            if("_".join(folder.split("_")[:-1]) == training_name and not folder.split('_')[-1] in ("positions", "predictions")):
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
    make_end_pics(order)
    make_together_pic(order)
    make_mega_vid(order)
    new_text("\n\nDone!")



