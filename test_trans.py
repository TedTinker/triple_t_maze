### Play an episode by hand and test transitioner.

#%%

from utils import plot_rewards, load_agent
from env import Env 
from agent import Agent

import torch
import matplotlib.pyplot as plt
from math import degrees
from copy import deepcopy

def plot_predition(actions, image, predicted_image, predicted_speed, next_image, next_speed, args):
    yaws = [] ; spes = []
    while actions != []:
        yaw = actions.pop(0) * args.max_yaw_change
        yaw = round(degrees(yaw))
        spe = args.min_speed + ((actions.pop(0) + 1)/2) * \
            (args.max_speed - args.min_speed)
        spe = round(spe)
        yaws.append(yaw)
        spes.append(spe)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.title.set_text("Before")
    ax1.imshow(image)
    ax1.axis('off')
    ax2.title.set_text("Prediction: {} speed".format(predicted_speed))
    ax2.imshow(predicted_image)
    ax2.axis('off')
    ax3.title.set_text("After: {} speed".format(next_speed))
    ax3.imshow(next_image)
    ax3.axis('off')
    title = ""
    for i in range(args.lookahead):
        title += "Action: {} degrees, {} speed".format(yaws[i], spes[i])
        if(i < args.lookahead): title += "\n"
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=1.2)
    plt.show()
    plt.close()
    print()

def play(folder, suf, arena_name):
    args = torch.load(folder + "/args.pt")
    agent = Agent(args = args)
    agent = load_agent(agent, folder, suf)
    env = Env(arena_name, GUI = True, args = args)
    env.reset()   
    env.render("body") 
    hidden = None 
    
    while(True):
        
        actual_image, actual_speed = env.get_obs()
        _, hidden = agent.act(actual_image.detach(), actual_speed.detach(), hidden)
        
        testing = input("\nTest transitioner? y/n\n")
        
        while(testing == "y"):
            print("\nInput transitioner-testing actions:\n")
            actions = [] 
            for _ in range(env.args.lookahead):
                yaw   = input("\nYaw?\n") ; speed = input("\nSpeed?\n")
                if(yaw == ""): yaw = 0
                if(speed == ""): speed = 1
                actions.append(float(yaw)) ; actions.append(float(speed))
            torch_image = actual_image.unsqueeze(0).unsqueeze(0)
            torch_speed = actual_speed.unsqueeze(0).unsqueeze(0)
            action_tensor = torch.tensor(actions).unsqueeze(0).unsqueeze(0)
            predicted_image, predicted_speed = agent.transitioner(
                torch_image.detach(), 
                torch_speed.detach(), 
                action_tensor.detach(), 
                hidden)
            predicted_image = predicted_image.cpu().detach().squeeze(0).squeeze(0)
            predicted_speed = predicted_speed.cpu().detach().squeeze(0).squeeze(0)
            
            old_pos, old_yaw = env.body.pos, env.body.yaw
            actions_copy = deepcopy(actions)
            while actions != []:
                yaw = actions.pop(0)
                speed = actions.pop(0)
                _, _, _ = env.step_by_hand(yaw, speed, verbose = False)
            real_image, real_speed = env.get_obs()
            env.reposition(old_pos, old_yaw)           
            predicted_speed = round(predicted_speed.item(),2)#round(env.args.min_speed + (env.args.max_speed - env.args.min_speed) * ((predicted_speed.item()+1)/2), 2)
            plot_predition(
                actions_copy, 
                (actual_image[:,:,:-1]+1)/2, 
                (predicted_image[:,:,:-1]+1)/2, 
                predicted_speed,
                (real_image[:,:,:-1]+1)/2, 
                round(real_speed.item(),2),
                env.args)
            testing = input("\nTest transitioner? y/n\n")
        
        print()
        yaw   = input("\nYaw?\n")
        speed = input("\nSpeed?\n")
        if(yaw == ""): yaw = 0
        if(speed == ""): speed = 1
        _, _, _ = env.step_by_hand(float(yaw), float(speed))
            
play(r"/home/ted/Desktop/examples/bad_exits_lookahead_2/entropy_2_curious_2_001", "04500", "3")
# %%
