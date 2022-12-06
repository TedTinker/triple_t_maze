#%%

import torch
import numpy as np
import pybullet as p
from math import degrees, pi, cos, sin
from itertools import product
from matplotlib import pyplot as plt
from torchvision.transforms.functional import resize

from utils import args, plot_rewards
from arena import get_physics, Arena



# Made an environment! 
class Env():   
    
    def __init__(self, arena_name, args = args, GUI = False):
        self.args = args
        self.GUI = GUI
        self.arena = Arena(arena_name, self.args, self.GUI)
        self.body = None
        self.steps, self.resets = 0, 0
        
    def change(self, args = args, GUI = False):
        if(args != self.args or GUI != self.GUI):
            self.close(True)
            self.args = args
            self.GUI = GUI

    def close(self, forever = False):
        if(self.body != None and not forever):
            p.removeBody(self.body.num, physicsClientId = self.arena.physicsClient)
        if(self.resets % 100 == 99 and self.GUI and not forever):
            p.disconnect(self.arena.physicsClient)
            self.arena.already_constructed = False
            self.arena.physicsClient = get_physics(self.GUI, self.arena.w, self.arena.h)
        if(forever):
            try: p.disconnect(self.arena.physicsClient)  
            except: pass

    def reset(self):
        self.resets += 1; self.steps = 0
        self.body = self.arena.start_arena()
        self.prev_action = torch.tensor([0, 0])
        return(self.get_obs())
    
    def reposition(self, pos, yaw):
        self.body.pos = pos 
        self.body.yaw = yaw
        ors = p.getQuaternionFromEuler([pi/2, 0, yaw])
        p.resetBasePositionAndOrientation(self.body.num, pos, ors, physicsClientId = self.arena.physicsClient)
        
    def get_obs(self):
        image_size = self.args.image_size
        x, y = cos(self.body.yaw), sin(self.body.yaw)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [self.body.pos[0], self.body.pos[1], .4], 
            cameraTargetPosition = [self.body.pos[0] - x, self.body.pos[1] - y, .4], 
            cameraUpVector = [0, 0, 1], physicsClientId = self.arena.physicsClient)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, aspect = 1, nearVal = .01, 
            farVal = 10, physicsClientId = self.arena.physicsClient)
        _, _, rgba, depth, _ = p.getCameraImage(
            width=32, height=32,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
            physicsClientId = self.arena.physicsClient)
        
        rgb = np.divide(rgba[:,:,:-1], 255) * 2 - 1
        d = np.nan_to_num(np.expand_dims(depth, axis=-1), nan=1)
        if(d.max() == d.min()): pass
        else: d = (d.max() - d)/(d.max()-d.min())
        d = d*2 - 1
        rgbd = np.concatenate([rgb, d], axis = -1)
        rgbd = torch.from_numpy(rgbd).float()
        rgbd = resize(rgbd.permute(-1,0,1), (image_size, image_size)).permute(1,2,0)
        spe = torch.tensor(self.body.spe).unsqueeze(0)
        return(rgbd, spe, self.prev_action.float())

    def render(self, view = "body"):
        if(view == "body" or "both"):
            rgbd, _, _ = self.get_obs()
            rgb = (rgbd[:,:,0:3] + 1)/2
            plt.figure(figsize = (5,5))
            plt.imshow(rgb)
            plt.title("Body's view")
            plt.show()
            plt.close()
            plt.ioff()
            print("\n")
            
        if(view == "body"):
            return
    
        x = self.body.pos[0]
        y = self.body.pos[1]
        dist = 2
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [x, y, dist + 1], 
            cameraTargetPosition = [x, y, 0], 
            cameraUpVector = [1, 0, 0], physicsClientId = self.arena.physicsClient)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, aspect = 1, nearVal = 0.001, 
            farVal = dist + 2, physicsClientId = self.arena.physicsClient)
        _, _, rgba, _, _ = p.getCameraImage(
            width=128, height=128,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, 
            physicsClientId = self.arena.physicsClient)
        
        rgb = rgba[:,:,:-1]
        rgb = np.divide(rgb,255)
        plt.figure(figsize = (10,10))
        plt.imshow(rgb)
        plt.title("View from above")
        plt.show()
        plt.close()
        plt.ioff()
    
    def change_velocity(self, yaw_change, speed, verbose = False):
        old_yaw = self.body.yaw
        new_yaw = old_yaw + yaw_change
        new_yaw %= 2*pi
        orn = p.getQuaternionFromEuler([self.body.roll,self.body.pitch,new_yaw])
        p.resetBasePositionAndOrientation(self.body.num,(self.body.pos[0], self.body.pos[1], .5), orn, physicsClientId = self.arena.physicsClient)
        
        old_speed = self.body.spe
        x = -cos(new_yaw)*speed / self.args.steps_per_step
        y = -sin(new_yaw)*speed / self.args.steps_per_step
        p.resetBaseVelocity(self.body.num, (x,y,0), (0,0,0), physicsClientId = self.arena.physicsClient)
        _, self.body.yaw, _ = self.arena.get_pos_yaw_spe(self.body.num)
                
        if(verbose):
            print("\n\nOld yaw:\t{}\nChange:\t\t{}\nNew yaw:\t{}".format(
                round(degrees(old_yaw)) % 360, round(degrees(yaw_change)), round(degrees(new_yaw))))
            print("Old speed:\t{}\nNew speed:\t{}".format(old_speed, speed))
            self.render(view = "body")  
            print("\n")
            
    def real_yaw_spe(self, yaw, spe):
        yaw = [-self.args.max_yaw_change, self.args.max_yaw_change, yaw]
        yaw.sort()
        spe = [self.args.min_speed, self.args.max_speed, spe]
        spe.sort()
        return(yaw[1], spe[1])
  
    def step(self, agent):
        self.steps += 1
        image, speed, prev_action = self.get_obs()
        with torch.no_grad():
            self.body.action, self.body.hidden = agent.act(
                image, speed, prev_action, self.body.hidden)
        self.prev_action = self.body.action
        yaw = -self.body.action[0].item() * self.args.max_yaw_change
        spe = self.args.min_speed + ((self.body.action[1].item() + 1)/2) * \
            (self.args.max_speed - self.args.min_speed)
        yaw, spe = self.real_yaw_spe(yaw, spe)
        self.change_velocity(yaw, spe)
      
        for _ in range(self.args.steps_per_step):
            p.stepSimulation(physicsClientId = self.arena.physicsClient)
        self.body.pos, self.body.yaw, self.body.spe = self.arena.get_pos_yaw_spe(self.body.num)
        end, which, reward = self.arena.end_collisions(self.body.num)
        col = self.arena.other_collisions(self.body.num)
        if(col): reward -= self.args.wall_punishment
        if(not end):  end = self.steps >= self.args.max_steps
        exit = which[0] != "FAIL"
        if(end and not exit): reward = -1
        next_image, next_speed, _ = self.get_obs()

        self.body.to_push.add(
            image.cpu(), speed.cpu(), 
            self.body.action, reward,
            next_image.cpu(), next_speed.cpu(), end)
        return(end, exit, which, self.body.pos)
    
    def step_by_hand(self, yaw, spe, verbose = True):
        self.steps += 1
        self.body.action[0] = yaw
        self.body.action[1] = spe
        yaw = -self.body.action[0].item() * self.args.max_yaw_change
        spe = self.args.min_speed + ((self.body.action[1].item() + 1)/2) * \
            (self.args.max_speed - self.args.min_speed)
        yaw, spe = self.real_yaw_spe(yaw, spe)
        self.change_velocity(yaw, spe, verbose = verbose)
        
        for _ in range(self.args.steps_per_step):
            p.stepSimulation(physicsClientId = self.arena.physicsClient)
        self.body.pos, self.body.yaw, self.body.spe = self.arena.get_pos_yaw_spe(self.body.num)
        end, which, reward = self.arena.end_collisions(self.body.num)
        
        col = self.arena.other_collisions(self.body.num)
        if(col): reward -= self.args.wall_punishment
        if(not end):  end = self.steps >= self.args.max_steps
        exit = which[0] != "FAIL"
        if(end and not exit): reward = -1
        return(end, exit, reward)
      


if __name__ == "__main__":
    env = Env("3", GUI = True)
    env.reset()   
    env.render("body") 
    end = False
    while(end == False):
        print()
        yaw   = input("\nYaw?\n")
        speed = input("\nSpeed?\n")
        if(yaw == ""): yaw = 0
        if(speed == ""): speed = 1
        end, exit, reward = env.step_by_hand(float(yaw), float(speed))
        env.body.to_push.rew.append(reward)
    env.body.to_push.finalize_rewards()
    plot_rewards(env.body.to_push.rew)
    env.close(forever = True)



print("env.py loaded.")
# %%
