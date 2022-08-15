#%%

# What is an agent's body?
import torch
from random import choices
from itertools import product

from utils import args, add_discount, arena_dict, new_text

class to_push:
    def __init__(self, GAMMA):
        self.GAMMA = GAMMA
        self.obs = []
        self.spe = []
        self.act = []
        self.rew = []
        self.next_obs = []
        self.next_spe = []
        self.done = []

    def add(self, obs, spe, act, rew, next_obs, next_spe, done):
        self.obs.append(obs)
        self.spe.append(spe)
        self.act.append(act)
        self.rew.append(rew)
        self.next_obs.append(next_obs)
        self.next_spe.append(next_spe)
        self.done.append(done)
        
    def finalize_rewards(self):
        for i in range(len(self.rew)):
            if(self.rew[i] > 0):
                self.rew[i] = self.rew[i]* (self.GAMMA**i)
        self.rew = add_discount(self.rew, .9)
        
    def push(self, memory):
        for _ in range(len(self.rew)):
            done = self.done.pop(0)
            memory.push(self.obs.pop(0), self.spe.pop(0), self.act.pop(0), \
                self.rew.pop(0), self.next_obs.pop(0), self.next_spe.pop(0), done, done)
            
    def empty(self):
        self.obs = []
        self.spe = []
        self.act = []
        self.rew = []
        self.next_obs = []
        self.next_spe = []
        self.done = []

        

class Body:
    def __init__(self, num, pos, spe, roll, pitch, yaw, GAMMA):
        self.num = num
        self.pos = pos; self.spe = spe
        self.roll = roll; self.pitch = pitch; self.yaw = yaw
        self.age = 0
        self.action = torch.tensor([0.0, 0.0])
        self.hidden = None
        self.to_push = to_push(GAMMA)



# How to make physicsClients.
import pybullet as p

def get_physics(GUI, w, h):
    if(GUI):
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
    else:   
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath("pybullet_data/")
    return(physicsClient)



# Get arena from image.
import numpy as np
from math import pi, sin, cos
import cv2
from string import ascii_uppercase as LETTERS

class Arena():
    def __init__(self, arena_name, args = args, GUI = False):
        self.args = args
        self.arena_name = arena_name
        self.arena_dict = arena_dict[arena_name + ".png"]
        self.arena_map = cv2.imread("arenas/" + arena_name + ".png")
        self.w, self.h, _ = self.arena_map.shape
        self.physicsClient = get_physics(GUI, self.w, self.h)
        self.start = self.arena_dict[0]
        self.ends = {}
        self.colors = {}
        self.already_constructed = False

    def start_arena(self):
        if(not self.already_constructed):
            p.loadURDF("plane.urdf", [0,0,0], globalScaling = .5,
                       useFixedBase = True, physicsClientId = self.physicsClient) 
            p.loadURDF("plane.urdf", [10,0,0], globalScaling = .5,
                       useFixedBase = True, physicsClientId = self.physicsClient) 
            p.loadURDF("plane.urdf", [0,10,0], globalScaling = .5,
                       useFixedBase = True, physicsClientId = self.physicsClient) 
            p.loadURDF("plane.urdf", [10,10,0], globalScaling = .5,
                       useFixedBase = True, physicsClientId = self.physicsClient) 
            self.ends = {}
            end_num = 0
            end_names = [self.arena_name + LETTERS[i] for i in range(len(self.arena_dict[1].keys()))]
            for loc in ((x,y) for x in range(self.w) for y in range(self.h)):
                pos = [loc[0],loc[1],.5]
                if((self.arena_map[loc] == [255]).all()):
                    if(loc in list(self.arena_dict[1].keys())):
                        end = ((pos[0]-.5, pos[0] + .5), (pos[1] - .5, pos[1] + .5))
                        self.ends[end_names[end_num]] = (end, self.arena_dict[1][loc])
                        end_num += 1
                else:
                    ors = p.getQuaternionFromEuler([0,0,0])
                    color = self.arena_map[loc][::-1] / 255
                    color = np.append(color, 1)
                    cube_size = 1/self.args.boxes_per_cube
                    cubes = [p.loadURDF("cube.urdf", (pos[0]+i*cube_size, pos[1]+j*cube_size, pos[2]+k*cube_size), 
                                    ors, globalScaling = cube_size, useFixedBase = True, physicsClientId = self.physicsClient) \
                                        for i, j, k in product([l/2 for l in range(-self.args.boxes_per_cube+1, self.args.boxes_per_cube+1, 2)], repeat=3)]
                    bigger_cube = p.loadURDF("cube.urdf", pos, ors, globalScaling = 1.1,
                                    useFixedBase = True, 
                                    physicsClientId = self.physicsClient)
                    self.colors[bigger_cube] = (0,0,0,0)
                    for cube in cubes:
                        self.colors[cube] = color
            self.already_constructed = True
            
            self.colorize()
            #p.saveWorld("arenas/" + self.args.arena_name + ".urdf")
                
        inherent_roll = pi/2
        inherent_pitch = 0
        yaw = 0
        spe = self.args.min_speed
        color = [1,0,0,1]
        file = "ted_duck.urdf"
        
        pos = (self.start[0], self.start[1], .5)
        orn = p.getQuaternionFromEuler([inherent_roll,inherent_pitch,yaw])
        num = p.loadURDF(file,pos,orn,
                           globalScaling = self.args.body_size, 
                           physicsClientId = self.physicsClient)
        p.changeDynamics(num, 0, maxJointVelocity=10000)
        x, y = cos(yaw)*spe, sin(yaw)*spe
        p.resetBaseVelocity(num, (x,y,0),(0,0,0), physicsClientId = self.physicsClient)
        p.changeVisualShape(num, -1, rgbaColor = color, physicsClientId = self.physicsClient)
        body = Body(num, pos, spe, inherent_roll, inherent_pitch, yaw, self.args.gamma)
                
        return(body)
    
    def colorize(self):
        for cube, color in self.colors.items():
            p.changeVisualShape(cube, -1, rgbaColor = color, physicsClientId = self.physicsClient)
        
    def get_pos_yaw_spe(self, num):
        pos, ors = p.getBasePositionAndOrientation(num, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors)[-1]
        (x, y, _), _ = p.getBaseVelocity(num, physicsClientId = self.physicsClient)
        spe = (x**2 + y**2)**.5
        return(pos, yaw, spe)
    
    def pos_in_box(self, num, box):
        (min_x, max_x), (min_y, max_y) = box 
        pos, _, _ = self.get_pos_yaw_spe(num)
        in_x = pos[0] >= min_x and pos[0] <= max_x 
        in_y = pos[1] >= min_y and pos[1] <= max_y 
        return(in_x and in_y)
    
    def end_collisions(self, num):
        col = False
        which = ("FAIL", -1)
        reward = 0
        for end_name, (end, end_reward) in self.ends.items():
            if self.pos_in_box(num, end):
                col = True
                which = (end_name, end_reward)
                reward = end_reward
        if(type(reward) in (int, float)): pass
        else:
            weights = [w for w, r in reward]
            reward_index = choices([i for i in range(len(reward))], weights = weights)[0]
            reward = reward[reward_index][1]
        return(col, which, reward)
    
    def other_collisions(self, num):
        col = False
        for cube in self.colors.keys():
            if 0 < len(p.getContactPoints(num, cube, physicsClientId = self.physicsClient)):
                col = True
        return(col)

if __name__ == "__main__":
    arena = Arena(arena_name = "1", GUI = False) #True)
    arena.start_arena()
    
new_text("arena.py loaded.")
# %%
