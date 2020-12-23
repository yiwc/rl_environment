import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import my_robots.iiwa_sim as iiwa_sim
import os
import shutil
import gym
from XboxController import XboxController
import pybullet_utils.bullet_client as bc
import yaml
from collections import namedtuple
import cv2
from gym import spaces
from gym.utils import seeding
from My_Env import *

expid = 0


if __name__=="__main__":
    taskname="yw_insert_v4img3cm"
    taskname="yw_insert_g1cimg"
    taskname="sparse3"
    # env1 = yw_robotics_env(taskname, DIRECT=0,gan_srvs=4)

    show_exp(taskname,1)

    # loop=0
    # ret=0
    # while(True):
    #     loop+=1
    #     action = np.random.uniform(-1,1,[env1.args.action_len])
    #     obs, rew, done, info=env1.step(action) # env1.step(np.random.random([2])-0.3)
    #     ret+=rew
    #     if(done):
    #         print("ret=",ret," || avret=",ret/loop)
    #         ret=0
    #         looo=0
    #     if(rew>1):
    #         print("success")
    #     if(done):
    #         print("Finished!")
    #     env1.render()

# Exp Recording Scripts
# if __name__=="__main__":
#     # Expert Controller

#     task="yw_reach_v1img"
#     my_env=yw_robotics_env(task,DIRECT=0,exp_recording=1)
#
#     train=1
#     if(train):
#         my_env.reset()
#         e=0
#         for s in range(100000):
#             # print(s)
#             action=actor.get_action()
#             timestep=my_env.step(action)
#             if(timestep.done):
#                 my_env.reset()
#                 e+=1
#                 print("e,s=",e,",",s)
