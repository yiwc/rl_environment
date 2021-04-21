import sys
try:
    rospth1='/opt/ros/kinetic/lib/python2.7/dist-packages'
    rospth2='/home/radoe-1/movo_ws/devel/lib/python2.7/dist-packages'
    # # rospth='/opt/ros/kinetic/lib/python2.7/dist-packages'
    # if rospth in sys.path:
    sys.path.remove(rospth1)
    sys.path.remove(rospth2)
except:
    pass
from My_Env import yw_robotics_env
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
import pandas as pd
import matplotlib.pyplot as plt

if __name__=="__main__":

    taskname="alpoderl"
    env1 = yw_robotics_env(taskname, DIRECT=1, gan_srvs=1, gan_dgx=True, gan_port=5660)

    obs_data=[]
    for i in range(5000):
        print('stp=',i)
        action=env1.action_space.sample()
        # try:
        obs, rew, done, info=env1.step(action)
        obs_data.append(obs)
        env1.render()
        # input("test")
    df=pd.DataFrame(np.array(obs_data),columns=["theta_cam0","theta_cam1"])
    df.plot.kde(bw_method=0.1)
    plt.show()

    # input("test")
    # print(obs_data)