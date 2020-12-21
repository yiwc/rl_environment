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


expid = 0

class xbox_actor(object):
    def __init__(self):
        pass

    def set_controller(self,xboxcontroller):
        self.controller=xboxcontroller
        self.control_value=xboxcontroller.controlValues

    def get_all(self):
        self.control_value = self.controller.controlValues
        return self.control_value

    def get_action(self):
        # print(1)
        zoom_xyz=0.003
        all=self.get_all()
        action=[0,0,0,0,0,0,0]
        action[0]=  all[1] / 100
        action[1]=  all[0] / 100
        # print(all,'acction',action)

        if ((all[4] < 60 and all[4] > 40)):
            pass
        else:
            action[2]= -(all[4] - 50) / 50
        action[3] = all[7]
        action[3] -= all[8]
        action[4] = all[9] - all[6]
        # action[4] -= all[6]
        action[5] = all[3]/100
        action[6] = (all[5]-50)/50
        # print(all[1])
        # print('acction',action)
        # print(all, 'acction', action)
        # print(action)
        return action

    @staticmethod
    def controlCallBack(xboxControlId, value):
        return

class yw_robotics_env(gym.Env):
    # combine Pybullet Configs and IIWASIM together

    def __init__(self,task,DIRECT=1,exp_recording=0,**kwargs):
        import pybullet as p
        import pybullet_utils.bullet_client as bc

        #import args
        base_dir = os.path.abspath(os.path.dirname(__file__))

        # if "yw_insd" in task:
        yaml_name="yw_insd" if "yw_insd" in task else task
        yaml_name="yw_insf" if "yw_insf" in task else task
        yaml_f=os.path.join(base_dir, "configs/"+yaml_name+".yaml")
        if os.path.isfile(yaml_f):
            self.args = yaml.load(open(yaml_f, 'r'), yaml.Loader)
            self.args = namedtuple('arg', self.args.keys())(**self.args)
        else:
            raise NotImplementedError("No Yaml file, please create a ->"+yaml_f)
            # self.args= None

        self.createVideo = 0
        fps = 240
        self.dt = 1. / fps
        self.DIRECT=DIRECT

        if self.DIRECT:
            p0 = bc.BulletClient(connection_mode=p.DIRECT)
            # p.connect(p.DIRECT)
        elif self.createVideo:
            p0 = p.connect(p.GUI,options="--minGraphicsUpdateTimeMs=0 --mp4=\"pybullet_grasp.mp4\" --mp4fps=" + str(fps))
            # p0 = bc.BulletClient(connection_mode=)
            # p0.connect(p.GUI, )
        else:
            p.connect(p.GUI,
                           options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
            p0=p
            # p0 = bc.BulletClient(connection_mode=p.GUI,options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
            # p0.connect(p.GUI)s

        if not self.DIRECT: # GUI setting
            p0.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
            p0.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p0.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,1)
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1-self.DIRECT)


        p0.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
        p0.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=90, cameraPitch=-22,
                                     cameraTargetPosition=[0.5, 0.2, -0.4])
        p0.setAdditionalSearchPath(pd.getDataPath())
        p0.setTimeStep(self.dt)
        p0.setGravity(0, -9.8, 0)

        self.p0=p0
        self.Env = iiwa_sim.PandaSim(p0, task, self.args,**kwargs)
        self.Env.control_dt = self.dt
        self.Env.expid = expid
        self.Env.exp_recording = exp_recording
        # logId = self.Env.bullet_client.startStateLogging(self.Env.bullet_client.STATE_LOGGING_PROFILE_TIMINGS,os.path.join(os.getcwd(), "logs_timing", "log.json"))
        # self.Env.bullet_client.submitProfileTiming("start")
        if (self.Env.exp_recording):
            print("You are gonna Recording, data will be covered, with expid=" + str(
                self.Env.expid) + ".Press [y] button to go on")
            while (self.Env.if_key_board_detected() != "y"):
                time.sleep(0.01)
        pass

        # gym
        if self.args is not None:
            self.observation_space = spaces.Box(0, 255, shape=(3,*self.args.img_size), dtype='float32')
            self.action_space = spaces.Box(self.args.action_low,self.args.action_high, shape=(self.args.action_len,), dtype='float32')
            self.viewer = None


    def step(self,action):
        timestep=self.Env.step(action)
        # reward,obs,done=self.Env.step(action)
        obs=timestep.get_obs()
        rew=float(timestep.get_reward())
        done=timestep.get_done()
        info={}
        if self.createVideo:
            self.p0.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        if not self.createVideo:
            time.sleep(self.dt)
        if not self.DIRECT:
            time.sleep(self.dt)
        # print("rew,done",rew,done)
        return obs, rew, done, info
        # return timestep

    def reset(self):
        timestep=self.Env.reset()
        obs=timestep.get_obs()
        # print("obs=",obs)
        return obs
    def render(self, mode='human'):
        self.Env.render(0)

    def close(self):
        self.p0.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __getattr__(self, name):
        # try:
        return getattr(self.Env, name)
        # except:
        #     return

if __name__=="__main__":

    actor = xbox_actor()
    xboxCont = XboxController(xbox_actor.controlCallBack, deadzone=30, scale=100, invertYAxis=True)
    xboxCont.start()
    actor.set_controller(xboxCont)

    taskname="yw_insf_v1"
    env1 = yw_robotics_env(taskname, DIRECT=0,gan_srvs=4)

    loop=0
    ret=0
    while(True):
        env1.test_robot()
        p.stepSimulation()
        time.sleep(0.04)
        continue

        loop+=1
        action = actor.get_action()
        obs, rew, done, info=env1.step(action) # env1.step(np.random.random([2])-0.3)

        ret+=rew
        if(done):
            print("ret=",ret," || avret=",ret/loop)
            ret=0
            looo=0
        if(rew>1):
            print("success")
        if(done):
            print("Finished!")
        # env1.render()
