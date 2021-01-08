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
        # print(all)
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
        yaml_name = task
        yaml_name_dict=["yw_insd","yw_insf","yw_srw","yw_inss"]
        for i in yaml_name_dict:
            if i in task:
                yaml_name=i

        # yaml_name="yw_insd" if "yw_insd" in task else task
        # yaml_name="yw_insf" if "yw_insf" in task else task
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
            try:
                self.observation_space,self.action_space=self.Env.get_spaces()
            except Exception as err:
                print(err)
                self.observation_space = spaces.Box(0, 255, shape=(3,*self.args.obs_shape), dtype='float32')
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
        return getattr(self.Env, name)
def show_exp(task,expid):
    task = task
    env1 = yw_robotics_env(task, DIRECT=0)

    if(expid=="ALL"):
        for expid in range(21):
            print(expid)
            save_list = env1.get_saves_list(task, expid)
            for i in range(0,len(save_list)):
            # for i in range(170,186):
                env1.load_game(task, expid, i)
                time.sleep(1/240)
    # if(expid=="")
    else:
        save_list = env1.get_saves_list(task, expid)
        while(1):
            for i in range(0,len(save_list)):
            # for i in range(170,186):
                env1.load_game(task, expid, i)
                time.sleep(1/240)

    while (1):
        print("结束")
        time.sleep(1)
def show_save_id_reverse(task,expid):
    task = task
    env1 = yw_robotics_env(task, DIRECT=0)
    while (1):
        save_list = env1.get_saves_list(task, expid)
        for i in reversed(range(0,len(save_list))): #
        # for i in reversed(range(len(save_list)-40,len(save_list)-30)): #
            env1.load_game(task, expid, i)
            time.sleep(0.01)


        print("结束")
        time.sleep(1)

def copy_1_to_0(task):
    path_root=os.path.join(os.getcwd(),"game_saves",task)
    for expid in os.listdir(path_root):
        if("b" in expid):
            continue
        save_path=os.path.join(path_root,expid)
        shutil.copy2(os.path.join(save_path,"1"),os.path.join(save_path,"0"))
class img_save(object):
    def __init__(self,start_counter_int):
        self.img_counter=start_counter_int
    def save(self,img):
        print("img save",self.img_counter)
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        cv2.imwrite("z_imgs/"+str(self.img_counter)+".png",img)
        self.img_counter+=1


if __name__=="__main__":
    # show_exp("yw_reach_v1img",expid="ALL") #0 1
    # show_save_id_reverse("yw_reach_v1img",expid=14)
    # show_save_id_reverse("yw_pick_v1img_5cm",expid=0)
    # copy_1_to_0("yw_reach_v1img")
    # env1 = yw_robotics_env("yw_reach_v1img", DIRECT=0)

    # actor = xbox_actor()
    # xboxCont = XboxController(xbox_actor.controlCallBack, deadzone=30, scale=100, invertYAxis=True)
    # xboxCont.start()
    # actor.set_controller(xboxCont)
    # saver=img_save(10000)
    # env1 = yw_robotics_env("yw_insert_v1img3cm", DIRECT=0)

    taskname="yw_inss_v1"
    env1 = yw_robotics_env(taskname, DIRECT=0, gan_srvs=1, gan_dgx=True, gan_port=5660)

    loop=0
    # action=[0,0]
    ret=0
    while(True):
        loop+=1
        print(loop)
        action=env1.action_space.sample()
        action=[0,0]
        obs, rew, done, info=env1.step(action)

        ret+=rew
        if(done):
            print("ret=",ret," || avret=",ret/loop)
            ret=0
            looo=0
        if(rew>1):
            print("success")
        if(done):
            print("Finished!")

        env1.render()

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
