import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import my_robots.iiwa_sim as iiwa_sim
import os
import shutil
from XboxController import XboxController
import pybullet_utils.bullet_client as bc


#
expid = 0
# set controller

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

class yw_robotics_env(object):
    # combine Pybullet Configs and IIWASIM together

    def __init__(self,task,DIRECT=1,exp_recording=0):
        import pybullet as p
        import pybullet_utils.bullet_client as bc
        self.createVideo = 0
        fps = 240.
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
            p0 = bc.BulletClient(connection_mode=p.GUI)
            # p0.connect(p.GUI)

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
        self.Env = iiwa_sim.PandaSim(p0, task)
        self.Env.control_dt = self.dt
        self.Env.expid = expid
        self.Env.exp_recording = exp_recording
        # logId = self.Env.bullet_client.startStateLogging(self.Env.bullet_client.STATE_LOGGING_PROFILE_TIMINGS,
        #                                             os.path.join(os.getcwd(), "logs_timing", "log.json"))
        # self.Env.bullet_client.submitProfileTiming("start")
        if (self.Env.exp_recording):
            print("You are gonna Recording, data will be covered, with expid=" + str(
                self.Env.expid) + ".Press [y] button to go on")
            while (self.Env.if_key_board_detected() != "y"):
                time.sleep(0.01)
        pass

    def step(self,action):
        timestep=self.Env.step(action)
        # reward,obs,done=self.Env.step(action)

        if self.createVideo:
            self.p0.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        if not self.createVideo:
            time.sleep(self.dt)
        if not self.DIRECT:
            time.sleep(self.dt)
        return timestep

    def reset(self):
        return self.Env.reset()
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

if __name__=="__main__":
    # show_exp("yw_reach_v1img",expid="ALL") #0 1
    # show_save_id_reverse("yw_reach_v1img",expid=14)
    # show_save_id_reverse("yw_pick_v1img_5cm",expid=0)
    # copy_1_to_0("yw_reach_v1img")
    # env1 = yw_robotics_env("yw_reach_v1img", DIRECT=0)


    actor = xbox_actor()
    xboxCont = XboxController(xbox_actor.controlCallBack, deadzone=30, scale=100, invertYAxis=True)
    xboxCont.start()
    actor.set_controller(xboxCont)
    env1 = yw_robotics_env("yw_insert_v1img3cm", DIRECT=0)
    loop=0
    # action=[0,0]
    while(True):
        loop+=1
        print(loop)
        action = actor.get_action()
        step_fb=env1.step(action[0:2]) # env1.step(np.random.random([2])-0.3)
        if(step_fb.reward):
            print("success")
        if(step_fb.done):
            print("Finished!")
        # print(step_fb.reward)
        # env1.render(0)
        # env1.render(1)

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