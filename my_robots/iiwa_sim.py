import time
import numpy as np
import math
import os
import pybullet_data
import random
import pickle
import copy
import shutil
import cv2
# import gym
# from keras.models import load_model
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# import tensorflow as tf
import zmq
try:
    from gym.envs.classic_control import rendering
except:
    pass
useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11  # 8
pandaNumDofs = 7

kukaEndEffectorIndex = 7  # 8

jacoEndEffectorIndex = 6  # 8
# kukaNumDofs = 7 # DEBUG
# KUKA_GRIPPER_INDEX = 8

MAX_FORCE=50
FINGER_FORCE=5000
FINGER_TIP_FORCE=50

ROBOT_BASE_POS_1_DEFAULT=[5,0,0]# PAND
ROBOT_BASE_POS_3_DEFAULT=[0.6,0.8,-1]# jaco
ROBOT_BASE_ORN_DEFAULT=[ 0, -0.7071068, 0, 0.7071068 ] #p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])


ROBOT_ENDEFFECT_SAFE_POSE=[0.1,0.4,-0.5] # IIWA # x z y
ROBOT_ENDEFFECT_SAFE_ORN_EU=[math.pi / 2., 0., 0.] # IIWA



# KUKA CONSTRAIN
# lower limits for null space
KUKA_LL = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
# upper limits for null space
KUKA_UL = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
# joint ranges for null space
KUKA_JR = [5.8, 4, 5.8, 4, 5.8, 4, 6]
# restposes for null space
KUKA_RP = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
# joint damping coefficents
KUKA_JD = [
    0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
    0.00001, 0.00001, 0.00001, 0.00001
]


# lower limits for null space
JACO_ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
# upper limits for null space
JACO_ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
# joint ranges for null space
JACO_jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
# restposes for null space
JACO_rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
JACO_DEFAULT_JP=[
    0, 3.14, 0, 3.14,  0, 3.14, 0, # 0123456 Link1-7
    # -2.6, 2.0, 0.0, 2.0, 0.0, 0.0, 1.0, # 0123456 Link1-7
    #0, # 7 endeffector
    0,#0, # 89 finger 1
    0,#0, # 10 11 finger 2
    0,#0, # 12 13 finger 3
    #0  # 14 base
]
# JACO_INSERT_JP=#[1.9926424422805127, 1.6617884211189398, 1.6349933203463376, 1.3846616707724921, -0.3924618025379057, 3.395157502583005, -2.376403383785282, 1.3, 1.3, 1.3]
JACO_INSERT_JP=[1.9450396065385336, 1.628769735977028, 1.5838618332079535, 1.273678168872743, -0.3930265152848413, 3.226788116105112, -2.3583587290817274, 1.3, 1.3, 1.3]


# TODO define jaco LL UL JR limits

# PANDA_LL = [-7] * pandaNumDofs
# PANDA_UL = [7] * pandaNumDofs
# PANDA_JR = [7] * pandaNumDofs
# restposes for null space
# fPANDA_jointPositions = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
# PANDA_RP = PANDA_jointPositions
# class fake_args(object):
#     def __init__(self):
#         pass
class spec(object):
    def __init__(self, maximum,minimum,shape):
        self.maximum=maximum
        self.minimum=minimum
        self.shape=shape
# class observatio_spec(object):
#     def __init__(self):

class TimeStep(object):
    def __init__(self,sim):
        self.sim=sim
        self.reward=0
        self.discount=1
        self.done = 0
        self.observation={}
        self.success=0

    def load(self):
        self.reward=self.sim._reward()
        self.observation=self.sim._observe()
        self.done = self.sim._terminal()
        return self
    def get_obs(self):
        return self.observation
    def get_done(self):
        return self.done
    def get_reward(self):
        return self.reward
    # @property
    def last(self):
        return self.done

class PandaSim(object):
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("the simulator has closed!")
    def __init__(self, bullet_client,Task,args,root_folder_name="rl_environment",gan_srvs=1,gan_dgx=False,**kwargs):
        print("YOURTASK=",Task)

        self.gan_srvs=gan_srvs
        self.gan_dgx=gan_dgx
        self.root_folder_name=root_folder_name

        self.Task = Task
        self.task= Task
        self.timestep=TimeStep(sim=self)
        self.temp_timestep=TimeStep(sim=self)
        self.pyb=bullet_client
        self.p=bullet_client
        self.bullet_client=bullet_client
        self.args=args
        #log
        self.allows_log=False
        # self.logId = self.p.startStateLogging(self.p.STATE_LOGGING_PROFILE_TIMINGS, "chrome_about_tracing.json")
        self.logs_steps=0

        # from gym.envs.classic_control import rendering

        self.image=None
        self.img_dirpth = "z_imgs/{}".format(str(time.time()).split(".")[0])

        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)

        self.taskobj=self.get_taskobj_by_name(self.Task)

        self.control_dt = 1. / 240.# will be set outside the loop

        #Recording
        self.exp_recording=0 # 自动保存所有的轨迹路线，每一个step
        self.expid=None
        self.saves_list_dict={}# {'task':{'expid':[1,2,3,4...]}}
        self.exp_num=0
        self.action=[] # to be given by user or agent

        # Robust Paras
        self.hash_update_per_epis=0
        self.hash_rt15=0.9

        # Load Items and Robot
        self.taskobj.init_load_items()

        # Reset state
        self.reset()

    # pending
    def get_taskobj_by_name(self,name):

        # from my_robots.task import sparse3
        if "yw_insd" in name:
            n="yw_insd"
            level=int(name.split("v")[1])
            return eval(n + "(self,level)")

        if "yw_insf" in name:
            n="yw_insf"
            level=int(name.split("v")[1])
            return eval(n + "(self,level)")

        return eval(name+"(self)")
        pass


    # MARCON

    def _get_model_rootdir(self):
        dir=os.getcwd()
        dirs=[self.root_folder_name,"my_robots","my_models"]
        for d in dirs:
            if(d not in dir.__str__()):
                dir=os.path.join(dir,d)
        return dir
    def _get_textures_rootdir(self):
        dir=os.getcwd()
        dirs=[self.root_folder_name,"my_robots","textures"]
        for d in dirs:
            if(d not in dir.__str__()):
                dir=os.path.join(dir,d)
        return dir
    def _get_x_rootdir(self,x):
        dir=os.getcwd()
        dirs=[self.root_folder_name,"my_robots",x]
        for d in dirs:
            if(d not in dir.__str__()):
                dir=os.path.join(dir,d)
        return dir

    def _update_hash_per_epis(self):
        # self.last_hash_update_per_epis=self.hash_update_per_epis
        self.hash_update_per_epis=random.random()

    # Learning Framework
    def reset(self): # Set the env to step = 0 and state from zero state
        # init some flags
        self.Task_Success_Updated = 0
        self._update_hash_per_epis()
        if(self.exp_recording):
            self.clean_game_svae_buffer()
        # Reset Time
        self.t = 0.
        self.steps = 0
        self.Task_Success_Updated=0

        self.taskobj.reset()

        #Load observation and so on
        self.timestep.load()

        return self.timestep

    def step(self,action):

        self.logs_steps+=1
        if(self.if_key_board_detected()):
            return

        def step_t(self):
            self.t += self.control_dt
            self.steps+=1
        step_t(self)

        # print(self.steps)

        self.taskobj._apply_action_to_sim(action)
        self.action=action[:]
        self.bullet_client.stepSimulation()
        self.timestep.load()
        self.Task_Success_Updated=0
        done=self.timestep.get_done()

        # print(self.detect_coli_case)

        if(done):
            if(self.exp_recording):
                print("Do You Like Last Recording? Y/N")
                while(True):
                    press=self.if_key_board_detected()
                    if(press=="y"):
                        self.save_game_into_disk(self.expid)
                        self.expid+=1
                        print("OK，开始下一次录制。expid=",self.expid)
                        break
                    elif(press=="n"):
                        print("没关系，重新录制一次。expid=",self.expid)
                        break
                    time.sleep(0.001)

            self.temp_timestep.load()
            self.reset()

        if(self.exp_recording):
            self.save_game_into_buffer(self.expid)

        if(done):
            return self.temp_timestep
        else:
            return self.timestep

    def _observe(self):
        return self.taskobj._observe()
    def _reward(self):
        return self.taskobj._reward()
    def _terminal(self):
        return self.taskobj._terminal()
    def _task_success(self):
        return self.taskobj._task_success()
    def _get_external_observe(self):
        return self.taskobj._get_external_observe()


    # Detection
    def detect_coli(self, item_id):
        return len(
            self.bullet_client.getContactPoints(
                self.RobotUid,
                item_id
            )) > 0

    # Property
    def action_spec(self):
        return self.taskobj._action_spec()
    def observation_spec(self):
        raise NotImplementedError("you havent set image spec, may this is not useful")
    def get_end_effect_pose(self, ee_index=kukaEndEffectorIndex):
        state = self.bullet_client.getLinkState(self.RobotUid, ee_index)
        # print("state", state
        return list(state[4])  # state[4] is the worldLinkFramePosition
    def get_bolt_pose(self):
        # Only For Kinova Robut with Bolt
        state = self.bullet_client.getLinkState(self.RobotUid, 6)
        pos=list(state[4])
        pos[2]+=0.3
        return  pos# state[4] is the worldLinkFramePosition
    def get_end_effect_orn(self, ee_index=kukaEndEffectorIndex):
        state = self.bullet_client.getLinkState(self.RobotUid, ee_index)
        # print("state", state
        # print("state,",state)
        return list(state[5])  # state[4] is the worldLinkFramePosition
    def get_end_effect_pos_orn(self, ee_index=kukaEndEffectorIndex):
        state = self.bullet_client.getLinkState(self.RobotUid, ee_index)
        # print("state", state
        return list(state[4:6])  #

    @property
    def _info(self):
        info = {}
        info["target_e_pose"] = self.target_e_pose
        info["target_e_orn"] = self.target_e_orn
        info["steps"] = self.steps
        info["maxsteps"] = self.maxsteps
        info["robot_pose"] = self.bullet_client.getBasePositionAndOrientation(self.my_items["robot"][0])
        info["lego[0]_pose"] = self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0])
        return info

    # Expert Recordings
    def save_game_into_buffer(self,expid):

        steps=self.steps
        file_dir_path=os.path.join("game_saves",self.Task,"buffer")

        os.makedirs(file_dir_path,exist_ok=True)

        # assert

        # shutil.rmtree(file_dir_path)
        # os.mkdir(file_dir_path) # clean the buffer
        # if(not os.path.exists(file_dir_path)):
        #
        #     os.mkdir(file_dir_path) # clean the buffer
            # raise NotImplemented("Sorry, pls create the dir mannually~")

        file_path=os.path.join(file_dir_path,str(steps)+"")

        # get saving data
        # data={"test":1}
        data=[]

        config_dict={"name":"config",
                     "target_e_pose":self.target_e_pose,
                     "target_e_orn":self.target_e_orn,
                     "steps":self.steps,
                     "maxsteps":self.maxsteps,
                     "Task":self.Task}
        item_dict={"name":"None","sub_id":0,"pos":[0,0,0],"orn":[0,0,0,0],"v_pos":[0,0,0],"v_ang":[0,0,0]}

        action_dict={"name":"action","sub_id":0,
                     "action":self.action}

        observe_dict=self.taskobj.get_observe_dict()
        observe_dict["name"]="observation"
        observe_dict["subid"]=0

        # Save Items states
        for name in self.my_items.keys():
            for sub_id in range(len(self.my_items[name])):
                if(name in ["robot"]):
                    continue
                uid=self.my_items[name][sub_id]
                item_dict_i = item_dict.copy()
                item_dict_i["name"]=name
                item_dict_i["sub_id"]=sub_id
                pos, orn = self.bullet_client.getBasePositionAndOrientation(uid)
                v_pos, v_ang = self.bullet_client.getBaseVelocity(uid)
                item_dict_i["pos"]=pos
                item_dict_i["orn"]=orn
                item_dict_i["v_pos"]=v_pos
                item_dict_i["v_ang"]=v_ang
                data.append(item_dict_i.copy())
        # Save Robots States
        for sub_id in range(len(self.my_items["robot"])):
            uid=self.my_items["robot"][sub_id]
            joint_states=self.bullet_client.getJointStatesMultiDof(uid,[i for i in range(self.bullet_client.getNumJoints(uid))])
            # print(joint_states)
            robot_dict = {"name": "robot",
                          "sub_id":sub_id,
                          "jp": [],
                          "jv": [],
                          "jointReactionForces": [],
                          "appliedJointMotorTorque": []}
            for item in joint_states:
                if(any(item[0])):
                    jp=item[0][0]
                    jv=item[1][0]
                else:
                    jp=0
                    jv=0
                # jrf=item[2][0]
                # ajmt=item[3][0]
                robot_dict["jp"].append(jp)
                robot_dict["jv"].append(jv)
                # robot_dict["jointReactionForces"].append(jrf)
                # robot_dict["appliedJointMotorTorque"].append(ajmt)
            data.append(robot_dict)
        data.append(config_dict)

        # Save User's action input
        data.append(action_dict)
        # Save Observe
        data.append(observe_dict)

        save_file = open(file_path, "wb")
        pickle.dump(data, save_file)
        save_file.close()
        return
    def clean_game_svae_buffer(self):
        file_dir_rood=os.path.join("game_saves",self.Task)
        if(os.path.exists(file_dir_rood) == False):
            os.mkdir(file_dir_rood)

        file_dir_path_old = os.path.join("game_saves", self.Task, "buffer")
        if(os.path.exists(file_dir_path_old)):
            shutil.rmtree(file_dir_path_old)
            os.mkdir(file_dir_path_old)
        else:
            os.mkdir(file_dir_path_old)
    def save_game_into_disk(self,expid):
        steps = self.steps
        file_dir_path_old = os.path.join("game_saves", self.Task, "buffer")
        # shutil.rmtree(file_dir_path_old)
        # if (not os.path.exists(file_dir_path)):
        #     os.mkdir(file_dir_path)
        file_dir_path_new = os.path.join("game_saves", self.Task, str(expid))
        if(os.path.exists(file_dir_path_new)):
            os.removedirs(file_dir_path_new)
            print("Old Records has been deleted!->",file_dir_path_new)
        os.rename(file_dir_path_old,file_dir_path_new)
        # shutil.rmtree(file_dir_path_old)
        self.clean_game_svae_buffer()
        print("New Records Saved!->",file_dir_path_new)
    def get_saves_list(self,task,expid):
        if(task in self.saves_list_dict.keys()):
            if(type(self.saves_list_dict[task]) is dict):
                pass
            else:
                self.saves_list_dict[task]={}
            if(expid in self.saves_list_dict[task].keys()):
                return self.saves_list_dict[task][expid]
            else:
                file_dir_path = os.path.join("game_saves", task, str(expid))
            # print(file_dir_path)
            try:
                assert os.path.exists(file_dir_path)
            except:
                pass
                # if (not os.path.exists(file_dir_path)):
                #     os.mkdir(file_dir_path)
            flist = os.listdir(file_dir_path)
            # print(flist)
            llist = [int(name) for name in flist]
            llist.sort()
            # print(llist)
            flist = [str(i) for i in llist]
            self.saves_list_dict[task][expid]=flist
            return self.saves_list_dict[task][expid]
        else:
            self.saves_list_dict[task]={}
            file_dir_path = os.path.join("game_saves", task, str(expid))
            try:
                assert os.path.exists(file_dir_path)
            except:
                pass
            # if (not os.path.exists(file_dir_path)):
            #     os.mkdir(file_dir_path)
            flist=os.listdir(file_dir_path)
            # print(flist)
            llist=[int(name) for name in flist]
            llist.sort()
            # print(llist)
            flist=[str(i) for i in llist]
            self.saves_list_dict[task][expid]=flist
            return flist
    def get_exp_num(self,task):
        # if(self.exp_num!=0):
        #     return self.exp_num
        # else:
        file_dir_path = os.path.join("game_saves", task)
        flist = os.listdir(file_dir_path)
        self.exp_num=0
        for f in flist:
            if "buffer" in f:
                pass
            else:
                self.exp_num+=1
        # self.exp_num=len(flist)
        return self.exp_num
    def load_game(self,task,expid,saveid): # it should like reset, but at steps=x
        # task : yw_pick_v1
        # expid : 1
        # saveid: 0  or saveid:1   or saveid:2 ...

        # print("You have load to,",task," exp=",expid," saveid=",saveid)
        saveid=int(saveid)
        saves_list = self.get_saves_list(task=task,expid=expid)
        saves_count = len(saves_list)
        try:
            assert saves_count>saveid
        except:
            pass
        try:
            steps=saves_list[saveid]
        except Exception as err:
            print(err)
            pass

        file_dir_path = os.path.join("game_saves", self.Task, str(expid))
        assert os.path.exists(file_dir_path)
        # if (not os.path.exists(file_dir_path)):
        #     os.mkdir(file_dir_path)

        file_path = os.path.join(file_dir_path, str(steps) + "")

        save_file = open(file_path, "rb")
        data = pickle.load(save_file)

        # print("Data:",data)

        robot_data=[]
        config_data={}
        for item_dict in data:
            if('name' in item_dict.keys()):
                name=item_dict["name"]
            else:
                continue
            if name=="robot":
                # print(item_dict)
                robot_data.append(item_dict.copy())
                continue
            if name == "config":
                config_data=item_dict.copy()
                continue
            if name in ["action","observation"]:
                continue
            sub_id = item_dict["sub_id"]
            pos = item_dict["pos"]
            orn = item_dict["orn"]
            v_pos = item_dict["v_pos"]
            v_ang = item_dict["v_ang"]
            uid=self.my_items[name][sub_id]
            # self.my_items[name][sub_id].
            self.bullet_client.resetBasePositionAndOrientation(uid,pos,orn)
        for robot_dict in robot_data:
            # print(robot_dict)
            sub_id =robot_dict["sub_id"]
            jp = robot_dict["jp"]
            jv = robot_dict["jv"]
            jp =[[jpi] for jpi in jp]
            jv =[[jvi] for jvi in jv]
            # print(len(jp))
            # print(len(jv))
            self.bullet_client.resetJointStatesMultiDof(self.my_items["robot"][sub_id],[i for i in range(len(jp))],targetValues=jp,targetVelocities=jv)
            # self.bullet_client.resetJointStateMultiDof(self.my_items["robot"][sub_id],0,targetValue=0,targetVelocity=0)
            # self.bullet_client.resetJointStateMultiDof(self.RobotUid,0,targetValue=[0],targetVelocity=[0])
        self.target_e_pose=config_data["target_e_pose"]
        self.target_e_orn=config_data["target_e_orn"]
        for key in config_data.keys():
            if(not key in["name","Task"] ):
                exec("self."+key+"=config_data"+"[\""+key+"\"]")
                # print("Load-->",key,":",config_data[key])


        self.timestep.load()
        return self.timestep

    # GAN
    def _init_GAN(self):
        ip = "localhost" if self.gan_dgx == False else "172.16.127.89"

        self.context = zmq.Context()

        self.sabs=[]
        self.sbas=[]
        ptab=5600
        ptba=5601
        for i in range(self.gan_srvs):
            q=self.context.socket(zmq.REQ)
            q.connect("tcp://{}:".format(ip)+str(ptab))
            self.sabs.append(q)
            q=self.context.socket(zmq.REQ)
            q.connect("tcp://{}:".format(ip)+str(ptba))
            self.sbas.append(q)
            ptab+=2
            ptba+=2

        # self.sab =
        # self.sab.connect("tcp://localhost:5600")
        # self.sba = self.context.socket(zmq.REQ)
        # self.sba.connect("tcp://localhost:5601")


    def gan_gen(self, img, ab):
        # img=np.ones([128,128,3])
        assert ab in ["ab", "ba"]
        if ab == "ab":
            # skt = self.sab
            skt=random.choice(self.sabs)
        else:
            skt=random.choice(self.sbas)
            # skt = self.sba
        img = img.astype(float)
        skt.send(img.tobytes())
        message = skt.recv()
        genimg = np.frombuffer(message, dtype=np.uint8).reshape([128, 128, 3])
        return genimg
    # Debug
    def profile_submit(self,cont):
        if(self.allows_log):
            if(self.logs_steps>100):
                    print('log stoped')
                    self.p.stopStateLogging(self.logId)
            else:
                self.p.submitProfileTiming(cont)
                pass
    def Apply_action_kuka(self,action):
        raise NotImplementedError("")
    # Developing
    def get_random_rgb(self):
        color = np.random.uniform(0, 1, [3])
        # color[3] = a
        return list(color)

    # Tools
    def z_get_eye_img_byrot(self,eye_pos,rot_green,rot_red,fov):
        cam_pos=eye_pos
        eye_rot = np.array(
            self.bullet_client.getMatrixFromQuaternion(self.bullet_client.getQuaternionFromEuler([rot_red,
                                                                                                  rot_green,
                                                                                                  0]))).reshape(3, 3)

        eye_rot_to_world = np.linalg.inv(eye_rot)
        forward_direct = eye_rot_to_world[2, :]
        cam_target_pos = list(np.add(cam_pos, forward_direct))
        cam_up_vec = [0, 1, 0]  # follow the green axis
        viewMat = self.bullet_client.computeViewMatrix(cam_pos, cam_target_pos, cam_up_vec)
        fov = 25  # self.args.fov
        size = self.args.img_size
        clip = [0.1, 4.0]
        projMatrix = self.bullet_client.computeProjectionMatrixFOV(fov, size[0] / size[1], clip[0], clip[1])
        imgh = size[0]
        imgw = size[1]
        color, depth, segmask = self.bullet_client.getCameraImage(
            width=imgh,
            height=imgw,
            viewMatrix=viewMat,
            projectionMatrix=projMatrix)[2:5]
        rgb = color[:, :, :3]
        np_img_arr = np.reshape(rgb, (imgh, imgw, 3)).astype(np.uint8)
        return np_img_arr
    def z_get_eye_img_bytargetpos(self,eye_pos,target_pose,fov):
        cam_pos=eye_pos
        # eye_rot = np.array(
        #     self.bullet_client.getMatrixFromQuaternion(self.bullet_client.getQuaternionFromEuler([rot_red,
        #                                                                                           rot_green,
        #                                                                                           0]))).reshape(3, 3)
        #
        # eye_rot_to_world = np.linalg.inv(eye_rot)
        # forward_direct = eye_rot_to_world[2, :]
        cam_target_pos = list(target_pose)
        cam_up_vec = [0, 1, 0]  # follow the green axis
        viewMat = self.bullet_client.computeViewMatrix(cam_pos, cam_target_pos, cam_up_vec)
        fov = fov  # self.args.fov
        key = "single_img_size" if "single_img_size" in self.args.__dir__() else "img_size"
        size = eval("self.args."+key)
        clip = [0.1, 4.0]
        projMatrix = self.bullet_client.computeProjectionMatrixFOV(fov, size[0] / size[1], clip[0], clip[1])
        imgh = size[0]
        imgw = size[1]
        color, depth, segmask = self.bullet_client.getCameraImage(
            width=imgh,
            height=imgw,
            viewMatrix=viewMat,
            projectionMatrix=projMatrix)[2:5]
        rgb = color[:, :, :3]
        np_img_arr = np.reshape(rgb, (imgh, imgw, 3)).astype(np.uint8)
        # np_img_arr=np.ones((imgh,imgw,3)).astype(np.uint8)
        return np_img_arr
    def z_fuse_img(self,img0,img1,size):
        i0_resize = cv2.resize(img0, (int(size[0] / 2), size[1]))
        i1_resize = cv2.resize(img1, (int(size[0] / 2), size[1]))
        cat_img=np.concatenate([i0_resize,i1_resize],axis=1).astype(np.uint8)
        return cat_img
    def z_parse_tname(self,name,target_task):
        if target_task =="yw_insd":
            if(not target_task in name):
               return False
            level=int(name.split("v")[1])
            return level
        return False
    def get_random_color(self):
        color = np.random.uniform(0, 1, [4])
        color[3] = 1
        return color
    def render(self, camera_index):
        # print(self.image)
        try:
            if self.images[camera_index] is None:
                pass
            else:
                img=self.images[camera_index].transpose([1,2,0])
                self.viewers[camera_index].imshow(img)
        except:
            pass
    def if_key_board_detected(self):
        keys = self.bullet_client.getKeyboardEvents()
        if len(keys) > 0:
            for k, v in keys.items():
                if v & self.bullet_client.KEY_WAS_TRIGGERED:
                    if (k == ord('r')):
                        self.reset()
                    if (k == ord('s')):
                        expid=1
                        self.save_game_into_buffer(1)
                    if (k == ord('l')):
                        expid = 9
                        self.load_game(task=self.Task,expid=expid,saveid=150)
                    if (k == ord("i")):
                        info=self._info
                        print("\n==================\nINFORMATION:")
                        for k in info.keys():
                            print(k,":",info[k])
                        # print("INFORMATION:",self)
                    if (k == ord("y")):
                        return "y"
                    if (k == ord("n")):
                        return "n"
                    if (k == ord('1')):
                        pass
                    if (k == ord('2')):
                        pass
                if v & self.bullet_client.KEY_WAS_RELEASED:
                    pass
        return None
    def z_save_img(self,level,img):
        if level in [13]:
            os.makedirs(self.img_dirpth,exist_ok=True)
            try:
                self.save_img_counter+=1
            except:
                self.save_img_counter=0


            imgpth=self.img_dirpth+"/simtask13_{}.png".format(self.save_img_counter)
            cv2.imwrite(imgpth,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
            print("saved!"+imgpth)
    def __getattr__(self, item):
        return getattr(self.taskobj,item)

class base_task(object):
    def __init__(self,env):
        self.env=env
        self.e=env
        # self.env.bullet_client=self.p
        env.flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        env.legos = []

        # Load items
        env.my_items = {}
        env.my_items["lego"] = []
        env.my_items["table"] = []
        env.my_items["robot"] = []
        env.my_items["case"] = []
        env.my_items["bolt"] = []
        env.my_items["plug"] = []
        env.my_items["case_sig"] = []
        env.my_items["insert_detect"] = []

        self.m=self.Macron(self.e)
        pass
    def init_load_items(self):
        raise NotImplementedError("")
    def reset(self):
        raise NotImplementedError("")
    def step(self):
        raise NotImplementedError("")
    def _observe(self):
        raise NotImplementedError("")
    def _reward(self):
        raise NotImplementedError("")
    def _terminal(self):
        raise NotImplementedError("")
    def _terminal_of_maxsteps(self):

        # print("self.maxsteps->",self.maxsteps)
        return 1 if self.steps>self.maxsteps else 0
    def _get_external_observe(self):
        raise NotImplementedError("")
    def _apply_action_to_sim(self,action):
        raise NotImplementedError("")
    def _task_success(self):
        raise NotImplementedError("")
    def __getattr__(self, item):
        return eval("self.env."+item)
    def change_maxsteps(self,maxsteps):
        self.maxsteps=maxsteps
        # print("env maxsteps changed to->",maxsteps)

    class Macron(object):
        def __init__(self,env):
            self.e=env
            pass
        def __getattr__(self, item):
            return eval("self.e."+item)
        def load_a_standard_table(self):
            self.my_items["table"].append(
                self.bullet_client.loadURDF("table/table.urdf", [0, -0.6, -0.6], [-0.5, -0.5, -0.5, 0.5],
                                            flags=self.flags))
#Task1: pick and reach
class yw_pick_v1(base_task):
    def __init__(self,env):
        super().__init__(env)
        self.env.robot_name='kuka'
        self.KUKA_DEFAULT_JP=[
            1.6, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
            -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
        ]
        pass
    def init_load_items(self):
        self.m.load_a_standard_table()
        self.env.maxsteps=120
        self.env.my_items["lego"].append(
            self.bullet_client.loadURDF("lego/lego.urdf", np.array([-0.1, 0.3, -0.5]), flags=self.flags))
        # load kuka
        kukaobjs = self.bullet_client.loadSDF(
            os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/kuka_with_gripper2.sdf"))
        self.env.RobotUid = kukaobjs[0]
        self.env.my_items["robot"].append(self.env.RobotUid)
    def reset(self):
        # print("reset!")
        self.env.jointPositions = self.KUKA_DEFAULT_JP[:]
        self.bullet_client.resetBasePositionAndOrientation(self.RobotUid,
                                                           [0,0,0],
                                                           [-0.707107, 0.0, 0.0, 0.707107])

        self.env.numJoints = self.bullet_client.getNumJoints(self.RobotUid)
        for jointIndex in range(self.numJoints):
            self.bullet_client.resetJointState(self.RobotUid, jointIndex, self.jointPositions[jointIndex])
        self.env.target_e_pose = ROBOT_ENDEFFECT_SAFE_POSE[:]
        self.env.target_e_orn = self.bullet_client.getQuaternionFromEuler(ROBOT_ENDEFFECT_SAFE_ORN_EU[:])
        self.bullet_client.resetBasePositionAndOrientation(self.my_items["lego"][0],
                                                           [-0.1 + random.random() * 0.1, 0.1,
                                                            -0.5 + random.random() * 0.1], [1, 1, 1, 1])
    def get_observe_dict(self):
        # observe=super(sparse1, self)._observe()
        observe=self.timestep.get_obs()
        return {"image":observe}

    def set_gripper(self, value):  # 0 open, 255 close
        # print(value)
        fingerAngle = - (value - 255) / 255 * 0.5
        # print("Fingle angle,",fingerAngle)
        # print("FingerL",fingerAngle)
        self.bullet_client.setJointMotorControl2(self.RobotUid,
                                                 8,
                                                 self.bullet_client.POSITION_CONTROL,
                                                 targetPosition=-fingerAngle,
                                                 force=FINGER_FORCE)
        self.bullet_client.setJointMotorControl2(self.RobotUid,
                                                 11,
                                                 self.bullet_client.POSITION_CONTROL,
                                                 targetPosition=fingerAngle,
                                                 force=FINGER_FORCE)
        self.bullet_client.setJointMotorControl2(self.RobotUid,
                                                 10,
                                                 self.bullet_client.POSITION_CONTROL,
                                                 targetPosition=0,
                                                 force=FINGER_TIP_FORCE)
        self.bullet_client.setJointMotorControl2(self.RobotUid,
                                                 13,
                                                 self.bullet_client.POSITION_CONTROL,
                                                 targetPosition=0,
                                                 force=FINGER_TIP_FORCE)
    def _observe(self):

        observe_dict = {}
        cartesian_posorn = self.get_end_effect_pos_orn()
        cartesian_posorn[1] = self.bullet_client.getEulerFromQuaternion(cartesian_posorn[1])
        block_posorn = list(self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0]))
        block_posorn[1] = self.bullet_client.getEulerFromQuaternion(block_posorn[1])
        observe_dict["cartesian_pos"] = np.array(cartesian_posorn[0])
        observe_dict["cartesian_orn"] = np.array(cartesian_posorn[1])
        observe_dict["block_pos"] = np.array(block_posorn[0])
        observe_dict["block_orn"] = np.array(block_posorn[1])
        return observe_dict

    def _apply_action_to_sim(self,action):
        # print("action -> sim")
        # pass
        action = action[:]
        # self.action=action
        zoom_xyz = 0.003
        zoom_rpy = 0.005
        action[0]=action[0]*zoom_xyz
        action[1]=action[1]*zoom_xyz
        action[2]=action[2]*zoom_xyz
        action[3]=action[3]*zoom_rpy
        action[4]=action[4]*zoom_rpy
        action[5]=action[5]*zoom_rpy
        action[6]=(action[6]+1)/2*255 # 0~255

        # t=self.t
        target_e_pos=self.target_e_pose
        # print(target_e_pos)
        target_e_orn_eu=list(self.bullet_client.getEulerFromQuaternion(self.target_e_orn))
        updated=0
        # print(self.bullet_client.getLinkState(self.RobotUid, 0)[5])
        for i in range(len(action)):
            a_i=action[i]
            if (i == 0):
                target_e_pos[0] = target_e_pos[0] + a_i
                updated+=1
            if (i == 1):
                target_e_pos[2] = target_e_pos[2] + a_i
                updated+=1
            if (i == 2):
                target_e_pos[1] = target_e_pos[1] + a_i
                updated+=1
            if (i == 5):  # yaw
                target_e_orn_eu[1] = target_e_orn_eu[1] + a_i
                target_e_orn_eu[1] = np.clip(target_e_orn_eu[1],-1.5,1.5)
                updated+=1
                # print("target_eu",target_e_orn_eu)
                pass
            if (i == 4):  # pitch
                target_e_orn_eu[0] = target_e_orn_eu[0] + a_i
                updated+=1
            if (i == 3): # roll
                target_e_orn_eu[2] = target_e_orn_eu[2] + a_i
                updated+=1

            if (i ==6):# finger 0~255,0 open, 255 close
                self.set_gripper(a_i)
                updated+=1


        self.env.target_e_pose=target_e_pos
        self.env.target_e_orn=self.bullet_client.getQuaternionFromEuler(target_e_orn_eu)

        jointPoses_iiwa = self.bullet_client.calculateInverseKinematics(
            self.RobotUid,
            kukaEndEffectorIndex,
            self.target_e_pose,
            self.target_e_orn,
            KUKA_LL,
            KUKA_UL,
            KUKA_JR,
            KUKA_RP,
            maxNumIterations=5
        )

        for i in range(kukaEndEffectorIndex):
            self.bullet_client.setJointMotorControl2(self.RobotUid, i, self.bullet_client.POSITION_CONTROL,
                                                     jointPoses_iiwa[i], force=5 * 240.)

        self.env.prev_pos = self.target_e_pose

        return None

    def _reward(self):
        lego_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0])[0]
        table_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["table"][0])[0]
        d_height = lego_height[1] - table_height[1]
        # print(d_height)
        if (abs(d_height) > 0.65 + 0.1):
            # print("You Win, Reward = 1")
            return 1
        return 0

    def _terminal(self):
        done=0
        if self._terminal_of_maxsteps():
            return 1
        if self._table_collied():
            done=1

        lego_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0])[0]
        table_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["table"][0])[0]
        d_height = lego_height[1] - table_height[1]
        # print(d_height)
        if (abs(d_height) > 0.65 + 0.1):
            done=1

        return done

    def _table_collied(self):
        table_collision = self.bullet_client.getContactPoints(self.RobotUid, self.my_items["table"][0])
        if (table_collision):
            return 1
        else:
            return 0
    def check_collied(self,a,b):

        tcollision = self.bullet_client.getContactPoints(
            a,
            b)
        if (tcollision):
            return 1
        else:
            return 0
    def _action_spec(self):
        maxmimum = np.array([1, 1, 1, 1, 1, 1, 1])
        minimum = np.array([-1, -1, -1, -1, -1, -1, -1])
        shape = tuple([7])
        myspec = spec(maximum=maxmimum, minimum=minimum, shape=shape)
        return myspec
class yw_pick_v1img(yw_pick_v1):
    def __init__(self,env):
        super().__init__(env)
        # self.args=fake_args()
        # self.args.img_size=[64,64]
        pass
    def _observe(self):

        def clear_dict(obser_dict):
            for key in obser_dict.keys():
                del obser_dict[key]

        observe_dict={}
        cartesian_posorn = self.get_end_effect_pos_orn()
        cartesian_posorn[1] = self.bullet_client.getEulerFromQuaternion(cartesian_posorn[1])
        block_posorn = list(self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0]))
        block_posorn[1] = self.bullet_client.getEulerFromQuaternion(block_posorn[1])
        clear_dict(observe_dict)
        observe_dict["cartesian_pos"] = np.array(cartesian_posorn[0])
        observe_dict["cartesian_orn"] = np.array(cartesian_posorn[1])
        observe_dict["block_pos"] = np.array(block_posorn[0])
        observe_dict["block_orn"] = np.array(block_posorn[1])
        # observe_dict["image"]=np.array(self.bullet_client.getCameraImage(64,64))
        # del observe_dict["image"]
        observe_dict["image"] = self._get_external_observe()
        return observe_dict
    def _get_external_observe(self):
        cameraPOS = [0, 0.2, -0.6]
        distance = 0.1
        yaw = 0
        pitch = 0
        roll = 0
        upAxisIndex = 1
        viewMat = self.bullet_client.computeViewMatrixFromYawPitchRoll(cameraPOS, distance, yaw, pitch, roll,
                                                                       upAxisIndex)
        projMatrix = [
            0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0,
            -0.02000020071864128, 0.0
        ]
        imgh = self.args.img_size[0]#64
        imgw = self.args.img_size[1]#64
        img_arr = self.bullet_client.getCameraImage(width=imgh,
                                                    height=imgw,
                                                    viewMatrix=viewMat,
                                                    projectionMatrix=projMatrix)
        rgb = img_arr[2][:, :, :3]
        np_img_arr = np.reshape(rgb, (imgh, imgw, 3))
        return np_img_arr
class yw_pick_v1img_5cm(yw_pick_v1img):
    def __init__(self,env):
        super(yw_pick_v1img_5cm, self).__init__(env)
        pass
    def _reward(self):
        lego_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0])[0]
        table_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["table"][0])[0]
        d_height = lego_height[1] - table_height[1]
        # print(d_height)
        if (abs(d_height) > 0.65 + 0.05):
            # print("You Win, Reward = 1")
            return 1
        return 0
    def _terminal(self):
        if self._table_collied():
            return 1
        lego_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0])[0]
        table_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["table"][0])[0]
        d_height = lego_height[1] - table_height[1]
        # print(d_height)
        if (abs(d_height) > 0.65 + 0.05):
            return 1
        else:
            return 0
class yw_reach_v1img(yw_pick_v1img):
    def __init__(self,env):
        super(yw_reach_v1img, self).__init__(env)
        pass
    def _reward(self):
        return self._task_success()
    def _task_success(self):
        if (self.env.Task_Success_Updated):
            return self.env.Task_Success
        # lego_pos = self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0])
        # robot_pos = self.bullet_client.getBasePositionAndOrientation(self.my_items["robot"][0])
        # table_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["table"][0])[0]
        lego_contact = self.bullet_client.getContactPoints(self.RobotUid, self.my_items["lego"][0])
        if (lego_contact):
            self.env.Task_Success = 1
        else:
            self.env.Task_Success = 0
        self.env.Task_Success_Updated = 1
        # reset in every step
        return self.env.Task_Success
    def _terminal(self):
        if self._terminal_of_maxsteps():
            return 1
        if self._table_collied():
            return 1
        if (self._task_success()):
            return self._task_success()
        # else:
        return 0

#Task2: insert
class yw_insert_v1img3cm(base_task):
    def __init__(self, env):
        super(yw_insert_v1img3cm, self).__init__(env)
        self.e.robot_name = "jaco"
        self.e.maxsteps = 40
        self.e.images = [None, None]
        try:
            self.e.viewers = [rendering.SimpleImageViewer(), rendering.SimpleImageViewer()]
        except:
            pass

        pass

    def _terminal(self):
        if self._terminal_of_maxsteps():
            return 1
        if (self.detect_coli(self.my_items["case"][0])):
            self.e.detect_coli_case += 1
            if (self.detect_coli_case > self.args.casecoli_done):
                return 1
        return 0

    def _reward(self):
        # Success Reward
        # success = self._task_success()
        pos_gripper = np.array(list(self.get_end_effect_pose(jacoEndEffectorIndex)))[:2]
        target_pos = np.array(self.args.max_reward_ee_xy)
        dist = np.linalg.norm(pos_gripper - target_pos)
        align_reward = -80 * dist + 1
        reward = align_reward  # float(success) +
        # if (success):
        #     pass

        if self._task_success():
            print("success!")
        return float(reward)

        # return self._task_success()

    def Apply_action_jaco_insert(self, action):

        action = action[:]

        zoom_xyz = 0.02 * self.args.action_default_vel_scale
        zoom_rpy = 0.02 * self.args.action_default_vel_scale

        action[0] = action[0] * zoom_xyz
        action[1] = -action[1] * zoom_xyz
        # action1 for up/down -> targetpose[1]
        # target_e_pos = self.target_e_pose
        target_e_pos = list(self.get_end_effect_pos_orn(jacoEndEffectorIndex)[0])
        target_e_orn_eu = list(self.bullet_client.getEulerFromQuaternion(self.target_e_orn))
        updated = 0

        for i in range(len(action)):
            a_i = action[i]
            if (i == 1):
                target_e_pos[0] = target_e_pos[0] + a_i
                updated += 1
            if (i == 0):
                target_e_pos[1] = target_e_pos[1] + a_i
                updated += 1

        z_default_vel = self.args.z_default_vel
        target_e_pos[2] += z_default_vel

        self.e.target_e_pose = target_e_pos
        self.e.target_e_orn = self.bullet_client.getQuaternionFromEuler(target_e_orn_eu)

        # self.target_e_pose=[
        #     self.p.readUserDebugParameter(self.debug_p0),
        #     self.p.readUserDebugParameter(self.debug_p1),
        #     self.p.readUserDebugParameter(self.debug_p2)
        # ]

        jp = self.bullet_client.calculateInverseKinematics(
            self.my_items["robot"][0],
            jacoEndEffectorIndex,
            self.target_e_pose,
            self.target_e_orn,
            JACO_ll,
            JACO_ul,
            JACO_jr,
            JACO_rp,  # TODO Change LL UL JR RP to JACO
            maxNumIterations=5
        )
        jp = list(jp)
        jp[7] = 1.3
        jp[8] = 1.3
        jp[9] = 1.3

        # print(jp)

        numJoints = self.bullet_client.getNumJoints(self.my_items["robot"][0])

        for i in range(numJoints):
            # self.bullet_client.resetJointState(self.RobotUid, i, jp[i])
            self.bullet_client.setJointMotorControl2(self.RobotUid,
                                                     i,
                                                     self.bullet_client.POSITION_CONTROL,
                                                     targetPosition=jp[i],
                                                     force=5e4)

        # print(jp)
        self.e.prev_pos = self.target_e_pose
        return None

    def _observe(self):
        observe_dict = {}
        observe_dict["image"] = self._get_external_observe()
        return observe_dict["image"]

    def _apply_action_to_sim(self, action):
        self.Apply_action_jaco_insert(action)

    def _reset_jaco_pos(self):
        def _set_jaco_ee_pose_yw_insert_v2img3cm(self):
            self.e.target_e_pose = [0.3, 0.4, -0.72]
            self.e.target_e_orn = self.bullet_client.getQuaternionFromEuler([0, math.pi, 1.13])
            jp = JACO_INSERT_JP
            jp = [[jpi] for jpi in jp]
            # jv = [[0] for i in range(len(jp))]
            self.bullet_client.resetJointStatesMultiDof(self.RobotUid, [i for i in range(len(jp))],
                                                        targetValues=jp)

        return _set_jaco_ee_pose_yw_insert_v2img3cm(self)

    def _reset_jaco_base(self):
        # reset some flags
        self.e.detect_coli_case = 0

        # reset states
        self.bullet_client.resetBasePositionAndOrientation(self.RobotUid, ROBOT_BASE_POS_3_DEFAULT[:],
                                                           ROBOT_BASE_ORN_DEFAULT[:])
        self.e.numJoints = self.bullet_client.getNumJoints(self.RobotUid)

    def reset(self):
        self._reset_jaco_base()
        self._reset_jaco_pos()

    def init_load_items(self):
        print("Init Load Items for yw_insert_v1img3cm")
        self.m.load_a_standard_table()
        self.e.maxsteps = 40
        self.e.images = [None, None]
        try:
            self.e.viewers = [rendering.SimpleImageViewer(), rendering.SimpleImageViewer()]
        except:
            pass
        case_xyz = np.array([0.25, 0.3, -0.4])
        hole1_shiftxyz = np.array([0.112, 0.17, -0.03])
        idet1_shiftxyz = np.array([0.06, 0.12, -0.01])
        idet2_shiftxyz = np.array([0.06, 0.12, -0.0])
        self.e.case_xyz = case_xyz
        self.e.idet1_shiftxyz = idet1_shiftxyz

        # load case
        case_urdf = os.path.join(self._get_model_rootdir(), "objects", "case_hole_detect.urdf")
        # case_urdf = os.path.join(self._get_model_rootdir(),"objects","case.urdf")
        self.e.my_items["case"].append(self.bullet_client.loadURDF(case_urdf, case_xyz, [0, 0, 1, 1], flags=self.flags))

        case_sig_ring_urdf = os.path.join(self._get_model_rootdir(), "objects", "case_sig_ring_front1.urdf")
        csig_euler = [3.14 / 2, 3.14, 0]
        csig_qorn = self.bullet_client.getQuaternionFromEuler(csig_euler)

        # load detect shape
        shape_thin = 0.001
        idet1_size = [0.005, 0.001]
        idet2_size = [0.001, 0.15]
        idet1_visual = self.pyb.createVisualShape(
            self.pyb.GEOM_CYLINDER,
            length=idet1_size[1],
            radius=idet1_size[0],
            rgbaColor=[1, 0.1, 0.1, 1])
        idet1_colli = self.pyb.createCollisionShape(
            self.pyb.GEOM_CYLINDER,
            height=idet1_size[1],
            radius=idet1_size[0])

        self.e.pos_insert_detect1 = case_xyz + idet1_shiftxyz
        self.e.insert_detect1 = self.bullet_client.createMultiBody(
            baseMass=0,
            basePosition=self.pos_insert_detect1,
            baseOrientation=[0, 0, 0, 1],
            baseCollisionShapeIndex=idet1_colli,
            baseVisualShapeIndex=idet1_visual
        )
        self.e.my_items["insert_detect"].append(self.insert_detect1)

        jaco_urdf = os.path.join(self._get_model_rootdir(), 'jaco_yw', "j2s7s300_bolt_2.urdf")
        jacoobjs = self.bullet_client.loadURDF(fileName=jaco_urdf,
                                               flags=self.bullet_client.URDF_MERGE_FIXED_LINKS | self.bullet_client.URDF_MAINTAIN_LINK_ORDER
                                               )
        self.e.RobotUid = jacoobjs
        self.e.my_items["robot"].append(self.RobotUid)

    def _action_spec(self):
        maxmimum = np.array([1, 1])
        minimum = np.array([-1, -1])
        shape = tuple([2])
        myspec = spec(maximum=maxmimum, minimum=minimum, shape=shape)
        return myspec

    def _task_success(self):
        if (self.e.Task_Success_Updated):
            return self.e.Task_Success
        insertdetect_1 = self.detect_coli(self.my_items["insert_detect"][0])
        success = insertdetect_1
        self.e.Task_Success = success
        self.e.Task_Success_Updated = 1
        return self.e.Task_Success

    def _get_external_observe(self):
        ee_pose, ee_orn = self.bullet_client.getLinkState(self.my_items["robot"][0], jacoEndEffectorIndex)[4:6]
        shift_xyz = self.args.camera_shitft_xyz
        shift_rpy = self.args.camera_shift_rpy
        cam_pos, cam_rot = self.bullet_client.multiplyTransforms(ee_pose, ee_orn,
                                                                 shift_xyz,
                                                                 self.bullet_client.getQuaternionFromEuler(
                                                                     shift_rpy))
        self.e.cam_pos = cam_pos
        self.e.cam_rot = cam_rot
        cam_rot = np.array(self.bullet_client.getMatrixFromQuaternion(cam_rot)).reshape(3, 3)
        viewMat = self.bullet_client.computeViewMatrix(cam_pos, cam_pos + cam_rot[:, 0], cam_rot[:, 2].tolist())

        fov = self.args.fov
        size = self.args.img_size
        clip = [0.1, 4.0]
        projMatrix = self.bullet_client.computeProjectionMatrixFOV(fov, size[0] / size[1], clip[0], clip[1])

        imgh = size[0]
        imgw = size[1]
        color, depth, segmask = self.bullet_client.getCameraImage(
            width=imgh,
            height=imgw,
            viewMatrix=viewMat,
            projectionMatrix=projMatrix)[2:5]
        rgb = color[:, :, :3]
        np_img_arr = np.reshape(rgb, (imgh, imgw, 3))
        self.e.images[0] = np_img_arr.astype(np.uint8)
        return self.e.images[0]
        # return self.taskobj._get_external_observe()

    def _loadstuff_tv(self):
        tv_visual = self.pyb.createVisualShape(
            self.pyb.GEOM_BOX,
            halfExtents=[0.2, 0.2, 0.001],
            rgbaColor=[1, 1, 1, 1])
        tv_shiftxyz = [0, 0, 0.1]
        self.e.tv = self.bullet_client.createMultiBody(
            baseMass=0,
            basePosition=self.case_xyz + tv_shiftxyz,
            baseOrientation=[0, 0, 0, 1],
            # baseCollisionShapeIndex=tv_colli,
            baseVisualShapeIndex=tv_visual
        )
        print('load textures')
        self.e.bg_imglist = os.listdir(os.path.join(self._get_textures_rootdir(), "background"))
        self.e.bg_textures = [
            self.bullet_client.loadTexture(os.path.join(self._get_textures_rootdir(), "background", img))
            for img in self.e.bg_imglist]

class yw_insert_v2img3cm(yw_insert_v1img3cm):
    def __init__(self,env):
        super(yw_insert_v2img3cm, self).__init__(env)
        pass
    def reset(self):
        super(yw_insert_v2img3cm, self)._reset_jaco_base()
        super(yw_insert_v2img3cm, self)._reset_jaco_pos()

        base_color = np.array([0.8, 0.2, 0.2, 1])
        random_color = np.random.random([4])
        random_color[3] = np.clip(random_color[3] - 0.5 + 0.9, 0, 1)
        random_color[0] = np.clip(random_color[0] - 0.5 + 0.4, 0, 1)
        random_color[1] = np.clip(random_color[1] - 0.5 - 0.2, 0, 1)
        random_color[2] = np.clip(random_color[2] - 0.5 - 0.2, 0, 1)
        random_color = random_color * 0.5
        base_color = np.clip(base_color + random_color, 0, 1)
        self.bullet_client.changeVisualShape(self.my_items["case"][0], 0, rgbaColor=base_color)

        base_new = ROBOT_BASE_POS_3_DEFAULT[:]
        pose_noise = (np.random.random([3]) - 0.5) * 0.02
        pose_noise[2] = 0
        pose_noise[0] += 0.01
        pose_noise[1] += 0.01
        base_new = np.array(base_new) + pose_noise
        self.bullet_client.resetBasePositionAndOrientation(self.RobotUid, base_new,
                                                           ROBOT_BASE_ORN_DEFAULT[:])

        self.bullet_client.changeVisualShape(self.my_items["case_sig"][0], 0, rgbaColor=[0, 0, 0, 0])
        insert_base_color = np.array([0.8, 0.1, 0.1, 0.7])
        random_color = np.random.random([4])
        random_color[3] = np.clip(random_color[3] - 0.5 + 0.9, 0, 1)
        random_color[0] = np.clip(random_color[0] - 0.5 + 0.4, 0, 1)
        random_color[1] = np.clip(random_color[1] - 0.5 - 0.2, 0, 1)
        random_color[2] = np.clip(random_color[2] - 0.5 - 0.2, 0, 1)
        random_color = random_color * 0.5
        detect_color = np.clip(random_color + insert_base_color, 0, 1)
        self.bullet_client.changeVisualShape(self.my_items["insert_detect"][0], -1, rgbaColor=detect_color)


    def _get_external_observe(self):
        ee_pose, ee_orn = self.bullet_client.getLinkState(self.my_items["robot"][0], jacoEndEffectorIndex)[4:6]
        shift_xyz = self.args.camera_shitft_xyz
        shift_rpy = self.args.camera_shift_rpy
        cam_pos, cam_rot = self.bullet_client.multiplyTransforms(ee_pose, ee_orn,
                                                                 shift_xyz,
                                                                 self.bullet_client.getQuaternionFromEuler(
                                                                     shift_rpy))

        cam_pos, cam_rot = self.bullet_client.multiplyTransforms(ee_pose, ee_orn,
                                                                 shift_xyz,
                                                                 self.bullet_client.getQuaternionFromEuler(
                                                                     shift_rpy))
        self.e.cam_pos = cam_pos
        self.e.cam_rot = cam_rot
        cam_rot = np.array(self.bullet_client.getMatrixFromQuaternion(cam_rot)).reshape(3, 3)
        viewMat = self.bullet_client.computeViewMatrix(cam_pos, cam_pos + cam_rot[:, 0], cam_rot[:, 2].tolist())

        fov = self.args.fov
        size = self.args.img_size
        clip = [0.1, 4.0]
        projMatrix = self.bullet_client.computeProjectionMatrixFOV(fov, size[0] / size[1], clip[0], clip[1])

        imgh = size[0]
        imgw = size[1]
        color, depth, segmask = self.bullet_client.getCameraImage(
            width=imgh,
            height=imgw,
            viewMatrix=viewMat,
            projectionMatrix=projMatrix,
            renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL)[2:5]
        rgb = color[:, :, :3]
        np_img_arr = np.reshape(rgb, (imgh, imgw, 3))
        self.e.images[0] = np_img_arr.astype(np.uint8)

        return self.e.images[0]
    def init_load_items(self):
        self.m.load_a_standard_table()
        case_xyz = np.array([0.25, 0.3, -0.4])
        hole1_shiftxyz = np.array([0.112, 0.17, -0.03])
        idet1_shiftxyz = np.array([0.06, 0.12, -0.005])
        # idet1_shiftxyz=np.array([0.06,0.12,-0.05])
        idet2_shiftxyz = np.array([0.06, 0.12, -0.0])

        # load case
        case_urdf = os.path.join(self._get_model_rootdir(), "objects", "case_hole_detect.urdf")
        # case_urdf = os.path.join(self._get_model_rootdir(),"objects","case.urdf")
        self.e.my_items["case"].append(self.bullet_client.loadURDF(case_urdf, case_xyz, [0, 0, 1, 1], flags=self.flags))

        case_sig_ring_urdf = os.path.join(self._get_model_rootdir(), "objects", "case_sig_ring_front1.urdf")
        csig_euler = [3.14 / 2, 3.14, 0]
        csig_qorn = self.bullet_client.getQuaternionFromEuler(csig_euler)
        self.e.my_items["case_sig"].append(
            self.e.bullet_client.loadURDF(case_sig_ring_urdf, case_xyz + hole1_shiftxyz, csig_qorn, flags=self.flags))
        # self.bullet_client.changeVisualShape(self.my_items["case_sig"][0],0,rgbaColor=[1,1,1,0.5])

        # load detect shape
        shape_thin = 0.001
        idet1_size = [0.005, 0.001]
        idet2_size = [0.001, 0.15]
        idet1_visual = self.pyb.createVisualShape(
            self.pyb.GEOM_CYLINDER,
            length=idet1_size[1],
            radius=idet1_size[0],
            rgbaColor=[0.5, 0.1, 0.1, 0])
        idet1_colli = self.pyb.createCollisionShape(
            self.pyb.GEOM_CYLINDER,
            height=idet1_size[1],
            radius=idet1_size[0])

        self.e.pos_insert_detect1 = case_xyz + idet1_shiftxyz
        self.e.insert_detect1 = self.bullet_client.createMultiBody(
            baseMass=0,
            basePosition=self.pos_insert_detect1,
            baseOrientation=[0, 0, 0, 1],
            baseCollisionShapeIndex=idet1_colli,
            baseVisualShapeIndex=idet1_visual
        )
        self.e.my_items["insert_detect"].append(self.insert_detect1)

        jaco_urdf = os.path.join(self._get_model_rootdir(), 'jaco_yw', "j2s7s300_bolt_2.urdf")
        jacoobjs = self.bullet_client.loadURDF(fileName=jaco_urdf,
                                               flags=self.bullet_client.URDF_MERGE_FIXED_LINKS | self.bullet_client.URDF_MAINTAIN_LINK_ORDER
                                               )
        self.e.RobotUid = jacoobjs
        self.e.my_items["robot"].append(self.RobotUid)
class yw_insert_v3img3cm(yw_insert_v1img3cm):
    def __init__(self,env):
        super(yw_insert_v3img3cm, self).__init__(env)
        pass
    def reset(self):
        # reset position
        super(yw_insert_v3img3cm, self)._reset_jaco_pos()

        base_new = ROBOT_BASE_POS_3_DEFAULT[:]
        pose_noise = (np.random.random([3]) - 0.5) * 0.02
        pose_noise[2] = 0
        pose_noise[0] += 0.01
        pose_noise[1] += 0.01
        base_new = np.array(base_new) + pose_noise
        self.bullet_client.resetBasePositionAndOrientation(self.RobotUid, base_new,
                                                           ROBOT_BASE_ORN_DEFAULT[:])

        # reset case front color
        self.bullet_client.changeVisualShape(self.my_items["case_sig"][0], 0, rgbaColor=[0, 0, 0, 0])

        insert_base_color = np.array([0.8, 0.1, 0.1, 0.7])
        random_color = np.random.random([4])
        random_color[3] = np.clip(random_color[3] - 0.5 + 0.9, 0, 1)
        random_color[0] = np.clip(random_color[0] - 0.5 + 0.4, 0, 1)
        random_color[1] = np.clip(random_color[1] - 0.5 - 0.2, 0, 1)
        random_color[2] = np.clip(random_color[2] - 0.5 - 0.2, 0, 1)
        random_color = random_color * 0.5
        detect_color = np.clip(random_color + insert_base_color, 0, 1)
        self.bullet_client.changeVisualShape(self.my_items["insert_detect"][0], -1, rgbaColor=detect_color)
        pass
    def _get_external_observe(self):
        # random light
        self.bullet_client.configureDebugVisualizer(lightPosition=[np.random.uniform(-3, 3),
                                                                   np.random.uniform(-3, 3),
                                                                   np.random.uniform(-5, 1)])

        # random robot color
        tip_color = (np.random.random([4]) - 0.5) * 0.1 + np.array([0.8, 0.8, 0.8, 1])
        tip_color[3] = 1
        self.bullet_client.changeVisualShape(self.my_items["robot"][0], 7, rgbaColor=tip_color)
        self.bullet_client.changeVisualShape(self.my_items["robot"][0], 8, rgbaColor=tip_color)
        self.bullet_client.changeVisualShape(self.my_items["robot"][0], 9, rgbaColor=tip_color)

        bolt_color = (np.random.random([4]) - 0.5) * 0.2 + np.array([0.2, 0.2, 0.2, 1])
        bolt_color[3] = 1
        self.bullet_client.changeVisualShape(self.my_items["robot"][0], 6, rgbaColor=bolt_color)

        case_color = np.array([0.8, 0.2, 0.2, 1]) + (np.random.random([4]) - 0.5) * 0.5
        case_color[3] = 1
        self.bullet_client.changeVisualShape(self.my_items["case"][0], 0, rgbaColor=case_color)

        # change tv text
        # imglist=os.listdir(os.path.join(self._get_textures_rootdir(),"background"))
        # texUid = self.bullet_client.loadTexture("tex256.png")
        self.bullet_client.changeVisualShape(self.tv, -1, textureUniqueId=random.choice(self.bg_textures))

        self.bullet_client.changeVisualShape(self.my_items["case"][0], -1,
                                             textureUniqueId=random.choice(self.case_textures))

        self.bullet_client.changeVisualShape(self.my_items["insert_detect"][0], -1, rgbaColor=[0, 0, 0, 0])

        ee_pose, ee_orn = self.bullet_client.getLinkState(self.my_items["robot"][0], jacoEndEffectorIndex)[4:6]
        shift_xyz = self.args.camera_shitft_xyz
        shift_rpy = self.args.camera_shift_rpy

        cam_pos, cam_rot = self.bullet_client.multiplyTransforms(ee_pose, ee_orn,
                                                                 shift_xyz,
                                                                 self.bullet_client.getQuaternionFromEuler(
                                                                     shift_rpy))
        self.e.cam_pos = cam_pos
        self.e.cam_rot = cam_rot
        cam_rot = np.array(self.bullet_client.getMatrixFromQuaternion(cam_rot)).reshape(3, 3)
        viewMat = self.bullet_client.computeViewMatrix(cam_pos, cam_pos + cam_rot[:, 0], cam_rot[:, 2].tolist())

        # fov=25 # from 40 to 60
        # size=[128,128]# h , w
        fov = self.args.fov
        size = self.args.img_size
        clip = [0.1, 4.0]
        projMatrix = self.bullet_client.computeProjectionMatrixFOV(fov, size[0] / size[1], clip[0], clip[1])

        imgh = size[0]
        imgw = size[1]
        color, depth, segmask = self.bullet_client.getCameraImage(
            width=imgh,
            height=imgw,
            viewMatrix=viewMat,
            projectionMatrix=projMatrix,
            renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL)[2:5]
        rgb = color[:, :, :3]
        np_img_arr = np.reshape(rgb, (imgh, imgw, 3))

        self.e.images[0] = np_img_arr.astype(np.uint8)
        # self.images[1]=  np_img_arr.astype(np.uint8)
        return self.e.images[0]
    def init_load_items(self):
        self.m.load_a_standard_table()
        case_xyz = np.array([0.25, 0.3, -0.4])
        self.e.case_xyz = case_xyz
        hole1_shiftxyz = np.array([0.112, 0.17, -0.03])
        idet1_shiftxyz = np.array([0.06, 0.12, -0.005])
        # idet1_shiftxyz=np.array([0.06,0.12,-0.05])
        idet2_shiftxyz = np.array([0.06, 0.12, -0.0])

        # load case
        case_urdf = os.path.join(self._get_model_rootdir(), "objects", "case_hole_detect.urdf")
        # case_urdf = os.path.join(self._get_model_rootdir(),"objects","case.urdf")
        self.e.my_items["case"].append(self.bullet_client.loadURDF(case_urdf, case_xyz, [0, 0, 1, 1], flags=self.flags))

        case_sig_ring_urdf = os.path.join(self._get_model_rootdir(), "objects", "case_sig_ring_front1.urdf")
        csig_euler = [3.14 / 2, 3.14, 0]
        csig_qorn = self.bullet_client.getQuaternionFromEuler(csig_euler)
        self.e.my_items["case_sig"].append(
            self.bullet_client.loadURDF(case_sig_ring_urdf, case_xyz + hole1_shiftxyz, csig_qorn, flags=self.flags))
        # self.bullet_client.changeVisualShape(self.my_items["case_sig"][0],0,rgbaColor=[1,1,1,0.5])

        # load detect shape
        shape_thin = 0.001
        idet1_size = [0.005, 0.001]
        idet2_size = [0.001, 0.15]
        idet1_visual = self.pyb.createVisualShape(
            self.pyb.GEOM_CYLINDER,
            length=idet1_size[1],
            radius=idet1_size[0],
            rgbaColor=[0.5, 0.1, 0.1, 1])
        idet1_colli = self.pyb.createCollisionShape(
            self.pyb.GEOM_CYLINDER,
            height=idet1_size[1],
            radius=idet1_size[0])

        self.e.pos_insert_detect1 = case_xyz + idet1_shiftxyz
        self.e.insert_detect1 = self.bullet_client.createMultiBody(
            baseMass=0,
            basePosition=self.pos_insert_detect1,
            baseOrientation=[0, 0, 0, 1],
            baseCollisionShapeIndex=idet1_colli,
            baseVisualShapeIndex=idet1_visual
        )
        self.e.my_items["insert_detect"].append(self.insert_detect1)

        jaco_urdf = os.path.join(self._get_model_rootdir(), 'jaco_yw', "j2s7s300_bolt_2.urdf")
        jacoobjs = self.bullet_client.loadURDF(fileName=jaco_urdf,
                                               flags=self.bullet_client.URDF_MERGE_FIXED_LINKS | self.bullet_client.URDF_MAINTAIN_LINK_ORDER
                                               )
        self.e.RobotUid = jacoobjs
        self.e.my_items["robot"].append(self.RobotUid)

        # TV
        self._loadstuff_tv()

        self.e.case_imglist = os.listdir(os.path.join(self._get_textures_rootdir(), "case"))
        self.e.case_textures = [
            self.bullet_client.loadTexture(os.path.join(self._get_textures_rootdir(), "case", img))
            for img in self.case_imglist]
        print('loaded textures')
class yw_insert_v4img3cm(yw_insert_v1img3cm):
    def __init__(self,env):
        super(yw_insert_v4img3cm, self).__init__(env)
        pass
    def reset(self):
        super(yw_insert_v4img3cm, self)._reset_jaco_pos()
        base_new = ROBOT_BASE_POS_3_DEFAULT[:]
        pose_noise = (np.random.random([3]) - 0.5) * 0.02
        pose_noise[2] = 0
        pose_noise[0] += 0.01
        pose_noise[1] += 0.01
        base_new = np.array(base_new) + pose_noise
        self.bullet_client.resetBasePositionAndOrientation(self.RobotUid, base_new,
                                                           ROBOT_BASE_ORN_DEFAULT[:])

        # reset case front color
        self.bullet_client.changeVisualShape(self.my_items["case_sig"][0], 0, rgbaColor=[0, 0, 0, 0])

        insert_base_color = np.array([0.8, 0.1, 0.1, 0.7])
        random_color = np.random.random([4])
        random_color[3] = np.clip(random_color[3] - 0.5 + 0.9, 0, 1)
        random_color[0] = np.clip(random_color[0] - 0.5 + 0.4, 0, 1)
        random_color[1] = np.clip(random_color[1] - 0.5 - 0.2, 0, 1)
        random_color[2] = np.clip(random_color[2] - 0.5 - 0.2, 0, 1)
        random_color = random_color * 0.5
        detect_color = np.clip(random_color + insert_base_color, 0, 1)
        self.bullet_client.changeVisualShape(self.my_items["insert_detect"][0], -1, rgbaColor=detect_color)
        # self.bullet_client.changeVisualShape(self.my_items["case"][0],1,rgbaColor=[1,1,0,1])

    def _get_external_observe(self):
        # print('get observe from v4')
        def get_random_color():
            color = np.random.uniform(0, 1, [4])
            color[3] = 1
            return color

        # case front
        case_front_color = get_random_color()
        case_front_color[3] = np.random.random(1)
        show_case_front = np.random.random() < 0.5
        if (show_case_front == False):
            case_front_color[3] = 0
        self.bullet_client.changeVisualShape(self.my_items["case_sig"][0], 0, rgbaColor=case_front_color)

        # random light
        self.bullet_client.configureDebugVisualizer(lightPosition=[np.random.uniform(0, 3),
                                                                   np.random.uniform(0, 3),
                                                                   np.random.uniform(0, -5)])

        # # finger tip color
        # self.bullet_client.changeVisualShape(
        #     self.my_items['robot'][0],10,rgbaColor=[1,0,0,1]
        # )

        # random robot color
        # tip_color=(np.random.random([4])-0.5)*0.2+np.array([0.8,0.8,0.8,1])
        # tip_color=[0,1,0,1]
        self.bullet_client.changeVisualShape(self.my_items["robot"][0], 7, rgbaColor=get_random_color())
        self.bullet_client.changeVisualShape(self.my_items["robot"][0], 8, rgbaColor=get_random_color())
        self.bullet_client.changeVisualShape(self.my_items["robot"][0], 9, rgbaColor=get_random_color())

        # bolt_color=(np.random.random([4])-0.5)*0.2+np.array([0.2,0.2,0.2,1])
        # bolt_color[3]=1
        self.bullet_client.changeVisualShape(self.my_items["robot"][0], 6, rgbaColor=get_random_color())
        # self.bullet_client.changeVisualShape(self.my_items["robot"][0],6,textureUniqueId=random.choice(self.bg_textures))

        case_color = np.array([0., 0., 0., 1]) + (np.random.random([4]))
        case_color[3] = 1
        # case_color=get_random_color()
        self.bullet_client.changeVisualShape(self.my_items["case"][0], 0, rgbaColor=case_color)

        # change tv text
        # imglist=os.listdir(os.path.join(self._get_textures_rootdir(),"background"))
        # texUid = self.bullet_client.loadTexture("tex256.png")
        self.bullet_client.changeVisualShape(self.tv, -1, textureUniqueId=random.choice(self.bg_textures))

        self.bullet_client.changeVisualShape(self.my_items["case"][0], -1,
                                             textureUniqueId=random.choice(self.case_textures))

        # change detect signal color
        random_signal_color = np.random.random([4])
        self.bullet_client.changeVisualShape(self.my_items["insert_detect"][0], -1, rgbaColor=random_signal_color)

        ee_pose, ee_orn = self.bullet_client.getLinkState(self.my_items["robot"][0], jacoEndEffectorIndex)[4:6]
        shift_xyz = self.args.camera_shitft_xyz
        shift_rpy = self.args.camera_shift_rpy
        noise_xyz = np.random.uniform(0, 0.003, [3])
        noise_rpy = np.random.uniform(0, 0.003, [3])
        shift_xyz = noise_xyz + shift_xyz
        shift_rpy = noise_rpy + shift_rpy
        cam_pos, cam_rot = self.bullet_client.multiplyTransforms(ee_pose, ee_orn,
                                                                 shift_xyz,
                                                                 self.bullet_client.getQuaternionFromEuler(
                                                                     shift_rpy))
        self.e.cam_pos = cam_pos
        self.e.cam_rot = cam_rot
        cam_rot = np.array(self.bullet_client.getMatrixFromQuaternion(cam_rot)).reshape(3, 3)
        viewMat = self.bullet_client.computeViewMatrix(cam_pos, cam_pos + cam_rot[:, 0], cam_rot[:, 2].tolist())

        # fov=25 # from 40 to 60
        # size=[128,128]# h , w
        fov = self.args.fov
        fov_noise = np.random.uniform(-1, 1, 1)
        fov = fov + fov_noise
        size = self.args.img_size
        clip = [0.1, 4.0]
        projMatrix = self.bullet_client.computeProjectionMatrixFOV(fov, size[0] / size[1], clip[0], clip[1])

        imgh = size[0]
        imgw = size[1]
        color, depth, segmask = self.bullet_client.getCameraImage(
            width=imgh,
            height=imgw,
            viewMatrix=viewMat,
            projectionMatrix=projMatrix,
            renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL)[2:5]
        rgb = color[:, :, :3]
        np_img_arr = np.reshape(rgb, (imgh, imgw, 3))

        # light noise
        np_img_arr = np_img_arr * (1 + np.random.uniform(-0.2, 0.2, 1)) + np.random.uniform(-20, 20, 1)
        np_img_arr = np.clip(np_img_arr.astype(np.uint8), 0, 255)

        # random noise
        if random.random() < 0.1:
            np_img_arr = np.random.uniform(-10, 10, np_img_arr.shape) + np_img_arr
        if random.random() < 0.03:
            np_img_arr = np.random.uniform(-20, 20, np_img_arr.shape) + np_img_arr

        self.e.images[0] = np_img_arr.astype(np.uint8)
        # self.images[1]=  np_img_arr.astype(np.uint8)
        return self.e.images[0]
    def init_load_items(self):
        self.m.load_a_standard_table()
        case_xyz = np.array([0.25, 0.3, -0.4])
        self.e.case_xyz = case_xyz
        hole1_shiftxyz = np.array([0.112, 0.17, -0.03])
        idet1_shiftxyz = np.array([0.06, 0.12, -0.005])
        # idet1_shiftxyz=np.array([0.06,0.12,-0.05])
        idet2_shiftxyz = np.array([0.06, 0.12, -0.0])

        # load case
        case_urdf = os.path.join(self._get_model_rootdir(), "objects", "case_hole_detect.urdf")
        # case_urdf = os.path.join(self._get_model_rootdir(),"objects","case.urdf")
        self.e.my_items["case"].append(self.bullet_client.loadURDF(case_urdf, case_xyz, [0, 0, 1, 1], flags=self.flags))

        case_sig_ring_urdf = os.path.join(self._get_model_rootdir(), "objects", "case_sig_ring_front1.urdf")
        csig_euler = [3.14 / 2, 3.14, 0]
        csig_qorn = self.bullet_client.getQuaternionFromEuler(csig_euler)
        self.e.my_items["case_sig"].append(
            self.bullet_client.loadURDF(case_sig_ring_urdf, case_xyz + hole1_shiftxyz, csig_qorn, flags=self.flags))
        # self.bullet_client.changeVisualShape(self.my_items["case_sig"][0],0,rgbaColor=[1,1,1,0.5])

        # load detect shape
        shape_thin = 0.001
        idet1_size = [0.005, 0.001]
        idet2_size = [0.001, 0.15]
        idet1_visual = self.pyb.createVisualShape(
            self.pyb.GEOM_CYLINDER,
            length=idet1_size[1],
            radius=idet1_size[0],
            rgbaColor=[0.5, 0.1, 0.1, 1])
        idet1_colli = self.pyb.createCollisionShape(
            self.pyb.GEOM_CYLINDER,
            height=idet1_size[1],
            radius=idet1_size[0])

        self.e.pos_insert_detect1 = case_xyz + idet1_shiftxyz
        self.e.insert_detect1 = self.bullet_client.createMultiBody(
            baseMass=0,
            basePosition=self.pos_insert_detect1,
            baseOrientation=[0, 0, 0, 1],
            baseCollisionShapeIndex=idet1_colli,
            baseVisualShapeIndex=idet1_visual
        )
        self.e.my_items["insert_detect"].append(self.insert_detect1)

        jaco_urdf = os.path.join(self._get_model_rootdir(), 'jaco_yw', "j2s7s300_bolt_3.urdf")
        jacoobjs = self.bullet_client.loadURDF(fileName=jaco_urdf,
                                               flags=self.bullet_client.URDF_MERGE_FIXED_LINKS | self.bullet_client.URDF_MAINTAIN_LINK_ORDER
                                               )
        self.e.RobotUid = jacoobjs
        self.e.my_items["robot"].append(self.RobotUid)

        # TV
        # tv_shape=
        tv_visual = self.pyb.createVisualShape(
            self.pyb.GEOM_BOX,
            halfExtents=[0.2, 0.2, 0.001],
            rgbaColor=[1, 1, 1, 1])
        # tv_colli=self.pyb.createCollisionShape(
        #     self.pyb.GEOM_BOX,
        #         halfExtents=[0.1,0.1,0.001],)

        tv_shiftxyz = [0, 0, 0.1]
        self.e.tv = self.bullet_client.createMultiBody(
            baseMass=0,
            basePosition=case_xyz + tv_shiftxyz,
            baseOrientation=[0, 0, 0, 1],
            # baseCollisionShapeIndex=tv_colli,
            baseVisualShapeIndex=tv_visual
        )

        print('load textures')
        self.e.bg_imglist = os.listdir(os.path.join(self._get_textures_rootdir(), "background"))
        self.e.bg_textures = [
            self.bullet_client.loadTexture(os.path.join(self._get_textures_rootdir(), "background", img))
            for img in self.bg_imglist]
        # img = self.bg_imglist[0]
        # self.bg_textures = [
        #     self.bullet_client.loadTexture(os.path.join(self._get_textures_rootdir(), "background", img))]

        self.e.case_imglist = os.listdir(os.path.join(self._get_textures_rootdir(), "case"))
        self.e.case_textures = [
            self.bullet_client.loadTexture(os.path.join(self._get_textures_rootdir(), "case", img))
            for img in self.case_imglist]
        # img = self.bg_imglist[0]
        # self.case_textures = [
        #     self.bullet_client.loadTexture(os.path.join(self._get_textures_rootdir(), "case", img))]

        print('loaded textures')
        # def get_random_bg_img_path(self):
        #     img=random.choice(self.bg_imglist)
        #     return os.path.join(self._get_textures_rootdir(), "background",img)
        # pass
        pass
class yw_insert_g1img(yw_insert_v1img3cm):
    def __init__(self,env):
        super(yw_insert_g1img, self).__init__(env)
        pass
    def reset(self):
        super(yw_insert_g1img, self)._reset_jaco_base()
        super(yw_insert_g1img, self)._reset_jaco_pos()
        # self._set_jaco_ee_pose_yw_insert_v2img3cm()
        base_new = ROBOT_BASE_POS_3_DEFAULT[:]
        pose_noise = (np.random.random([3]) - 0.5) * 0.02
        pose_noise[2] = 0
        pose_noise[0] += 0.01
        pose_noise[1] += 0.01
        base_new = np.array(base_new) + pose_noise
        self.bullet_client.resetBasePositionAndOrientation(self.RobotUid, base_new,
                                                           ROBOT_BASE_ORN_DEFAULT[:])

        # reset case front color
        self.bullet_client.changeVisualShape(self.my_items["case_sig"][0], 0, rgbaColor=[0, 0, 0, 0])

        insert_base_color = np.array([0.8, 0.1, 0.1, 0.7])
        random_color = np.random.random([4])
        random_color[3] = np.clip(random_color[3] - 0.5 + 0.9, 0, 1)
        random_color[0] = np.clip(random_color[0] - 0.5 + 0.4, 0, 1)
        random_color[1] = np.clip(random_color[1] - 0.5 - 0.2, 0, 1)
        random_color[2] = np.clip(random_color[2] - 0.5 - 0.2, 0, 1)
        random_color = random_color * 0.5
        detect_color = np.clip(random_color + insert_base_color, 0, 1)
        self.bullet_client.changeVisualShape(self.my_items["insert_detect"][0], -1, rgbaColor=detect_color)
        # self.bullet_client.changeVisualShape(self.my_items["case"][0],1,rgbaColor=[1,1,0,1])

    def _get_external_observe(self):
        if self.Task == "yw_insert_g1cimg":

            self.bullet_client.configureDebugVisualizer(lightPosition=[np.random.uniform(0, 3),
                                                                       np.random.uniform(0, 3),
                                                                       np.random.uniform(0, -5)])

        ee_pose, ee_orn = self.bullet_client.getLinkState(self.my_items["robot"][0], jacoEndEffectorIndex)[4:6]
        shift_xyz = self.args.camera_shitft_xyz
        shift_rpy = self.args.camera_shift_rpy

        cam_pos, cam_rot = self.bullet_client.multiplyTransforms(ee_pose, ee_orn,
                                                                 shift_xyz,
                                                                 self.bullet_client.getQuaternionFromEuler(
                                                                     shift_rpy))

        cam_pos, cam_rot = self.bullet_client.multiplyTransforms(ee_pose, ee_orn,
                                                                 shift_xyz,
                                                                 self.bullet_client.getQuaternionFromEuler(
                                                                     shift_rpy))
        self.e.cam_pos = cam_pos
        self.e.cam_rot = cam_rot
        cam_rot = np.array(self.bullet_client.getMatrixFromQuaternion(cam_rot)).reshape(3, 3)
        viewMat = self.bullet_client.computeViewMatrix(cam_pos, cam_pos + cam_rot[:, 0], cam_rot[:, 2].tolist())

        # fov=25 # from 40 to 60
        # size=[128,128]# h , w
        fov = self.args.fov
        size = self.args.img_size
        clip = [0.1, 4.0]
        projMatrix = self.bullet_client.computeProjectionMatrixFOV(fov, size[0] / size[1], clip[0], clip[1])

        imgh = size[0]
        imgw = size[1]
        color, depth, segmask = self.bullet_client.getCameraImage(
            width=imgh,
            height=imgw,
            viewMatrix=viewMat,
            projectionMatrix=projMatrix,
            renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL)[2:5]
        rgb = color[:, :, :3]
        np_img_arr = np.reshape(rgb, (imgh, imgw, 3))
        img = np_img_arr.astype(np.uint8)
        # imgs = np.expand_dims(imgs, 0)
        # ganimg=self.gan_gen(np.expand_dims(img, 0),"ba")
        if self.Task in ["yw_insert_g1img", "yw_insert_g1cimg"]:
            ganimg = self.gan_gen(img, "ba")
        elif self.Task == "yw_insert_g1bimg":
            if (random.random() < 0.4):
                ganimg = self.gan_gen(img, "ba")
                ganimg = self.gan_gen(ganimg, "ab")
            else:
                ganimg = self.gan_gen(img, "ab")
        # elif

        self.e.images[0] = ganimg
        # self.images[1]=  np_img_arr.astype(np.uint8)
        return self.e.images[0]
    def init_load_items(self):
        self.m.load_a_standard_table()
        self._init_GAN()

        case_xyz = np.array([0.25, 0.3, -0.4])
        hole1_shiftxyz = np.array([0.112, 0.17, -0.03])
        idet1_shiftxyz = np.array([0.06, 0.12, -0.005])
        idet2_shiftxyz = np.array([0.06, 0.12, -0.0])

        # load case
        case_urdf = os.path.join(self._get_model_rootdir(), "objects", "case_hole_detect.urdf")
        # case_urdf = os.path.join(self._get_model_rootdir(),"objects","case.urdf")
        self.e.my_items["case"].append(
            self.bullet_client.loadURDF(case_urdf, case_xyz, [0, 0, 1, 1], flags=self.flags))

        case_sig_ring_urdf = os.path.join(self._get_model_rootdir(), "objects", "case_sig_ring_front1.urdf")
        csig_euler = [3.14 / 2, 3.14, 0]
        csig_qorn = self.bullet_client.getQuaternionFromEuler(csig_euler)
        self.e.my_items["case_sig"].append(
            self.bullet_client.loadURDF(case_sig_ring_urdf, case_xyz + hole1_shiftxyz, csig_qorn,
                                        flags=self.flags))
        # self.bullet_client.changeVisualShape(self.my_items["case_sig"][0],0,rgbaColor=[1,1,1,0.5])

        # load detect shape
        shape_thin = 0.001
        idet1_size = [0.005, 0.001]
        idet2_size = [0.001, 0.15]
        idet1_visual = self.pyb.createVisualShape(
            self.pyb.GEOM_CYLINDER,
            length=idet1_size[1],
            radius=idet1_size[0],
            rgbaColor=[0.5, 0.1, 0.1, 0])
        idet1_colli = self.pyb.createCollisionShape(
            self.pyb.GEOM_CYLINDER,
            height=idet1_size[1],
            radius=idet1_size[0])

        self.e.pos_insert_detect1 = case_xyz + idet1_shiftxyz
        self.e.insert_detect1 = self.bullet_client.createMultiBody(
            baseMass=0,
            basePosition=self.pos_insert_detect1,
            baseOrientation=[0, 0, 0, 1],
            baseCollisionShapeIndex=idet1_colli,
            baseVisualShapeIndex=idet1_visual
        )
        self.e.my_items["insert_detect"].append(self.insert_detect1)

        jaco_urdf = os.path.join(self._get_model_rootdir(), 'jaco_yw', "j2s7s300_bolt_2.urdf")
        jacoobjs = self.bullet_client.loadURDF(fileName=jaco_urdf,
                                               flags=self.bullet_client.URDF_MERGE_FIXED_LINKS | self.bullet_client.URDF_MAINTAIN_LINK_ORDER
                                               )
        self.e.RobotUid = jacoobjs
        self.e.my_items["robot"].append(self.RobotUid)
class yw_insert_g1bimg(yw_insert_g1img):
    def __init__(self,env):
        super(yw_insert_g1bimg, self).__init__(env)
        pass
class yw_insert_g1cimg(yw_insert_g1img):
    def __init__(self,env):
        super(yw_insert_g1cimg, self).__init__(env)
        pass

class yw_insd(yw_insert_v1img3cm):
    def __init__(self,env,level):
        super(yw_insd, self).__init__(env)

        self.l=level
        self.level=level

        self.e.rtscale_1213 = 0.2  # disabled
        self.e.rtscale_15 = 0.8  # left right change
        self.e.rtscale_16 = 0.5  # rgb 2 gray
        self.e.rtscale_17 = 0.5  # random circle
        self.e.rtscale_18 = 0.5  # base noise
        self.e.rtscale_19 = 0.5  # one image miss
        self.e.rtscale_21 = 0.5  # gripper fingers


    def init_load_items(self):
        self.m.load_a_standard_table()


        level = self.l
        if level in [11, 12,14,15]:
            self._init_GAN()
        self.e.maxsteps = 40
        self.e.images = [None, None]
        try:
            self.e.viewers = [rendering.SimpleImageViewer(), rendering.SimpleImageViewer()]
        except:
            pass

        # self.p.addUserDebugParameter("test",1,2,0)
        # self.debug_p0 = self.p.addUserDebugParameter("eep0", 0.2, +0.5, 0.309)
        # self.debug_p1 = self.p.addUserDebugParameter("eep1", 0.2, 0.6, 0.406)
        # self.debug_p2 = self.p.addUserDebugParameter("eep2", -1, -0.3, -0.7)
        # self.debug_c1x = self.p.addUserDebugParameter("cx", -0.3, 0.3, 0.04)
        # self.debug_c1y = self.p.addUserDebugParameter("cy", -0.3, 0.3, 0.1)
        # self.debug_c1z = self.p.addUserDebugParameter("cz", -0.3, 0.3, 0.05)
        # self.debug_c2x = self.p.addUserDebugParameter("cx2", -0.3, 0.3, -0.04)
        # self.debug_c2y = self.p.addUserDebugParameter("cy2", -0.3, 0.3, 0.1)
        # self.debug_c2z = self.p.addUserDebugParameter("cz2", -0.3, 0.3, 0.05)
        # self.debug_fov = self.p.addUserDebugParameter("fov", 0, 50, 25)
        # self.debug_b1 = self.p.addUserDebugParameter("b1", 0, 1, 0.61)
        # self.debug_b2 = self.p.addUserDebugParameter("b2", 0, 1, 0.82)
        # self.debug_b3 = self.p.addUserDebugParameter("b3", -2, -1, -1)

        # self.dbg_b1=self.p.addUserDebugParameter("b1",0.5,0.9,0.61)
        # self.dbg_b2=self.p.addUserDebugParameter("b2",0.5,0.9,0.82)
        # self.dbg_b3=self.p.addUserDebugParameter("b3",-1.1,-0.9,-1)

        case_xyz = np.array([0.25, 0.3, -0.4])
        hole1_shiftxyz = np.array([0.112, 0.17, -0.03])
        idet1_shiftxyz = np.array([0.06, 0.12, -0.01])
        # idet1_shiftxyz=np.array([0.06,0.12,-0.05])
        idet2_shiftxyz = np.array([0.06, 0.12, -0.0])
        self.e.case_xyz = case_xyz
        self.e.idet1_shiftxyz = idet1_shiftxyz
        # load case
        case_urdf = os.path.join(self._get_model_rootdir(), "objects", "case_hole_detect.urdf")
        # case_urdf = os.path.join(self._get_model_rootdir(),"objects","case.urdf")
        self.e.my_items["case"].append(self.bullet_client.loadURDF(case_urdf, case_xyz, [0, 0, 1, 1], flags=self.flags))

        case_sig_ring_urdf = os.path.join(self._get_model_rootdir(), "objects", "case_sig_ring_front1.urdf")
        csig_euler = [3.14 / 2, 3.14, 0]
        csig_qorn = self.bullet_client.getQuaternionFromEuler(csig_euler)

        if level not in [3, 6, 7, 8, 10, 11, 12, 13, 14, 15]:  # Robust 4
            self.e.my_items["case_sig"].append(self.bullet_client.loadURDF(case_sig_ring_urdf,
                                                                         case_xyz + hole1_shiftxyz,
                                                                         csig_qorn,
                                                                         flags=self.flags))
        # Robust 8
        if level in [6, 7, 8, 9, 10,13]:
            self._loadstuff_tv()

        # load detect shape
        shape_thin = 0.001
        idet1_size = [0.005, 0.001]
        idet2_size = [0.001, 0.15]
        idet1_visual = self.pyb.createVisualShape(
            self.pyb.GEOM_CYLINDER,
            length=idet1_size[1],
            radius=idet1_size[0],
            rgbaColor=[1, 0.1, 0.1, 1])
        idet1_colli = self.pyb.createCollisionShape(
            self.pyb.GEOM_CYLINDER,
            height=idet1_size[1],
            radius=idet1_size[0])

        # self.bullet_client.changeVisualShape(
        #     idet1_visual,0,rgbaColor=[1,0,0,1]
        # )
        self.e.pos_insert_detect1 = case_xyz + idet1_shiftxyz
        self.e.insert_detect1 = self.bullet_client.createMultiBody(
            baseMass=0,
            basePosition=self.pos_insert_detect1,
            baseOrientation=[0, 0, 0, 1],
            baseCollisionShapeIndex=idet1_colli,
            baseVisualShapeIndex=idet1_visual
        )
        self.e.my_items["insert_detect"].append(self.insert_detect1)

        jaco_urdf = os.path.join(self._get_model_rootdir(), 'jaco_yw', "j2s7s300_bolt_2.urdf")
        jacoobjs = self.bullet_client.loadURDF(fileName=jaco_urdf,
                                               flags=self.bullet_client.URDF_MERGE_FIXED_LINKS | self.bullet_client.URDF_MAINTAIN_LINK_ORDER
                                               )
        self.e.RobotUid = jacoobjs
        self.e.my_items["robot"].append(self.RobotUid)

    def _get_external_observe(self):
        # return np.random.uniform(1, 255, [128, 128, 3])
        level = self.l
        # level = self.z_parse_tname(self.Task, target_task="yw_insd")
        s = level / 10
        # print("level=",level)
        ee_pose, ee_orn = self.bullet_client.getLinkState(self.my_items["robot"][0], jacoEndEffectorIndex)[4:6]

        # Robust 1 2
        if (level in [4, 5, 6, 7, 8, 9, 10]):
            mxyz = 0.01
            mrpy = 0.01
            s = level / 10
            noise_0 = np.random.uniform(-s * mrpy, s * mrpy, [3])
            noise_0[2] = 0  # Robust 1
            noise_1 = np.random.uniform(-s * mrpy, s * mrpy, [3])
            noise_1[2] = 0  # robust 1
            noise_xyz_0 = np.random.uniform(-s * mxyz, s * mxyz, [3])  # Robust 2
            noise_xyz_1 = np.random.uniform(-s * mxyz, s * mxyz, [3])  # Robust 2
        elif level in [11, 12, 13]:
            mxyz = 0.01
            mrpy = 0.01
            s = 0.5
            noise_0 = np.random.uniform(-s * mrpy, s * mrpy, [3])
            noise_0[2] = 0  # Robust 1
            noise_1 = np.random.uniform(-s * mrpy, s * mrpy, [3])
            noise_1[2] = 0  # robust 1
            noise_xyz_0 = np.random.uniform(-s * mxyz, s * mxyz, [3])  # Robust 2
            noise_xyz_1 = np.random.uniform(-s * mxyz, s * mxyz, [3])  # Robust 2
        elif level in [14,15]:
            noise_0 = self.cam_target_noise_0_from_reset
            noise_1 = self.cam_target_noise_1_from_reset
            noise_xyz_0 = self.cam_pos_noise_1_from_reset#np.array([0, 0, 0])
            noise_xyz_1 = self.cam_pos_noise_1_from_reset#np.array([0, 0, 0])
        elif level in [1, 2, 3]:
            noise_0 = self.noise_0_from_reset
            noise_1 = self.noise_1_from_reset
            noise_xyz_0 = np.array([0, 0, 0])
            noise_xyz_1 = np.array([0, 0, 0])

        # Robust 3
        if (level in [4, 5, 6, 7, 8, 9, 10]):
            mfov = (level - 3) / 2
            fov_noise_1 = np.random.uniform(-mfov, mfov, 1)[0]
            fov_noise_2 = np.random.uniform(-mfov, mfov, 1)[0]
        elif level in [11, 12, 13]:
            mfov = 0.5
            fov_noise_1 = np.random.uniform(-mfov, mfov, 1)[0]
            fov_noise_2 = np.random.uniform(-mfov, mfov, 1)[0]
        else:
            fov_noise_1 = 0
            fov_noise_2 = 0

        # Robust 5
        if level in [1, 2, 4, 5, 9]:
            if level in [1, 6]:
                front_color_alpha = 1
            else:
                front_color_alpha = 0
            front_color_rgb = self.get_random_rgb()
            front_color = front_color_rgb + [front_color_alpha]
            self.bullet_client.changeVisualShape(self.my_items["case_sig"][0], 0, rgbaColor=front_color)

        # Robust 7
        if level in [4, 5, 6, 7, 8, 9, 10]:
            if level in [4, 5, 6]:
                # random robot color
                tip_color = (np.random.random([4]) - 0.5) * 0.1 + np.array([0.8, 0.8, 0.8, 1])
                tip_color[3] = 1
                self.bullet_client.changeVisualShape(self.my_items["robot"][0], 7, rgbaColor=tip_color)
                self.bullet_client.changeVisualShape(self.my_items["robot"][0], 8, rgbaColor=tip_color)
                self.bullet_client.changeVisualShape(self.my_items["robot"][0], 9, rgbaColor=tip_color)

                bolt_color = (np.random.random([4]) - 0.5) * 0.2 + np.array([0.2, 0.2, 0.2, 1])
                bolt_color[3] = 1
                self.bullet_client.changeVisualShape(self.my_items["robot"][0], 6, rgbaColor=bolt_color)

                case_color = np.array([0.8, 0.2, 0.2, 1]) + (np.random.random([4]) - 0.5) * 0.5
                case_color[3] = 1
                self.bullet_client.changeVisualShape(self.my_items["case"][0], 0, rgbaColor=case_color)
            elif level in [7, 8, 9, 10]:
                self.bullet_client.changeVisualShape(self.my_items["robot"][0], 7,
                                                     rgbaColor=self.get_random_color())
                self.bullet_client.changeVisualShape(self.my_items["robot"][0], 8,
                                                     rgbaColor=self.get_random_color())
                self.bullet_client.changeVisualShape(self.my_items["robot"][0], 9,
                                                     rgbaColor=self.get_random_color())
                self.bullet_client.changeVisualShape(self.my_items["robot"][0], 6,
                                                     rgbaColor=self.get_random_color())
                self.bullet_client.changeVisualShape(self.my_items["case"][0], 0, rgbaColor=self.get_random_color())

        # Robust 6
        if level in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            self.bullet_client.configureDebugVisualizer(lightPosition=[np.random.uniform(0, 3),
                                                                       np.random.uniform(0, 3),
                                                                       np.random.uniform(0, -5)])

        # Robust 9
        if level in [6, 7, 8, 9, 10, 13]:
            self.bullet_client.changeVisualShape(self.tv, -1, textureUniqueId=random.choice(self.bg_textures))

        # Robust 11
        self.rt11_(level)

        # Robust 20
        self.rt20_(level)

        #
        self.rt21_(level)

        # cam 1
        bolt_tap_pos = self.get_bolt_pose()

        # debug_xyz1=[
        #     self.p.readUserDebugParameter(self.debug_c1x),
        #     self.p.readUserDebugParameter(self.debug_c1y),
        #     self.p.readUserDebugParameter(self.debug_c1z)
        # ]
        #
        # debug_xyz2=[
        #     self.p.readUserDebugParameter(self.debug_c2x),
        #     self.p.readUserDebugParameter(self.debug_c2y),
        #     self.p.readUserDebugParameter(self.debug_c2z)
        # ]
        # debug_fov=self.p.readUserDebugParameter(self.debug_fov)
        # noise_0=0
        # noise_1=0
        # noise_xyz_0=0
        # noise_xyz_1=0
        # fov_noise_1=0
        # fov_noise_2=0
        # camera_shitft_xyz=debug_xyz1
        # camera_shitft_xyz2=debug_xyz2

        cam_pos1 = list(np.add(ee_pose, self.args.camera_shitft_xyz) + noise_xyz_0)
        # cam_pos1 = list(np.add(ee_pose, debug_xyz1) + noise_xyz_0)
        target_point = list(np.add(bolt_tap_pos, noise_0))
        img0 = self.z_get_eye_img_bytargetpos(eye_pos=cam_pos1, target_pose=target_point,
                                              fov=self.args.fov + fov_noise_1)
        # fov=debug_fov + fov_noise_1)

        cam_pos2 = list(np.add(ee_pose, self.args.camera_shitft_xyz2) + noise_xyz_1)
        # cam_pos2 = list(np.add(ee_pose,debug_xyz2) + noise_xyz_1)
        target_point = list(np.add(bolt_tap_pos, noise_1))
        img1 = self.z_get_eye_img_bytargetpos(eye_pos=cam_pos2, target_pose=target_point,
                                              fov=self.args.fov2 + fov_noise_2)
        # fov=debug_fov + fov_noise_2)

        # Robust 12 Robust 13
        img0 = self.rt1213_(level, img0)
        img1 = self.rt1213_(level, img1)

        # # Robust 15
        # img0,img1=self.rt15_(level,img0,img1)
        # # Robust 16
        # img0,img1=self.rt16_(level,img0,img1)
        #
        # # Robust 17
        # img0=self.rt17_(level,img0)
        # img1=self.rt17_(level,img1)
        #
        # # Robust 19
        # img0=self.rt19_(level,img0,1)
        # img1=self.rt19_(level,img1,2)

        # Robust 22
        img0 = self.rt22_(level, img0)
        img1 = self.rt22_(level, img1)

        # Robust 23
        img0 = self.rt23_(level, img0)
        img1 = self.rt23_(level, img1)

        self.z_save_img(level, img0)
        self.z_save_img(level, img1)

        cat_img = self.z_fuse_img(img0, img1, self.args.img_size)
        cat_img = np.transpose(cat_img,[2,0,1])
        self.images[0] = cat_img
        return self.images[0]
    def reset(self):
        super(yw_insd, self)._reset_jaco_base()
        # super(yw_insd, self)._reset_jaco_pos()

        level=self.l
        # level = self.z_parse_tname(self.Task, target_task="yw_insd")
        # if level is None:
        #     self._set_jaco_ee_pose_yw_insert_v2img3cm()
        # else:
        if (level in [1, 2, 3]):
            max_translation = 0.01  # meter
            noise_0 = np.random.uniform(-level / 10 * max_translation, level / 10 * max_translation, [3])
            # noise_0[2]=0
            noise_1 = np.random.uniform(-level / 10 * max_translation, level / 10 * max_translation, [3])
            # noise_1[2]=0
            self.noise_0_from_reset = noise_0
            self.noise_1_from_reset = noise_1
        elif level in [14,15]:
            max_translation = 0.01  # meter
            n1 = np.random.uniform(-max_translation, max_translation, [3])
            n2 = np.random.uniform(-max_translation, max_translation, [3])
            n3 = np.random.uniform(-max_translation, max_translation, [3])
            n4 = np.random.uniform(-max_translation, max_translation, [3])
            # noise_1[2]=0
            self.cam_target_noise_0_from_reset = n1
            self.cam_target_noise_1_from_reset = n2
            self.cam_pos_noise_1_from_reset = n3
            self.cam_pos_noise_1_from_reset = n4

        # Robust 18
        self.rt18_(level)

    # Robust Tools Box (call to make change to env)
    def rt1_(self):
        pass

    def rt11_(self, l):
        assert len(self.my_items["insert_detect"]) > 0
        # insert detect color change
        # if l ==1:
        c = [0, 0, 0, 0]
        self.bullet_client.changeVisualShape(self.my_items["insert_detect"][0], -1, rgbaColor=c)

    def rt1213_(self, l, np_img_arr):
        if l in [1, 2, 3, 4]:
            return np_img_arr
        elif l in [5, 6, 7, 8, 9, 10]:
            s = (l - 4) / 6 * self.rtscale_1213

            # # light noise
            # np_img_arr = np_img_arr * (1 + np.random.uniform(-0.2, 0, 1)*s) + np.random.uniform(-20, 20, 1)*s
            # # random noise
            # if random.random() < 0.1:
            #     np_img_arr = np.random.uniform(-10*s, 0, np_img_arr.shape) + np_img_arr
            # if random.random() < 0.03:
            #     np_img_arr = np.random.uniform(-20*s, 0, np_img_arr.shape) + np_img_arr
            # np_img_arr = np.clip(np_img_arr.astype(np.uint8), 1, 255)

            return np_img_arr.astype(np.uint8)
        else:
            return np_img_arr

    def rt15_(self, l, img0, img1):
        # s=(l-3)/7
        if l in [6, 8, 10] and random.random() < 0.5 * self.rtscale_15:
            return img1, img0
        if l in [4, 5, 7] and self.hash_update_per_epis < 0.5 * self.rtscale_15:
            return img1, img0
        return img0, img1

    def rt16_(self, l, img0, img1):
        # print('l=',l)
        s = (l - 3) / 7 * self.rtscale_16
        # return img0.om
        if l in [6, 8, 10] and random.random() < 0.3 + 0.2 * s:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        if l in [6, 8, 10] and random.random() < 0.3 + 0.2 * s:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if l in [4, 5, 7]:
            np.random.seed(int(self.hash_update_per_epis * 10))
            rd1 = np.random.random()
            np.random.seed(int(self.hash_update_per_epis * 5))
            rd2 = np.random.random()
            np.random.seed(int(time.time() * 1000 % 2 ** 32))
            # print("rand2",np.random.random(),np.random.random())
            if rd1 < 0.3 + 0.2 * s:
                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            if rd2 < 0.3 + 0.2 * s:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # imgs=[img0,img1]
        if len(img0.shape) == 2:
            img0 = np.tile(img0, [3, 1, 1]).transpose([1, 2, 0])
        if len(img1.shape) == 2:
            img1 = np.tile(img1, [3, 1, 1]).transpose([1, 2, 0])
        return img0, img1

    def rt17_(self, level, img):
        def gen_a_random_circle(image, maxr):
            center_coordinates = tuple(np.random.uniform(1, image.shape[0] - 1, [2]).astype(np.int))
            radius = int(random.random() * maxr + 1)
            color = tuple(np.random.uniform(0, 255, [3]).astype(np.int))
            color = (int(color[0]), int(color[1]), int(color[2]))
            thickness = -1
            # try:
            # image = cv2.circle(cv2.UMat(image), center_coordinates, radius, color, thickness)
            # try:
            image = cv2.circle(cv2.UMat(image), center_coordinates, radius, color, thickness).get()
            # except:
            #     pass
            # except:
            #     pass
            return image

        if level in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            s = (level - 4) / 6 * self.rtscale_17
            # loop=0
            while (random.random() < 0.4 + 0.1 * s and s > 0):
                # loop+=1
                # print(loop)
                img = gen_a_random_circle(img, int(s * 20 + 1))
        return img
        # pass

    def rt18_(self, level):
        if level in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            s = level / 10 * 0.6 + 0.4
        elif level in [11, 12, 13]:
            s = 1 #0.5
        elif level in [14,15]:
            s = 1
        else:
            self._reset_jaco_pos()
            return None
        s = s * self.rtscale_18

        # reset position
        self._reset_jaco_pos()
        base_new = [0.61, 0.811, -1]
        # base_new = [self.p.readUserDebugParameter(self.dbg_b1),
        #             self.p.readUserDebugParameter(self.dbg_b2),
        #             self.p.readUserDebugParameter(self.dbg_b3)]
        # pose_noise = (np.random.random([3]) - 0.5) * 0.02
        pose_noise = np.random.uniform(-1,1,3)*0.02
        pose_noise[2] = 0
        base_new = np.array(base_new) + pose_noise * s #+[0,0,0.1]
        # print(base_new)
        self.bullet_client.resetBasePositionAndOrientation(self.RobotUid, base_new,
                                                           ROBOT_BASE_ORN_DEFAULT[:])

    # def _terminal(self):
    #     return 1

    def rt19_(self, l, img, seed):
        # Left Right One Image Black
        if l in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            s = (l - 3) / 7
            # if l in [6, 8, 10] and random.random() < 0.5: # by frame
            #     img = np.random.uniform(1, 254, img.shape)
            if l in [5, 8, 10]:  # by epis
                # np.random.seed(int(self.hash_update_per_epis * 10)+seed)
                rd1 = np.random.random()

                # np.random.seed(int(time.time() * 1000 % 2 ** 32))
                # print("rand2",np.random.random(),np.random.random())
                if rd1 < 0.05 * s * self.rtscale_19:
                    img = np.random.uniform(1, 254, img.shape)
        return img

    def rt20_(self, l):
        if l in [6, 7, 8, 10]:
            # self.bullet_client.changeVisualShape(self.my_items["robot"][0], 6, rgbaColor=bolt_color)
            self.bullet_client.changeVisualShape(self.my_items["robot"][0], 6,
                                                 textureUniqueId=random.choice(self.bg_textures))

    def rt21_(self, l):
        if l in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            d = 1.3
            s = (l - 3) / 7 * self.rtscale_21
            maxd = 0.5
        elif l in [11, 12, 13]:
            d = 1.3
            s = 0.2
            maxd = 0.5
        else:
            return None
        if s > 0:
            m = maxd * np.random.uniform(-1, 1, [3]) * s
            jp = list(m + d)
            jp = [[jpi] for jpi in jp]
            self.bullet_client.resetJointStatesMultiDof(self.RobotUid, [i + 7 for i in range(len(jp))],
                                                        targetValues=jp)

    def rt22_(self, l, img):
        #   gan BA
        if l in [11,14]:
            # if self.Task in ["yw_insert_g1img", "yw_insert_g1cimg"]:
            img = self.gan_gen(img, "ba")
        return img

    def rt23_(self, l, img):
        # GAN ba -> ab
        if l in [12]:
            # elif self.Task == "yw_insert_g1bimg":
            if (random.random() < 0.4):
                img = self.gan_gen(img, "ba")
                img = self.gan_gen(img, "ab")
            else:
                img = self.gan_gen(img, "ab")
        if l in [15]:
            img = self.gan_gen(img, "ba")
            img = self.gan_gen(img, "ab")
        return img
        # elif

#Task2.1: FAST Insert
class yw_insf(yw_insd):
    def __init__(self,env,level):
        super(yw_insf, self).__init__(env,level)
        self.reset_times=0
    def _reset_jaco_pos(self):
        jp=self.JacoCart_reset_jp
        self.bullet_client.resetJointStatesMultiDof(self.RobotUid, [i for i in range(len(jp))], targetValues=jp)
    def _reset_jaco_base(self):
        # reset states
        self.bullet_client.\
            resetBasePositionAndOrientation(
            self.RobotUid,
            # self.JacoCart_BasePos,
            self.JacoCart_BaseOrn)
        self.e.numJoints = self.bullet_client.getNumJoints(self.RobotUid)
    def get_end_effect_pos_orn(self):
        ee_index=self.JacoCartEEIndex
        state = self.bullet_client.getLinkState(self.RobotUid, ee_index)
        # print("state", state
        return list(state[4:6])  #
    def get_end_effect_pose(self):
        ee_index=self.JacoCartEEIndex
        state = self.bullet_client.getLinkState(self.RobotUid, ee_index)
        return list(state[4])
    def get_end_effect_pos_orn_calibrated(self):
        ee_pose_emerged,ee_orn_emerged=self.get_end_effect_pos_orn()
        ee_pose=list(ee_pose_emerged)
        ee_pose[2]=ee_pose[2]+0.2
        ee_orn=ee_orn_emerged
        return ee_pose,ee_orn
    def get_end_effect_pose_calibrated(self):
        eepose,_=self.get_end_effect_pos_orn_calibrated()
        return eepose
    def get_bolt_pose(self):

        ee_pose=self.get_end_effect_pose_calibrated()
        pos=list(ee_pose)
        pos[2] += 0.4
        pos[1] += 0.01
        pos[0] -= 0.01

        # pos[1] += 0.1
        #
        #
        # # Only For Kinova Robut with Bolt
        # state = self.bullet_client.getLinkState(self.RobotUid, self.BoltBodyIndex)
        # pos=list(state[4])
        # pos[2]+=0.3


        return  pos# state[4] is the worldLinkFramePosition

    def init_load_items(self):
        self.JacoCartEEIndex=2 #?
        self.BoltBodyIndex=4 #?
        self.JacoCart_BasePos=[0.217,0.37,-1]
        self.JacoCart_BaseOrn=self.p.getQuaternionFromEuler([-1.57,1.57,0])
        self.e.target_e_pose=[0.2,0.36,-1] # ?
        self.e.target_e_orn=self.p.getQuaternionFromEuler([0,0,0]) # ?
        self.JacoCart_reset_jp=[[0], [0], [0], [1.3], [1.3], [1.3]]


        self.m.load_a_standard_table()


        level = self.l
        if level in [11, 12,14,15]:
            self._init_GAN()
        self.e.maxsteps = 40
        self.e.images = [None, None]
        try:
            self.e.viewers = [rendering.SimpleImageViewer(), rendering.SimpleImageViewer()]
        except:
            pass

        # self.p.addUserDebugParameter("test",1,2,0)
        # self.debug_p0 = self.p.addUserDebugParameter("eep0", 0.2, +0.5, 0.309)
        # self.debug_p1 = self.p.addUserDebugParameter("eep1", 0.2, 0.6, 0.406)
        # self.debug_p2 = self.p.addUserDebugParameter("eep2", -1, -0.3, -0.7)
        # self.debug_c1x = self.p.addUserDebugParameter("cx", -0.3, 0.3, 0.04)
        # self.debug_c1y = self.p.addUserDebugParameter("cy", -0.3, 0.3, 0.1)
        # self.debug_c1z = self.p.addUserDebugParameter("cz", -0.3, 0.3, 0.05)
        # self.debug_c2x = self.p.addUserDebugParameter("cx2", -0.3, 0.3, -0.04)
        # self.debug_c2y = self.p.addUserDebugParameter("cy2", -0.3, 0.3, 0.1)
        # self.debug_c2z = self.p.addUserDebugParameter("cz2", -0.3, 0.3, 0.05)
        # self.debug_fov = self.p.addUserDebugParameter("fov", 0, 50, 25)
        # self.debug_b1 = self.p.addUserDebugParameter("b1", 0, 1, 0.61)
        # self.debug_b2 = self.p.addUserDebugParameter("b2", 0, 1, 0.82)
        # self.debug_b3 = self.p.addUserDebugParameter("b3", -2, -1, -1)

        # self.dbg_b1=self.p.addUserDebugParameter("b1",0.5,0.9,0.61)
        # self.dbg_b2=self.p.addUserDebugParameter("b2",0.5,0.9,0.82)
        # self.dbg_b3=self.p.addUserDebugParameter("b3",-1.1,-0.9,-1)

        # self.dbg_jp1=self.p.addUserDebugParameter("jp1",-1,1,0)
        # self.dbg_jp2=self.p.addUserDebugParameter("jp2",-1,1,0)
        # self.dbg_jp3=self.p.addUserDebugParameter("jp3",-1,1,0)
        # self.dbg_jp4=self.p.addUserDebugParameter("jp4",-1,1,0)
        # self.dbg_jp5=self.p.addUserDebugParameter("jp5",-1,1,0)
        # self.dbg_jp6=self.p.addUserDebugParameter("jp6",-1,1,0)
        # self.t1=self.p.addUserDebugParameter("t1",-1,1,0.2)
        # self.t2=self.p.addUserDebugParameter("t2",-1,1,0.36)
        # self.t3=self.p.addUserDebugParameter("t3",-1,1,-1)
        # self.t4=self.p.addUserDebugParameter("t4",-3,1,-1.57)
        # self.t5=self.p.addUserDebugParameter("t5",-1,5,1.57)
        # self.t6=self.p.addUserDebugParameter("t6",-1,1,0)

        case_xyz = np.array([0.25, 0.3, -0.4])
        hole1_shiftxyz = np.array([0.112, 0.17, -0.03])
        idet1_shiftxyz = np.array([0.06, 0.12, -0.01])
        # idet1_shiftxyz=np.array([0.06,0.12,-0.05])
        idet2_shiftxyz = np.array([0.06, 0.12, -0.0])
        self.e.case_xyz = case_xyz
        self.e.idet1_shiftxyz = idet1_shiftxyz
        # load case
        case_urdf = os.path.join(self._get_model_rootdir(), "objects", "case_hole_detect.urdf")
        # case_urdf = os.path.join(self._get_model_rootdir(),"objects","case.urdf")
        self.e.my_items["case"].append(self.bullet_client.loadURDF(case_urdf, case_xyz, [0, 0, 1, 1], flags=self.flags))

        case_sig_ring_urdf = os.path.join(self._get_model_rootdir(), "objects", "case_sig_ring_front1.urdf")
        csig_euler = [3.14 / 2, 3.14, 0]
        csig_qorn = self.bullet_client.getQuaternionFromEuler(csig_euler)

        if level not in [3, 6, 7, 8, 10, 11, 12, 13, 14, 15]:  # Robust 4
            self.e.my_items["case_sig"].append(self.bullet_client.loadURDF(case_sig_ring_urdf,
                                                                         case_xyz + hole1_shiftxyz,
                                                                         csig_qorn,
                                                                         flags=self.flags))
        # Robust 8
        if level in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            self._loadstuff_tv()

        # load detect shape
        shape_thin = 0.001
        idet1_size = [0.005, 0.001]
        idet2_size = [0.001, 0.15]
        idet1_visual = self.pyb.createVisualShape(
            self.pyb.GEOM_CYLINDER,
            length=idet1_size[1],
            radius=idet1_size[0],
            rgbaColor=[1, 0.1, 0.1, 1])
        idet1_colli = self.pyb.createCollisionShape(
            self.pyb.GEOM_CYLINDER,
            height=idet1_size[1],
            radius=idet1_size[0])

        # self.bullet_client.changeVisualShape(
        #     idet1_visual,0,rgbaColor=[1,0,0,1]
        # )
        self.e.pos_insert_detect1 = case_xyz + idet1_shiftxyz
        self.e.insert_detect1 = self.bullet_client.createMultiBody(
            baseMass=0,
            basePosition=self.pos_insert_detect1,
            baseOrientation=[0, 0, 0, 1],
            baseCollisionShapeIndex=idet1_colli,
            baseVisualShapeIndex=idet1_visual
        )
        self.e.my_items["insert_detect"].append(self.insert_detect1)

        # jaco_urdf = os.path.join(self._get_model_rootdir(), 'jaco_yw', "j2s7s300_bolt_2.urdf")
        jaco_urdf = os.path.join(self._get_model_rootdir(), 'jaco_description','urdf', "jaco_cartesian.urdf")
        jacoobjs = self.bullet_client.loadURDF(jaco_urdf,
                                               self.JacoCart_BasePos,#startPos,
                                               self.JacoCart_BaseOrn,
                                               flags=self.bullet_client.URDF_MAINTAIN_LINK_ORDER | self.bullet_client.URDF_MERGE_FIXED_LINKS
                                               ,useFixedBase=True)#,#startOrientation)

        self.bullet_client.changeVisualShape(jacoobjs,-1,rgbaColor=[0,0,0,0])
        self.bullet_client.changeVisualShape(jacoobjs,0,rgbaColor=[0,0,0,0])
        self.bullet_client.changeVisualShape(jacoobjs,1,rgbaColor=[0,0,0,0])
        self.bullet_client.changeVisualShape(jacoobjs,2,rgbaColor=[0,0,0,1])
        # self.bullet_client.changeVisualShape(jacoobjs,3,rgbaColor=[0,0,0,0])
        # self.bullet_client.changeVisualShape(jacoobjs,2,rgbaColor=[0,0,0,0])
        # self.bullet_client.changeVisualShape(jacoobjs,3,rgbaColor=[0,0,0,0])
        # self.bullet_client.changeVisualShape(jacoobjs,4,rgbaColor=[0,0,0,0])
        # (self.my_items["case"][0], 0, rgbaColor=base_color)

        # jacoobjs = self.bu
        self.e.RobotUid = jacoobjs
        self.e.my_items["robot"].append(self.RobotUid)

    def _reward(self):
        # Success Reward
       # success = self._task_success()
       #  print()
        # ee_pose=self.get_end_effect_pose()
        # print("best eepos",ee_pose)
        ee_pose = np.array(list(self.get_end_effect_pose()))[:2]
        #[0.3270845115184784, 0.40046796202659607 best ee_pose
        target_pos = np.array([0.3270845, 0.4004679])
        dist = np.linalg.norm(ee_pose - target_pos)
        align_reward = np.clip(-80 * dist + 1,-1,1)
        reward = align_reward #float(success) +
        # if (success):
        #     pass

        if self._task_success():
            print("success!")
        return float(reward)

    def _get_external_observe(self):

        level = self.l
        s = level / 10
        # ee_pose, ee_orn = self.bullet_client.getLinkState(self.my_items["robot"][0], self.JacoCartEEIndex)[4:6]
        ee_pose, ee_orn = self.get_end_effect_pos_orn_calibrated()

        # Robust 1 2
        if (level in [4, 5, 6, 7, 8, 9, 10]):
            mxyz = 0.01
            mrpy = 0.01
            s = level / 10
            noise_0 = np.random.uniform(-s * mrpy, s * mrpy, [3])
            noise_0[2] = 0  # Robust 1
            noise_1 = np.random.uniform(-s * mrpy, s * mrpy, [3])
            noise_1[2] = 0  # robust 1
            noise_xyz_0 = np.random.uniform(-s * mxyz, s * mxyz, [3])  # Robust 2
            noise_xyz_1 = np.random.uniform(-s * mxyz, s * mxyz, [3])  # Robust 2
        elif level in [11, 12, 13]:
            mxyz = 0.01
            mrpy = 0.01
            s = 0.5
            noise_0 = np.random.uniform(-s * mrpy, s * mrpy, [3])
            noise_0[2] = 0  # Robust 1
            noise_1 = np.random.uniform(-s * mrpy, s * mrpy, [3])
            noise_1[2] = 0  # robust 1
            noise_xyz_0 = np.random.uniform(-s * mxyz, s * mxyz, [3])  # Robust 2
            noise_xyz_1 = np.random.uniform(-s * mxyz, s * mxyz, [3])  # Robust 2
        elif level in [14,15]:
            noise_0 = self.cam_target_noise_0_from_reset
            noise_1 = self.cam_target_noise_1_from_reset
            noise_xyz_0 = self.cam_pos_noise_1_from_reset#np.array([0, 0, 0])
            noise_xyz_1 = self.cam_pos_noise_1_from_reset#np.array([0, 0, 0])
        elif level in [1, 2, 3]:
            noise_0 = self.noise_0_from_reset
            noise_1 = self.noise_1_from_reset
            noise_xyz_0 = np.array([0, 0, 0])
            noise_xyz_1 = np.array([0, 0, 0])

        # Robust 3
        if (level in [4, 5, 6, 7, 8, 9, 10]):
            mfov = (level - 3) / 2
            fov_noise_1 = np.random.uniform(-mfov, mfov, 1)[0]
            fov_noise_2 = np.random.uniform(-mfov, mfov, 1)[0]
        elif level in [11, 12, 13]:
            mfov = 0.5
            fov_noise_1 = np.random.uniform(-mfov, mfov, 1)[0]
            fov_noise_2 = np.random.uniform(-mfov, mfov, 1)[0]
        else:
            fov_noise_1 = 0
            fov_noise_2 = 0

        # Robust 5
        if level in [1, 2, 4, 5, 9]:
            if level in [1, 6]:
                front_color_alpha = 1
            else:
                front_color_alpha = 0
            front_color_rgb = self.get_random_rgb()
            front_color = front_color_rgb + [front_color_alpha]
            self.bullet_client.changeVisualShape(self.my_items["case_sig"][0], 0, rgbaColor=front_color)

        # Robust 7
        if level in [4, 5, 6, 7, 8, 9, 10]:
            if level in [4, 5, 6]:
                # random robot color
                tip_color = (np.random.random([4]) - 0.5) * 0.1 + np.array([0.8, 0.8, 0.8, 1])
                tip_color[3] = 1
                self.bullet_client.changeVisualShape(self.my_items["robot"][0], 7, rgbaColor=tip_color)
                self.bullet_client.changeVisualShape(self.my_items["robot"][0], 8, rgbaColor=tip_color)
                self.bullet_client.changeVisualShape(self.my_items["robot"][0], 9, rgbaColor=tip_color)

                bolt_color = (np.random.random([4]) - 0.5) * 0.2 + np.array([0.2, 0.2, 0.2, 1])
                bolt_color[3] = 1
                self.bullet_client.changeVisualShape(self.my_items["robot"][0], 6, rgbaColor=bolt_color)

                case_color = np.array([0.8, 0.2, 0.2, 1]) + (np.random.random([4]) - 0.5) * 0.5
                case_color[3] = 1
                self.bullet_client.changeVisualShape(self.my_items["case"][0], 0, rgbaColor=case_color)
            elif level in [7, 8, 9, 10]:
                self.bullet_client.changeVisualShape(self.my_items["robot"][0], 7,
                                                     rgbaColor=self.get_random_color())
                self.bullet_client.changeVisualShape(self.my_items["robot"][0], 8,
                                                     rgbaColor=self.get_random_color())
                self.bullet_client.changeVisualShape(self.my_items["robot"][0], 9,
                                                     rgbaColor=self.get_random_color())
                self.bullet_client.changeVisualShape(self.my_items["robot"][0], 6,
                                                     rgbaColor=self.get_random_color())
                self.bullet_client.changeVisualShape(self.my_items["case"][0], 0, rgbaColor=self.get_random_color())

        # Robust 6
        if level in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            self.bullet_client.configureDebugVisualizer(lightPosition=[np.random.uniform(0, 3),
                                                                       np.random.uniform(0, 3),
                                                                       np.random.uniform(0, -5)])

        # Robust 9
        if level in [6, 7, 8, 9, 10, 11,12,13]:
            self.bullet_client.changeVisualShape(self.tv, -1, textureUniqueId=random.choice(self.bg_textures))
        elif level in [14,15]:
            random.seed(self.reset_times)
            self.bullet_client.changeVisualShape(self.tv, -1, textureUniqueId=random.choice(self.bg_textures))
            random.seed(time.time())
        # Robust 11
        self.rt11_(level)

        # Robust 20
        self.rt20_(level)

        #
        self.rt21_(level)

        # cam 1
        bolt_tap_pos = self.get_bolt_pose()

        cam_pos1 = list(np.add(ee_pose, self.args.camera_shitft_xyz) + noise_xyz_0)
        # cam_pos1 = list(np.add(ee_pose, debug_xyz1) + noise_xyz_0)
        target_point = list(np.add(bolt_tap_pos, noise_0))
        img0 = self.z_get_eye_img_bytargetpos(eye_pos=cam_pos1, target_pose=target_point,
                                              fov=self.args.fov + fov_noise_1)
        # fov=debug_fov + fov_noise_1)

        cam_pos2 = list(np.add(ee_pose, self.args.camera_shitft_xyz2) + noise_xyz_1)
        # cam_pos2 = list(np.add(ee_pose,debug_xyz2) + noise_xyz_1)
        target_point = list(np.add(bolt_tap_pos, noise_1))
        img1 = self.z_get_eye_img_bytargetpos(eye_pos=cam_pos2, target_pose=target_point,
                                              fov=self.args.fov2 + fov_noise_2)
        # fov=debug_fov + fov_noise_2)

        # Robust 12 Robust 13
        img0 = self.rt1213_(level, img0)
        img1 = self.rt1213_(level, img1)

        # # Robust 15
        # img0,img1=self.rt15_(level,img0,img1)
        # # Robust 16
        # img0,img1=self.rt16_(level,img0,img1)
        #
        # # Robust 17
        # img0=self.rt17_(level,img0)
        # img1=self.rt17_(level,img1)
        #
        # # Robust 19
        # img0=self.rt19_(level,img0,1)
        # img1=self.rt19_(level,img1,2)

        # Robust 22
        img0 = self.rt22_(level, img0)
        img1 = self.rt22_(level, img1)

        # Robust 23
        img0 = self.rt23_(level, img0)
        img1 = self.rt23_(level, img1)

        self.z_save_img(level, img0)
        self.z_save_img(level, img1)

        cat_img = self.z_fuse_img(img0, img1)
        cat_img = np.transpose(cat_img,[2,0,1])
        self.images[0] = cat_img
        return self.images[0]

    def z_fuse_img(self,img0,img1):
        # i0_resize = cv2.resize(img0, (int(size[0] / 2), size[1]))
        # i1_resize = cv2.resize(img1, (int(size[0] / 2), size[1]))
        cat_img=np.concatenate([img0,img1],axis=1).astype(np.uint8)
        return cat_img
    def reset(self):
        # reset some flags
        self.e.detect_coli_case = 0
        level=self.l
        if (level in [1, 2, 3]):
            max_translation = 0.01  # meter
            noise_0 = np.random.uniform(-level / 10 * max_translation, level / 10 * max_translation, [3])
            # noise_0[2]=0
            noise_1 = np.random.uniform(-level / 10 * max_translation, level / 10 * max_translation, [3])
            # noise_1[2]=0
            self.noise_0_from_reset = noise_0
            self.noise_1_from_reset = noise_1
        elif level in [14,15]:
            max_translation = 0.01  # meter
            n1 = np.random.uniform(-max_translation, max_translation, [3])
            n2 = np.random.uniform(-max_translation, max_translation, [3])
            n3 = np.random.uniform(-max_translation, max_translation, [3])
            n4 = np.random.uniform(-max_translation, max_translation, [3])
            # noise_1[2]=0
            self.cam_target_noise_0_from_reset = n1
            self.cam_target_noise_1_from_reset = n2
            self.cam_pos_noise_1_from_reset = n3
            self.cam_pos_noise_1_from_reset = n4

        # Robust 18
        self.rt18_(level)

        # for generate random seed
        self.reset_times+=1

    def _apply_action_to_sim(self, action):

        action = action[:]

        zoom_xyz = 0.02 * self.args.action_default_vel_scale
        zoom_rpy = 0.02 * self.args.action_default_vel_scale

        action[0] = action[0] * zoom_xyz
        action[1] = -action[1] * zoom_xyz

        target_e_pos = list(self.get_end_effect_pos_orn()[0])
        target_e_orn_eu = list(self.bullet_client.getEulerFromQuaternion(self.target_e_orn))
        updated = 0

        for i in range(len(action)):
            a_i = action[i]
            if (i == 1):
                target_e_pos[0] = target_e_pos[0] + a_i
                updated += 1
            if (i == 0):
                target_e_pos[1] = target_e_pos[1] + a_i
                updated += 1

        z_default_vel=self.args.z_default_vel
        target_e_pos[2] += z_default_vel

        self.e.target_e_pose = target_e_pos
        self.e.target_e_orn = self.bullet_client.getQuaternionFromEuler(target_e_orn_eu)

        jp = self.bullet_client.calculateInverseKinematics(
                self.my_items["robot"][0],
                self.JacoCartEEIndex,
                self.target_e_pose,
                self.target_e_orn
            )
        jp=list(jp)
        jp[3]=1.3
        jp[4]=1.3
        jp[5]=1.3

        # print(jp)

        numJoints = self.bullet_client.getNumJoints(self.my_items["robot"][0])

        for i in range(numJoints):
            # self.bullet_client.resetJointState(self.RobotUid, i, jp[i])
            self.bullet_client.setJointMotorControl2(self.RobotUid,
                                                     i,
                                                     self.bullet_client.POSITION_CONTROL,
                                                     targetPosition=jp[i],
                                                     force=5e4)

        # print(jp)
        self.e.prev_pos = self.target_e_pose
        return None
    # Robust Tools Box (call to make change to env)
    def rt1_(self):
        pass

    def rt11_(self, l):
        assert len(self.my_items["insert_detect"]) > 0
        # insert detect color change
        # if l ==1:
        c = [0, 0, 0, 0]
        self.bullet_client.changeVisualShape(self.my_items["insert_detect"][0], -1, rgbaColor=c)

    def rt1213_(self, l, np_img_arr):
        if l in [1, 2, 3, 4]:
            return np_img_arr
        elif l in [5, 6, 7, 8, 9, 10]:
            s = (l - 4) / 6 * self.rtscale_1213

            # # light noise
            # np_img_arr = np_img_arr * (1 + np.random.uniform(-0.2, 0, 1)*s) + np.random.uniform(-20, 20, 1)*s
            # # random noise
            # if random.random() < 0.1:
            #     np_img_arr = np.random.uniform(-10*s, 0, np_img_arr.shape) + np_img_arr
            # if random.random() < 0.03:
            #     np_img_arr = np.random.uniform(-20*s, 0, np_img_arr.shape) + np_img_arr
            # np_img_arr = np.clip(np_img_arr.astype(np.uint8), 1, 255)

            return np_img_arr.astype(np.uint8)
        else:
            return np_img_arr

    def rt15_(self, l, img0, img1):
        # s=(l-3)/7
        if l in [6, 8, 10] and random.random() < 0.5 * self.rtscale_15:
            return img1, img0
        if l in [4, 5, 7] and self.hash_update_per_epis < 0.5 * self.rtscale_15:
            return img1, img0
        return img0, img1

    def rt16_(self, l, img0, img1):
        # print('l=',l)
        s = (l - 3) / 7 * self.rtscale_16
        # return img0.om
        if l in [6, 8, 10] and random.random() < 0.3 + 0.2 * s:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        if l in [6, 8, 10] and random.random() < 0.3 + 0.2 * s:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if l in [4, 5, 7]:
            np.random.seed(int(self.hash_update_per_epis * 10))
            rd1 = np.random.random()
            np.random.seed(int(self.hash_update_per_epis * 5))
            rd2 = np.random.random()
            np.random.seed(int(time.time() * 1000 % 2 ** 32))
            # print("rand2",np.random.random(),np.random.random())
            if rd1 < 0.3 + 0.2 * s:
                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            if rd2 < 0.3 + 0.2 * s:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # imgs=[img0,img1]
        if len(img0.shape) == 2:
            img0 = np.tile(img0, [3, 1, 1]).transpose([1, 2, 0])
        if len(img1.shape) == 2:
            img1 = np.tile(img1, [3, 1, 1]).transpose([1, 2, 0])
        return img0, img1

    def rt17_(self, level, img):
        def gen_a_random_circle(image, maxr):
            center_coordinates = tuple(np.random.uniform(1, image.shape[0] - 1, [2]).astype(np.int))
            radius = int(random.random() * maxr + 1)
            color = tuple(np.random.uniform(0, 255, [3]).astype(np.int))
            color = (int(color[0]), int(color[1]), int(color[2]))
            thickness = -1
            # try:
            # image = cv2.circle(cv2.UMat(image), center_coordinates, radius, color, thickness)
            # try:
            image = cv2.circle(cv2.UMat(image), center_coordinates, radius, color, thickness).get()
            # except:
            #     pass
            # except:
            #     pass
            return image

        if level in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            s = (level - 4) / 6 * self.rtscale_17
            # loop=0
            while (random.random() < 0.4 + 0.1 * s and s > 0):
                # loop+=1
                # print(loop)
                img = gen_a_random_circle(img, int(s * 20 + 1))
        return img
        # pass

    def rt18_(self, level):
        if level in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            s = level / 10 * 0.6 + 0.4
            # if level==1:
            #     s=0
        elif level in [11, 12, 13]:
            s = 1 #0.5
        elif level in [14,15]:
            s = 1
        else:
            self._reset_jaco_pos()
            return None
        s = s * self.rtscale_18

        # reset position
        self._reset_jaco_pos()
        base_new = self.JacoCart_BasePos
        # base_new = [self.p.readUserDebugParameter(self.dbg_b1),
        #             self.p.readUserDebugParameter(self.dbg_b2),
        #             self.p.readUserDebugParameter(self.dbg_b3)]
        # pose_noise = (np.random.random([3]) - 0.5) * 0.02
        pose_noise = np.random.uniform(-1,1,3)*0.02
        pose_noise[2] = 0
        base_new = np.array(base_new) + pose_noise * s #+[0,0,0.1]
        # print(base_new)
        self.bullet_client.resetBasePositionAndOrientation(self.RobotUid, base_new,
                                                           self.JacoCart_BaseOrn)

    # def _terminal(self):
    #     return 1

    def rt19_(self, l, img, seed):
        # Left Right One Image Black
        if l in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            s = (l - 3) / 7
            # if l in [6, 8, 10] and random.random() < 0.5: # by frame
            #     img = np.random.uniform(1, 254, img.shape)
            if l in [5, 8, 10]:  # by epis
                # np.random.seed(int(self.hash_update_per_epis * 10)+seed)
                rd1 = np.random.random()

                # np.random.seed(int(time.time() * 1000 % 2 ** 32))
                # print("rand2",np.random.random(),np.random.random())
                if rd1 < 0.05 * s * self.rtscale_19:
                    img = np.random.uniform(1, 254, img.shape)
        return img

    def rt20_(self, l):
        if l in [6, 7, 8, 10]:
            # self.bullet_client.changeVisualShape(self.my_items["robot"][0], 6, rgbaColor=bolt_color)
            self.bullet_client.changeVisualShape(self.my_items["robot"][0], 6,
                                                 textureUniqueId=random.choice(self.bg_textures))

    def rt21_(self, l):
        # Gripper
        # Open
        # Close
        # Degree
        if l in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            d = 0.8
            s = (l - 3) / 7 * self.rtscale_21
            maxd = 0.5
        elif l in [11, 12, 13]:
            d = 0.8
            s = 0.2
            maxd = 0.5
        else:
            return None
        if s > 0:
            m = maxd * np.random.uniform(-1, 1, [3]) * s
            jp = list(m + d)
            jp = [[jpi] for jpi in jp]
            self.bullet_client.resetJointStatesMultiDof(self.RobotUid, [3,4,5],
                                                        targetValues=jp)

    def rt22_(self, l, img):
        #   gan BA
        if l in [11,14]:
            # if self.Task in ["yw_insert_g1img", "yw_insert_g1cimg"]:
            img = self.gan_gen(img, "ba")
        return img

    def rt23_(self, l, img):
        # GAN ba -> ab
        if l in [12]:
            # elif self.Task == "yw_insert_g1bimg":
            if (random.random() < 0.4):
                img = self.gan_gen(img, "ba")
                img = self.gan_gen(img, "ab")
            else:
                img = self.gan_gen(img, "ab")
        if l in [15]:
            img = self.gan_gen(img, "ba")
            img = self.gan_gen(img, "ab")
        return img
        # elif
    def test_robot(self):
        jp = [
            # self.dbg_jp1
            [self.bullet_client.readUserDebugParameter(self.dbg_jp1)],
            [self.bullet_client.readUserDebugParameter(self.dbg_jp2)],
            [self.bullet_client.readUserDebugParameter(self.dbg_jp3)],
            [self.bullet_client.readUserDebugParameter(self.dbg_jp4)],
            [self.bullet_client.readUserDebugParameter(self.dbg_jp5)],
            [self.bullet_client.readUserDebugParameter(self.dbg_jp6)]
        ]
        self.bullet_client.resetJointStatesMultiDof(self.RobotUid, [i for i in range(len(jp))], targetValues=jp)

        xyz = [
            # self.dbg_jp1
            self.bullet_client.readUserDebugParameter(self.t1),
            self.bullet_client.readUserDebugParameter(self.t2),
            self.bullet_client.readUserDebugParameter(self.t3)
        ]
        orn=[
            self.bullet_client.readUserDebugParameter(self.t4),
            self.bullet_client.readUserDebugParameter(self.t5),
            self.bullet_client.readUserDebugParameter(self.t6)
        ]
        self.bullet_client.resetBasePositionAndOrientation(self.RobotUid, xyz,self.p.getQuaternionFromEuler(orn))


#Task3: Sparse Reward Envs
# 3.1 2d sparse insert
class sparse1(yw_insert_v1img3cm):
    def __init__(self,env):
        super(sparse1, self).__init__(env)

    def get_observe_dict(self):
        # observe=super(sparse1, self)._observe()
        observe=self.timestep.get_obs()
        return {"image":observe}

    def _reward(self):
        return self._task_success()
# 3.2 6d sparse pick (debug)
class sparse2(yw_pick_v1img_5cm):
    def __init__(self,env):
        super(sparse2, self).__init__(env)
        # self.args=fake_args()
        # self.args.img_size=[15,15]
    def _terminal(self):
        if self._terminal_of_maxsteps():
            return 1
        return super(sparse2, self)._terminal()
    def _reward(self):
        lego_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0])[0]
        table_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["table"][0])[0]
        d_height = lego_height[1] - table_height[1]
        # print(d_height)
        if (abs(d_height) > 0.65 + 0.05 and self.check_collied(self.RobotUid,self.my_items["lego"][0])):
            # print("You Win, Reward = 1")
            return 1
        return 0
# 3.3 3d sparse reach
class sparse3(yw_reach_v1img):
    def __init__(self,env):
        super(sparse3, self).__init__(env)
        self.e.maxsteps = 85
    def _apply_action_to_sim(self,action):
        assert len(action)==self.args.action_len
        a=list(action)
        a=a[:3]+[0,0,0]+a[-1:]
        return super(sparse3, self)._apply_action_to_sim(a)
# 3.3.2 3d sparse push (for GuoSheng Demo)
class sparse4(sparse3):
    def __init__(self,env):
        super(sparse3, self).__init__(env)
        self.delay_terminal=3
        self.delayed=3
    def _terminal(self):
        if self.delayed==0:
            done = super(sparse4, self)._terminal()
        if self.delayed:
            self.delayed+=1
        if self.delayed>self.delay_terminal:
            self.delayed=0
            return 1
        return 0

# multi stage task
class D3Touch_v1(yw_reach_v1img):
    def __init__(self,env):

        self.bsize=[0.02,0.02,0.02]
        self.lego_touched=[0 for i in range(5)]
        self.boxes_pos=[]

        super(D3Touch_v1, self).__init__(env)
        self.maxsteps=30
    def _apply_action_to_sim(self,action):
        assert len(action)==self.args.action_len
        a=list(action)
        a=a[:3]+[0,0,0]+[1]
        return super(D3Touch_v1, self)._apply_action_to_sim(a)
    def create_box_body(self,bsize,rgba):
        box_vis = self.pyb.createVisualShape(
            self.pyb.GEOM_BOX,
            halfExtents=bsize,
            rgbaColor=rgba)
        box_col = self.pyb.createCollisionShape(
            self.pyb.GEOM_BOX,
            halfExtents=bsize)
        box1 = self.bullet_client.createMultiBody(
            baseMass=0,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            baseVisualShapeIndex=box_vis,
            baseCollisionShapeIndex=box_col
        )
        return box1
    def init_load_items(self):
        self.m.load_a_standard_table()
        self.env.maxsteps = 120

        bsize=self.bsize
        self.boxes_pos.append(
            [-0.1,0.15,-0.45]
        )


        self.env.my_items["lego"].append(
            self.create_box_body(bsize,[1,0,0,1])
        )

        kukaobjs = self.bullet_client.loadSDF(
            os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/kuka_with_gripper2.sdf"))
        self.env.RobotUid = kukaobjs[0]
        self.env.my_items["robot"].append(self.env.RobotUid)

    def reset(self):
        self.lego_touched = [0 for i in range(5)]
        # print("reset!")
        self.env.jointPositions = self.KUKA_DEFAULT_JP[:]
        self.bullet_client.resetBasePositionAndOrientation(self.RobotUid,
                                                           [0, 0, 0],
                                                           [-0.707107, 0.0, 0.0, 0.707107])

        self.env.numJoints = self.bullet_client.getNumJoints(self.RobotUid)
        for jointIndex in range(self.numJoints):
            self.bullet_client.resetJointState(self.RobotUid, jointIndex, self.jointPositions[jointIndex])
        self.env.target_e_pose = ROBOT_ENDEFFECT_SAFE_POSE[:]
        self.env.target_e_orn = self.bullet_client.getQuaternionFromEuler(ROBOT_ENDEFFECT_SAFE_ORN_EU[:])
        for legoid in range(len(self.my_items["lego"])):
            print(legoid)
            self.bullet_client.resetBasePositionAndOrientation(self.my_items["lego"][legoid],
                                                               self.boxes_pos[legoid], [1, 1, 1, 1])

    def _reward(self):
        return self._task_success()

    def _task_success(self):
        if (self.env.Task_Success_Updated):
            return self.env.Task_Success

        for legoid in range(len(self.boxes_pos)):
            if self.lego_touched[legoid]==0:
                lego_contact = len(self.bullet_client.getContactPoints(self.RobotUid, self.my_items["lego"][legoid]))
                if legoid==0:
                    self.lego_touched[legoid]+=lego_contact
                elif self.lego_touched[legoid-1]>0:
                    self.lego_touched[legoid] += lego_contact


        print(self.lego_touched)
        self.env.Task_Success=int(all(self.lego_touched[:len(self.boxes_pos)]))
        self.env.Task_Success_Updated = 1
        return self.env.Task_Success

    def _terminal(self):
        if self._terminal_of_maxsteps():
            return 1
        if self._table_collied():
            return 1
        if (self._task_success()):
            return self._task_success()
        # else:
        return 0

class D3Touch_v2(D3Touch_v1):
    def __init__(self,env):
        super(D3Touch_v2, self).__init__(env)

        self.maxsteps=90
    def init_load_items(self):
        super(D3Touch_v2, self).init_load_items()

        bsize=self.bsize
        self.boxes_pos.append([0.3,0.15,-0.5])
        self.env.my_items["lego"].append(
            self.create_box_body(bsize,[0,1,0,1])
        )


class D3Touch_v3(D3Touch_v2):
    def __init__(self,env):
        super(D3Touch_v3, self).__init__(env)

        self.maxsteps=120
    def init_load_items(self):
        super(D3Touch_v3, self).init_load_items()
        self.boxes_pos.append([0.05,0.2,-0.7])
        bsize=self.bsize
        self.env.my_items["lego"].append(self.create_box_body(bsize,[0,0,1,1]))


if __name__=="__main__":
    env="None"
    a=yw_pick_v1(env)
    # pass