import time
import numpy as np
import math
import os
import pybullet_data
import random
import pickle
import copy
import shutil
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

ROBOT_BASE_ORN_DEFAULT=[-0.707107, 0.0, 0.0, 0.707107]  #p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
ROBOT_BASE_POS_1_DEFAULT=[5,0,0]# PAND
ROBOT_BASE_POS_2_DEFAULT=[0,0,0]# IIWA
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
KUKA_DEFAULT_JP=[
            1.6, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
            -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
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

# TODO define jaco LL UL JR limits

# PANDA_LL = [-7] * pandaNumDofs
# PANDA_UL = [7] * pandaNumDofs
# PANDA_JR = [7] * pandaNumDofs
# restposes for null space
# fPANDA_jointPositions = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
# PANDA_RP = PANDA_jointPositions

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
    def __init__(self, bullet_client,Task):

        self.timestep=TimeStep(sim=self)
        self.temp_timestep=TimeStep(sim=self)

        def init_load_items(self):
            flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
            self.flags = flags
            self.legos = []

            # Load items
            self.my_items = {}
            self.my_items["lego"] = []
            self.my_items["table"] = []
            self.my_items["robot"] = []
            self.my_items["case"] = []
            self.my_items["bolt"] = []
            self.my_items["plug"] = []
            # self.my_items_config["table"]={"height":0.}

            if (self.Task in ["yw_pick_v1","yw_pick_v1img","yw_pick_v1img_5cm","yw_reach_v1img"] ):
                self.maxsteps=500
                self.my_items["lego"].append(
                    self.bullet_client.loadURDF("lego/lego.urdf", np.array([-0.1, 0.3, -0.5]), flags=flags))
                # load kuka
                kukaobjs = self.bullet_client.loadSDF(
                    os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/kuka_with_gripper2.sdf"))
                self.RobotUid = kukaobjs[0]
                self.my_items["robot"].append(self.RobotUid)

            #TODO add kinova items
            elif (self.Task in ["yw_insert_v1img3cm"]):
                self.maxsteps = 500
                # self.my_items["lego"].append(
                #     self.bullet_client.loadURDF("lego/lego.urdf", np.array([-0.1, 0.3, -0.5]), flags=flags))

                # case_urdf=os.path.join(self._get_model_rootdir(),"objects","case.urdf")
                case_urdf=os.path.join(self._get_model_rootdir(),"objects","case_hole_detect.urdf")

                ## Insert Test
                # self.my_items["case"].append(
                #     # self.bullet_client.loadURDF(case_urdf,np.array([0, 0.3, -0.5]), [0, 0, 1, 1], flags=self.flags)
                #     self.bullet_client.loadURDF(case_urdf,np.array([0, 0.2, -0.5]), [1, 0, 0, 1], flags=self.flags)
                #     # self.bullet_client.loadURDF(case_urdf,np.array([0, 0.2, -0.5]), [0, 1, 0, 1], flags=self.flags)
                # )
                # bolt=self.bullet_client.createMultiBody(baseMass=0.01, basePosition=[0.125, 0.25, -0.44], #baseOrientation=[0, 0, 0, 1],
                #                                         baseOrientation=[1, 0, 0, 1],
                #                          baseCollisionShapeIndex=self.bullet_client.createCollisionShape(
                #                              self.bullet_client.GEOM_CYLINDER,height=0.15,radius=0.005),
                #                          baseVisualShapeIndex=self.bullet_client.createVisualShape(self.bullet_client.GEOM_CYLINDER,
                #                                                                     rgbaColor=[0.3, 0.3, 0.3, 1],
                #                                                                    length=0.15,radius=0.0045))

                self.my_items["case"].append(
                    self.bullet_client.loadURDF(case_urdf,np.array([0.25, 0.3, -0.4]), [0, 0, 1, 1],flags=self.flags)
                )

                # bolt=self.bullet_client.createMultiBody(baseMass=0.01, basePosition=[0.125, 0.25, -0.44], #baseOrientation=[0, 0, 0, 1],
                #                                         baseOrientation=[1, 0, 0, 1],
                #                          baseCollisionShapeIndex=self.bullet_client.createCollisionShape(
                #                              self.bullet_client.GEOM_CYLINDER,height=0.15,radius=0.005),
                #                          baseVisualShapeIndex=self.bullet_client.createVisualShape(self.bullet_client.GEOM_CYLINDER,
                #                                                                     rgbaColor=[0.3, 0.3, 0.3, 1],
                #                                                                    length=0.15,radius=0.0045))

                # bolt_urdf=os.path.join(self._get_model_rootdir(),"objects","M10CasingBolt.urdf")
                # bolt=self.bullet_client.loadURDF(bolt_urdf,np.array([0, 0.3, -0.5]), [0, 0, 1, 1], globalScaling=0.001,flags=self.flags)


                # self.my_items["bolt"].append(bolt)


                # plug_urdf=os.path.join(self._get_model_rootdir(),"objects","plug.urdf")
                # self.my_items["plug"].append(
                #     self.bullet_client.loadURDF(plug_urdf,np.array([0, 0.3, -0.5]), flags=self.flags)
                # )

                # jaco_urdf=os.path.join(self._get_model_rootdir(),'jaco_yw', "j2s7s300_standalone.urdf")
                jaco_urdf=os.path.join(self._get_model_rootdir(),'jaco_yw', "j2s7s300_bolt_2.urdf")
                jacoobjs=self.bullet_client.loadURDF(fileName=jaco_urdf,
                                                     flags=self.bullet_client.URDF_MERGE_FIXED_LINKS)

                self.RobotUid = jacoobjs
                self.my_items["robot"].append(self.RobotUid)

            elif (self.Task == "ToBeDetermined"):

                self.legos.append(
                    self.bullet_client.loadURDF("lego/lego.urdf", np.array([0.1, 0.3, -0.5]), flags=self.flags))
                self.bullet_client.changeVisualShape(self.legos[0], -1, rgbaColor=[1, 0, 0, 1])  # 换颜色
                self.legos.append(
                    self.bullet_client.loadURDF("lego/lego.urdf", np.array([-0.1, 0.3, -0.5]), flags=self.flags))
                self.legos.append(
                    self.bullet_client.loadURDF("lego/lego.urdf", np.array([0.1, 0.3, -0.7]), flags=self.flags))
                self.sphereId = self.bullet_client.loadURDF("sphere_small.urdf", np.array([0, 0.3, -0.6]),
                                                            flags=self.flags)
                self.bullet_client.loadURDF("sphere_small.urdf", np.array([0, 0.3, -0.5]), flags=self.flags)
                self.bullet_client.loadURDF("sphere_small.urdf", np.array([0, 0.3, -0.7]), flags=self.flags)
                self.bullet_client.loadURDF(os.path.join(os.getcwd(), "my_models", "dinnerware/pan_tefal.urdf"),
                                            np.array([0, 0.3, -0.7]) + self.offset, flags=flags)

            else:
                raise NotImplemented("Haven't given TASK NAME")
            self.my_items["table"].append(
                self.bullet_client.loadURDF("table/table.urdf", [0, -0.6, -0.6], [-0.5, -0.5, -0.5, 0.5],
                                            flags=self.flags))

            # # Load robot
            # self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", ROBOT_BASE_POS_1_DEFAULT, ROBOT_BASE_ORN_DEFAULT,useFixedBase=True, flags=flags)


            # load jaco


        self.Task=Task
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array([0,0,0]) # cancable
        # self.prev_pos = ROBOT_ENDEFFECT_SAFE_POSE[:]
        self.control_dt = 1. / 240.# will be set outside the loop
        self.exp_recording=0 # 自动保存所有的轨迹路线，每一个step
        self.expid=None
        self.saves_list_dict={}# {'task':{'expid':[1,2,3,4...]}}
        self.exp_num=0
        self.action=[] # to be given by user or agent
        self.Task_Success_Updated=0
        # self.finger_target = 0
        # self.gripper_height = 0.2

        if (self.Task in ["yw_pick_v1", "yw_pick_v1img", "yw_pick_v1img_5cm", "yw_reach_v1img"]):
            self.robot_name="kuka"
        elif (self.Task in ["yw_insert_v1img3cm"]):
            # TODO reset items # TODO change to Gearbox and bolt
            self.robot_name="jaco"
        else:
            raise NotImplementedError("Task Not defined, so the robot can not find")

        # Load Items and Robot
        init_load_items(self)

        # Define speces
        # self.action_spec=self.get_action_spec()


        # Reset state
        self.reset()

    def _get_model_rootdir(self):
        dir=os.getcwd()
        dirs=["pybullet_env","my_robots","my_models"]
        for d in dirs:
            if(d not in dir.__str__()):
                dir=os.path.join(dir,d)
        return dir
    def action_spec(self):


        if(self.Task in ["yw_pick_v1","yw_pick_v1img","yw_pick_v1img_5cm","yw_reach_v1img"]):
            maxmimum=np.array([1,1,1,1,1,1,1])
            minimum = np.array([-1,-1,-1,-1,-1,-1,-1])
            shape = tuple([7])
        elif(self.Task in ["yw_insert_v1img3cm"]):
            maxmimum=np.array([1,1])
            minimum = np.array([-1,-1])
            shape = tuple([2])
        myspec = spec(maximum=maxmimum,minimum=minimum,shape=shape)
        return myspec

    def observation_spec(self):#
        obs_spec_dict={}
        class Array(object):
            def __init__(self,shape,dtype,name):
                self.shape=shape
                self.dtype=dtype
                self.name=name
        if(self.Task=="yw_pick_v1"):
            obs_spec_dict["cartesian_pos"]=Array((3,),np.dtype('float64'),name="cartesian_pos")
            obs_spec_dict["cartesian_orn"]=Array((3,),np.dtype('float64'),name="cartesian_orn")
            obs_spec_dict["block_posorn"]=Array((3,),np.dtype('float64'),name="block_posorn")
            obs_spec_dict["block_posorn"]=Array((3,),np.dtype('float64'),name="block_posorn")
        if(self.Task=="yw_pick_v1img"):
            obs_spec_dict["cartesian_pos"]=Array((3,),np.dtype('float64'),name="cartesian_pos")
            obs_spec_dict["cartesian_orn"]=Array((3,),np.dtype('float64'),name="cartesian_orn")
            obs_spec_dict["block_posorn"]=Array((3,),np.dtype('float64'),name="block_posorn")
            obs_spec_dict["block_posorn"]=Array((3,),np.dtype('float64'),name="block_posorn")
            # obs_spec_dict["image"]=self.bullet_client.getCameraImage(64,64)
            raise NotImplementedError("you havent set image spec, may this is not useful")
        return obs_spec_dict

    def reset(self): # Set the env to step = 0 and state from zero state

        if(self.exp_recording):
            self.clean_game_svae_buffer()
        # Reset Time
        self.t = 0.
        self.steps = 0
        self.Task_Success_Updated=0
        # Reset KUKA (Position, Orn, Joint, )
        if(self.robot_name=="kuka"):
            self.jointPositions = KUKA_DEFAULT_JP[:]
            #reset base
            self.bullet_client.resetBasePositionAndOrientation(self.RobotUid, ROBOT_BASE_POS_2_DEFAULT[:],
                                                               ROBOT_BASE_ORN_DEFAULT[:])
            #reset jp
            self.numJoints = self.bullet_client.getNumJoints(self.RobotUid)
            for jointIndex in range(self.numJoints):
                self.bullet_client.resetJointState(self.RobotUid, jointIndex, self.jointPositions[jointIndex])
            # reset epose
            self.target_e_pose = ROBOT_ENDEFFECT_SAFE_POSE[:]
            self.target_e_orn = self.bullet_client.getQuaternionFromEuler(ROBOT_ENDEFFECT_SAFE_ORN_EU[:])

        elif(self.robot_name=="jaco"):
            print("RESET!")
            self.bullet_client.resetBasePositionAndOrientation(self.RobotUid, ROBOT_BASE_POS_3_DEFAULT[:],
                                                               ROBOT_BASE_ORN_DEFAULT[:])

            self.numJoints = self.bullet_client.getNumJoints(self.RobotUid)

            # self.target_e_pose = []

            # self.target_e_pose = [0.1,0.4,-0.5]
            self.target_e_pose = [0.3,0.4,-0.8] #[-outscreen/+inscreen,+up/-down,+right/-left]
            # self.target_e_orn =  self.bullet_client.getQuaternionFromEuler([0,0,0])/
            self.target_e_orn =  self.bullet_client.getQuaternionFromEuler([0,math.pi,0])

            jp = self.bullet_client.calculateInverseKinematics(
                self.RobotUid,
                jacoEndEffectorIndex,
                self.target_e_pose,
                self.target_e_orn,
                JACO_ll,
                JACO_ul,
                JACO_jr,
                JACO_rp,  # TODO Change LL UL JR RP to JACO
                maxNumIterations=20
            )

            for jointIndex in range(len(jp)):
                self.bullet_client.resetJointState(self.RobotUid, jointIndex, jp[jointIndex])
            for jointIndex in range(7,10):
                self.bullet_client.resetJointState(self.RobotUid, jointIndex, 1.3)

            # Reset Color
            # self.bullet_client.changeVisualShape(
            #     self.my_items["bolt"][0],-1,rgbaColor=[0.2,0.2,0.2,1]
            # )

            # Set Robot Color
            # finger_color=[0.7,0.7,0.7,1]
            # self.bullet_client.changeVisualShape(self.RobotUid,7,rgbaColor=finger_color)
            # self.bullet_client.changeVisualShape(self.RobotUid,8,rgbaColor=finger_color)
            # self.bullet_client.changeVisualShape(self.RobotUid,9,rgbaColor=finger_color)
            #
            # hand_color = [0, 0, 0, 1]
            # self.bullet_client.changeVisualShape(self.RobotUid, 6, rgbaColor=hand_color)


            # self.bullet_client.getLinkState()
        # #jaco
        # ROBOT_RARM_INSERT_READYPOSE=[0,0.4,-0.5] # IIWA # x z y
        # ROBOT_RARM_INSERT_READYPOSE_ORN=[math.pi / 2., 0., 0.] # IIWA
        # Reset Items

        if (self.Task in ["yw_pick_v1","yw_pick_v1img","yw_pick_v1img_5cm","yw_reach_v1img"]):
            self.bullet_client.resetBasePositionAndOrientation(self.my_items["lego"][0],
                                                               [-0.1 + random.random() * 0.1, 0.1,
                                                                -0.5 + random.random() * 0.1], [1, 1, 1, 1])
        elif (self.Task in ["yw_insert_v1img3cm"]):
            # TODO reset items # TODO change to Gearbox and bolt
            # self.bullet_client.resetBasePositionAndOrientation(self.my_items["lego"][0],
            #                                                    [-0.1 + random.random() * 0.1, 0.1,
            #                                                     -0.5 + random.random() * 0.1], [1, 1, 1, 1])
            pass
        else:
            raise NotImplemented

        #Load observation and so on
        self.timestep.load()
        # return self._observe()
        return self.timestep

    def save_game_into_buffer(self,expid):

        steps=self.steps
        file_dir_path=os.path.join("game_saves",self.Task,"buffer")
        # assert

        # shutil.rmtree(file_dir_path)
        # os.mkdir(file_dir_path) # clean the buffer
        if(not os.path.exists(file_dir_path)):

            os.mkdir(file_dir_path) # clean the buffer
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
        # robot_dict={"name":"kukav1",
        #             "jp":KUKA_DEFAULT_JP,
        #             "jv":KUKA_DEFAULT_JP,
        #             "jointReactionForces":KUKA_DEFAULT_JP,
        #             "appliedJointMotorTorque":KUKA_DEFAULT_JP}
        action_dict={"name":"action","sub_id":0,
                     "action":self.action}

        observe_dict=self._observe()
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
            assert os.path.exists(file_dir_path)

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
        if(self.exp_num!=0):
            return self.exp_num
        else:
            file_dir_path = os.path.join("game_saves", task)
            flist = os.listdir(file_dir_path)
            self.exp_num=len(flist)
            return len(flist)
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
    def step(self,action):

        # self._callback_before_every_step()

        # print(action)
        if(self.if_key_board_detected()):
            return
        def step_t(self):
            t=self.t
            self.t += self.control_dt
            self.steps+=1
        step_t(self)
        self.Task_Success_Updated=0
        if(self.Task in ["yw_pick_v1",
                         "yw_pick_v1img",
                         "yw_pick_v1img_5cm",
                         "yw_reach_v1img"]):
            # Kuka based env
            self.Apply_action_kuka(action)
        elif(self.Task in ["yw_insert_v1img3cm"]):
            # Kinova jaco based env
            self.Apply_action_jaco(action)
            pass
        # if(self.Task in ["yw_insert_v1img3cm"]):

        self.action=action[:]
        self.timestep.load()
        # obs=self.timestep.get_obs()
        # reward=self.timestep.get_reward()
        done=self.timestep.get_done()
        # obs=self._observe()
        # # print("obs= ",obs)
        # reward=self._reward()
        # done=self._terminal()

        if(done):
            # print("Done")
            if(self.exp_recording):
                print("Do You Like Last Recording? Y/N")
                # if(y=="Y" or y=="y"):
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
            # del temp_timestep
            self.temp_timestep.load()
            self.reset()
        if(self.exp_recording):
            self.save_game_into_buffer(self.expid)


        # timestep_copy=self.timestep
        # timestep
        if(done):
            # temp_timestep
            return self.temp_timestep
        else:
            self.timestep.load()
            return self.timestep
        # return

    def get_end_effect_pose(self):
        state = self.bullet_client.getLinkState(self.RobotUid, kukaEndEffectorIndex)
        # print("state", state
        return list(state[4])  # state[4] is the worldLinkFramePosition
    def get_end_effect_orn(self):
        state = self.bullet_client.getLinkState(self.RobotUid, kukaEndEffectorIndex)
        # print("state", state
        # print("state,",state)
        return list(state[5])  # state[4] is the worldLinkFramePosition
    def get_end_effect_pos_orn(self):
        state = self.bullet_client.getLinkState(self.RobotUid, kukaEndEffectorIndex)
        # print("state", state
        return list(state[4:6])  #
    def Apply_action_jaco(self,action):

        # TODO develop jaco action implement
        action = action[:]
        # self.action=action
        zoom_xyz = 0.003
        zoom_rpy = 0.005
        action[0] = action[0] * zoom_xyz
        # action0 for in/out screen -> targetpose[0]
        action[1] = action[1] * zoom_xyz
        # action1 for up/down -> targetpose[1]

        # t=self.t
        target_e_pos = self.target_e_pose
        # print(target_e_pos)
        target_e_orn_eu = list(self.bullet_client.getEulerFromQuaternion(self.target_e_orn))
        updated = 0
        # print(self.bullet_client.getLinkState(self.RobotUid, 0)[5])
        for i in range(len(action)):
            a_i = action[i]
            if (i == 0):
                target_e_pos[0] = target_e_pos[0] + a_i
                updated += 1
            if (i == 1):
                target_e_pos[1] = target_e_pos[1] + a_i
                updated += 1

        self.target_e_pose = target_e_pos
        self.target_e_orn = self.bullet_client.getQuaternionFromEuler(target_e_orn_eu)

        self.bullet_client.submitProfileTiming("IK")

        jp = self.bullet_client.calculateInverseKinematics(
                self.RobotUid,
                jacoEndEffectorIndex,
                self.target_e_pose,
                self.target_e_orn,
                JACO_ll,
                JACO_ul,
                JACO_jr,
                JACO_rp,  # TODO Change LL UL JR RP to JACO
                maxNumIterations=20
            )
        # print("self.target_e_orn",self.ta
        # rget_e_pose)
        # self.bullet_client.submitProfileTiming()  # ?
        for i in range(jacoEndEffectorIndex):
            self.bullet_client.setJointMotorControl2(self.RobotUid, i,
                                                     self.bullet_client.POSITION_CONTROL,
                                                     jp[i], force=5 * 240.)

        self.prev_pos = self.target_e_pose

        return None
    def Apply_action_kuka(self,action):
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


        self.target_e_pose=target_e_pos
        self.target_e_orn=self.bullet_client.getQuaternionFromEuler(target_e_orn_eu)

        self.bullet_client.submitProfileTiming("IK")

        jointPoses_iiwa = self.bullet_client.calculateInverseKinematics(
            self.RobotUid,
            kukaEndEffectorIndex,
            self.target_e_pose,
            self.target_e_orn,
            KUKA_LL,
            KUKA_UL,
            KUKA_JR,
            KUKA_RP,
            maxNumIterations=20
        )

        self.bullet_client.submitProfileTiming() # ?
        for i in range(kukaEndEffectorIndex):
            self.bullet_client.setJointMotorControl2(self.RobotUid, i, self.bullet_client.POSITION_CONTROL,
                                                     jointPoses_iiwa[i], force=5 * 240.)

        self.prev_pos = self.target_e_pose

        return None
        #
        # self.bullet_client.submitProfileTiming("step")
        # # self.update_state()
        # # print("self.state=",self.state)
        # # print("self.finger_target=",self.finger_target)
        # alpha = 0.9  # 0.99
        # if self.state == 1 or self.state == 2 or self.state == 3 or self.state == 4 or self.state == 7:
        #     # gripper_height = 0.034
        #     self.gripper_height = alpha * self.gripper_height + (1. - alpha) * 0.03
        #     if self.state == 2 or self.state == 3 or self.state == 7:
        #         self.gripper_height = alpha * self.gripper_height + (1. - alpha) * 0.2
        #
        #     t = self.t
        #     self.t += self.control_dt
        #     pos = [self.offset[0] + 0.2 * math.sin(1.5 * t), self.offset[1] + self.gripper_height,
        #            self.offset[2] + -0.6 + 0.1 * math.cos(1.5 * t)]
        #     if self.state == 3 or self.state == 4:
        #         pos, o = self.bullet_client.getBasePositionAndOrientation(self.legos[0])
        #         pos = [pos[0], self.gripper_height, pos[2]]
        #         self.prev_pos = pos
        #     if self.state == 7:
        #         pos = self.prev_pos
        #         diffX = pos[0] - self.offset[0]
        #         diffZ = pos[2] - (self.offset[2] - 0.6)
        #         self.prev_pos = [self.prev_pos[0] - diffX * 0.1, self.prev_pos[1], self.prev_pos[2] - diffZ * 0.1]
        #
        #     orn = self.bullet_client.getQuaternionFromEuler([math.pi / 2., 0., 0.])
        #     self.bullet_client.submitProfileTiming("IK")
        #     jointPoses = self.bullet_client.calculateInverseKinematics(self.panda,
        #                                                                pandaEndEffectorIndex,
        #                                                                pos,
        #                                                                orn,
        #                                                                ll,
        #                                                                ul,
        #                                                                jr, rp, maxNumIterations=20)
        #
        #     self.bullet_client.submitProfileTiming()
        #     for i in range(pandaNumDofs):
        #         self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
        #                                                  jointPoses[i], force=5 * 240.)
        #     # target for fingers
        # for i in [9, 10]:
        #     self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
        #                                              self.finger_target, force=10)
        # self.bullet_client.submitProfileTiming()

    def _reward(self):
        if (self.Task in ["yw_pick_v1","yw_pick_v1img"]):
            lego_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0])[0]
            table_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["table"][0])[0]
            d_height = lego_height[1] - table_height[1]
            # print(d_height)
            if (abs(d_height) > 0.65 + 0.1):
                # print("You Win, Reward = 1")
                return 1
            return 0
        elif (self.Task in ["yw_pick_v1img_5cm"]):
            lego_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0])[0]
            table_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["table"][0])[0]
            d_height = lego_height[1] - table_height[1]
            # print(d_height)
            if (abs(d_height) > 0.65 + 0.05):
                # print("You Win, Reward = 1")
                return 1
            return 0
        elif(self.Task in ["yw_reach_v1img"]): # Bug Here, not implemented
            # print("Error: Reward Not implemented")
            return self._task_success()
            # return 1
        elif(self.Task in ["yw_insert_v1img3cm"]):
            # TODO reward design not finished
            return 0
        # return 0
        raise NotImplementedError("TASK Not specified, no reward will be generated")
    def _terminal(self):
        if self.steps>self.maxsteps:
            return 1
        if (self.Task in ["yw_pick_v1","yw_pick_v1img"]):
            lego_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0])[0]
            table_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["table"][0])[0]
            d_height = lego_height[1] - table_height[1]
            # print(d_height)
            if (abs(d_height) > 0.65 + 0.1):
                return 1
            # print(table_height)
        if (self.Task in ["yw_pick_v1img_5cm"]):
            lego_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0])[0]
            table_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["table"][0])[0]
            d_height = lego_height[1] - table_height[1]
            # print(d_height)
            if (abs(d_height) > 0.65 + 0.05):
                return 1
        if (self.Task in ["yw_reach_v1img","yw_insert_v1img3cm"]):
            # print("Terminal Not implemented")
            if (self._task_success()):
                # print("Success!")
                return self._task_success()
            # return self._task_success()
            # return 0
            # print(table_height)
        # can not touch table
        table_collision = self.bullet_client.getContactPoints(self.RobotUid, self.my_items["table"][0])
        if (table_collision):
            return 1

        # print("done")
        return 0
        pass
    def _task_success(self):
        if(self.Task in ["yw_reach_v1img"]):
            if(self.Task_Success_Updated):
                return self.Task_Success
            # lego_pos = self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0])
            # robot_pos = self.bullet_client.getBasePositionAndOrientation(self.my_items["robot"][0])
            # table_height = self.bullet_client.getBasePositionAndOrientation(self.my_items["table"][0])[0]
            lego_contact = self.bullet_client.getContactPoints(self.RobotUid, self.my_items["lego"][0])
            if(lego_contact):
                self.Task_Success=1
            else:
                self.Task_Success=0
            self.Task_Success_Updated=1
            # reset in every step
            return self.Task_Success
        elif(self.Task in ["yw_insert_v1img3cm"]):
            #TODO task success not implemented
            return False
        raise NotImplemented("You'r Task is not defined")
    def _get_external_observe(self):

        # camEyePos = [0.03,0.236,0.54]
        # distance = 1.06
        # pitch=-56
        # yaw = 258
        # roll=0
        # upAxisIndex = 2
        # camInfo = p.getDebugVisualizerCamera()
        # print("width,height")
        # print(camInfo[0])
        # print(camInfo[1])
        # print("viewMatrix")
        # print(camInfo[2])
        # print("projectionMatrix")
        # print(camInfo[3])
        # viewMat = camInfo[2]
        if(self.Task in ["yw_insert_v1img3cm"]):
            # TODO add wrhist image
            return None
        else:
            cameraPOS=[0,0.2,-0.6]
            distance=0.1
            yaw=0
            pitch=0
            roll=0
            upAxisIndex=1
            viewMat = self.bullet_client.computeViewMatrixFromYawPitchRoll(cameraPOS,distance,yaw, pitch,roll,upAxisIndex)
            projMatrix = [
                0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0,
                -0.02000020071864128, 0.0
            ]
            imgh=64
            imgw=64
            img_arr = self.bullet_client.getCameraImage(width=imgh,
                                       height=imgw,
                                       viewMatrix=viewMat,
                                       projectionMatrix=projMatrix)
            rgb = img_arr[2][:,:,:3]
            np_img_arr = np.reshape(rgb, (imgh,imgw, 3))
            # self._observation = np_img_arr
            return np_img_arr


    def _observe(self):
        observe_dict={}
        def clear_dict(obser_dict):
            for key in obser_dict.keys():
                del obser_dict[key]
        if(self.Task=="yw_pick_v1"):
            cartesian_posorn=self.get_end_effect_pos_orn()
            cartesian_posorn[1]=self.bullet_client.getEulerFromQuaternion(cartesian_posorn[1])
            block_posorn=list(self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0]))
            block_posorn[1] = self.bullet_client.getEulerFromQuaternion(block_posorn[1])
            observe_dict["cartesian_pos"]=np.array(cartesian_posorn[0])
            observe_dict["cartesian_orn"]=np.array(cartesian_posorn[1])
            observe_dict["block_pos"]=np.array(block_posorn[0])
            observe_dict["block_orn"]=np.array(block_posorn[1])
            return observe_dict
        if(self.Task in ["yw_pick_v1img","yw_pick_v1img_5cm","yw_reach_v1img"]):
            cartesian_posorn=self.get_end_effect_pos_orn()
            cartesian_posorn[1]=self.bullet_client.getEulerFromQuaternion(cartesian_posorn[1])
            block_posorn=list(self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0]))
            block_posorn[1] = self.bullet_client.getEulerFromQuaternion(block_posorn[1])
            clear_dict(observe_dict)
            observe_dict["cartesian_pos"]=np.array(cartesian_posorn[0])
            observe_dict["cartesian_orn"]=np.array(cartesian_posorn[1])
            observe_dict["block_pos"]=np.array(block_posorn[0])
            observe_dict["block_orn"]=np.array(block_posorn[1])
            # observe_dict["image"]=np.array(self.bullet_client.getCameraImage(64,64))
            # del observe_dict["image"]
            observe_dict["image"]=self._get_external_observe()
            return observe_dict
        elif(self.Task in ["yw_insert_v1img3cm"]):
            # TODO define the whrist vedio input
            observe_dict["image"]=self._get_external_observe()
            return None
    def if_key_board_detected(self):
        keys = self.bullet_client.getKeyboardEvents()
        # print(1)
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
    @property
    def _info(self):
        info={}
        info["target_e_pose"]=self.target_e_pose
        info["target_e_orn"]=self.target_e_orn
        info["steps"]=self.steps
        info["maxsteps"]=self.maxsteps
        info["robot_pose"]=self.bullet_client.getBasePositionAndOrientation(self.my_items["robot"][0])
        info["lego[0]_pose"]=self.bullet_client.getBasePositionAndOrientation(self.my_items["lego"][0])
        return info
