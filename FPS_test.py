import time
import numpy as np
from My_Env import yw_robotics_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
expid = 0


def make_env():
    def _init():
        env = yw_robotics_env(taskname)
        return env
    return _init

if __name__=="__main__":


    taskname="yw_insf_v1"

    # env1 = yw_robotics_env(taskname, DIRECT=1,gan_srvs=4)
    num_cpu = 20

    if num_cpu>1:
        vecs = [make_env() for i in range(num_cpu)]
        env1 = SubprocVecEnv(vecs)
    else:
        env1 = yw_robotics_env(taskname, DIRECT=1, gan_srvs=4)

    loop=0
    ret=0

    lps=100
    while(True):
        st=time.time()
        for i in range(lps):
            loop+=1
            # action = np.random.uniform(-1,1,[6])
            action = np.random.uniform(-1,1,[num_cpu,6]).squeeze()
            obs, rew, done, info=env1.step(action) # env1.step(np.random.random([2])-0.3)
        print("FPS=",lps*num_cpu/(time.time()-st))