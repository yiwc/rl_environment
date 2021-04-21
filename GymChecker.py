# import stable_baselines2 as bs2
# import stable_baselines as bs
# from stable_baselines3.common import env_checker
import env_checker
from My_Env import yw_robotics_env
if __name__=="__main__":
    taskname="alpoderl2"
    env1 = yw_robotics_env(taskname,
                           DIRECT=1,
                           gan_srvs=1,
                           gan_dgx=True,
                           gan_port=5660)

    check_res=env_checker.check_env(env1)
    print("Checking->",check_res)