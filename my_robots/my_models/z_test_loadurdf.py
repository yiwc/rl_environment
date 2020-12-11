import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
robotid = p.loadURDF("jaco_yw/j2s7s300_bolt_2.urdf", flags=p.URDF_MERGE_FIXED_LINKS)

# cubeStartPos = [0,0,1]
# cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
# boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)

for i in range (10000):
    for jointIndex in range(7, 10):
        p.resetJointState(robotid, jointIndex, 1.3)
    p.stepSimulation()
    time.sleep(1./240.)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
p.disconnect()
