import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,0)
# planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
# boxId = p.loadURDF("my_robots/my_models/jaco_yw/jaco_v4_eebolt.urdf",startPos, startOrientation)
# boxId = p.loadURDF("my_robots/my_models/CartesianJaco/urdf/CartesianJaco2.urdf",startPos, startOrientation,useFixedBase=True)
# boxId = p.loadURDF("my_robots/my_models/jaco_description/urdf/jaco_robot.urdf.xacro",startPos, startOrientation,useFixedBase=True)
# boxId = p.loadURDF("my_robots/my_models/jaco_description/urdf/j.urdf",startPos, startOrientation,useFixedBase=True)
boxId = p.loadURDF("my_robots/my_models/jaco_description/urdf/jaco_cartesian.urdf",startPos, startOrientation,useFixedBase=True)
# boxId = p.loadURDF("my_robots/my_models/jaco_description/urdf/Cartesian.urdf",startPos, startOrientation,useFixedBase=True)
# p.changeVisualShape(boxId,)
# boxId = p.loadURDF("my_robots/my_models/jaco_description/urdf/jaco_yw.urdf.xacro",startPos, startOrientation)
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()
