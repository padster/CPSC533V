import math
import pickle

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.orientation import Orientation
from util.vec import Vec3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print ("\n\n****\nLOADING\n\n")
STATE_DICT = torch.load("C:/Users/User/code/CPSC533V/project/models/boostOnly1v1.pt")
print ("\n\nLOADED!\n")

def add_state_vec3(state, vec3):
    state.append(vec3.x)
    state.append(vec3.y)
    state.append(vec3.z)

def add_state_phys(state, phys):
    add_state_vec3(state, phys.location) # pos
    #state.append(phys.velocity.pitch)
    #state.append(phys.velocity.yaw)
    #state.append(phys.velocity.roll)
    add_state_vec3(state, phys.velocity) # vel
    #add_state_vec3(state, phys.rotation) # rot
    state.append(phys.rotation.pitch)
    state.append(phys.rotation.yaw)
    state.append(phys.rotation.roll)
    add_state_vec3(state, phys.angular_velocity) # ang_vel


## HACK - model, should be shared with notebook and trainer.
class MyModel(nn.Module):
    def __init__(self, state_size, action_analog_size, action_digital_size):
        super(MyModel, self).__init__()
        
        # Start with a super simple multi-layer perceptron, one hidden layer 
        self.dimH          = 32 # hidden layer has 16 dimensions
        self.dimIn         = state_size
        self.dimOutAnalog  = action_analog_size
        self.dimOutDigital = action_digital_size
        self.dimOut = action_analog_size + action_digital_size
     
        self.model = torch.nn.Sequential(
           nn.Linear(self.dimIn, self.dimH),
           nn.ReLU(),
           nn.Linear(self.dimH, self.dimOut),
        )
        #self.model.to(device)

    def forward(self, x):
        # input data type needs to be converted to float
        return self.model(x.float())
        
    def save(self, modelID):
        path = os.path.join("models", "%s.pt" % modelID)
        torch.save(self.state_dict(), path)
        print('Saved model!\n\t%s' % path)
        
    def load(self, modelID):
        path = "C:/Users/User/code/CPSC533V/project/models/%s.pt" % modelID
        self.load_state_dict(STATE_DICT)
        print('Loaded model!\n\t%s' % path)


def debugTxt(R, car, msg):
    print (msg)
    R.begin_rendering()
    # print the action that the bot is taking
    R.draw_string_3d(car.physics.location, 2, 2, msg, R.white())
    R.end_rendering()

def clip(lo, x, hi):
    return max(lo, min(x, hi))

# BOT
class MyBot(BaseAgent):

    def initialize_agent(self):
        print( "\n\n\n***\n\nCREATING BOT\n\n***\n\n")

        # This runs once before the bot starts up
        self.controller_state = SimpleControllerState()

        nPerTeam = 1
        self.stateDim = 3 + 2 * nPerTeam * (4*3 + 1)
        self.model = MyModel(self.stateDim, 2, 1)
        self.model.load("boostOnly1v1")
        print ("Model loaded!")

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        state = []

        ball_location = packet.game_ball.physics.location
        add_state_vec3(state, ball_location) # b_pos

        my_car = packet.game_cars[self.index]
        add_state_phys(state, my_car.physics) # me_
        state.append(my_car.boost)

        enemy_car = packet.game_cars[1 - self.index]
        add_state_phys(state, enemy_car.physics) # e0_
        state.append(enemy_car.boost)
        assert (len(state) == self.stateDim)

        state_tensor = torch.from_numpy(np.array(state)).float()

        action_tensor = self.model(state_tensor)
        action = action_tensor.detach().numpy()
        assert (len(action) == 3)
        debugTxt(self.renderer, my_car, "(%.2f %.2f %.2f)" % (action[0], action[1], action[2]))

        # throttle, steer, boost
        self.controller_state.throttle = clip(-1, action[0], 1)
        self.controller_state.steer = clip(-1, action[1], 1)
        self.controller_state.boost = (action[2] > 0)

        return (self.controller_state)


