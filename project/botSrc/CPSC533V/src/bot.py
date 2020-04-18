# RLBot code, for an interface between RLBot state/action and our
#       own trained agent's pytorch state/actions

# PARAMETERS
USE_GPU = False
BC_MODEL, DQN_Model = None, None
DQN_MODEL = None # "dqn.2layer.BN.SandARewards"
# BC_MODEL = "bc.all3Replays" # 3.3: BC trained
# DQN_MODEL = "dqn.forwardPenalized" # 4.3: DQN validation
DQN_MODEL = "dqn.all3Replays" # 4.4: DQN trained
assert ((BC_MODEL is None) ^ (DQN_MODEL is None)), "Need either a BC or DQN model"

import math
import numpy as np
import pickle
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from util.orientation import Orientation
from util.vec import Vec3

ROOT_PROJECT_PATH = 'C:/Users/User/code/CPSC533V/project'
sys.path.append(ROOT_PROJECT_PATH)
import lib.preprocess as libPreprocess
import lib.rewards as libRewards
from lib.SAtoVModel import SAtoV_Model
from lib.StoAModel import StoA_Model

# Load the model now: Note we can't load it while the bot is running, too slow.
print ("\n\n****\nLOADING pytorch model...\n\n")
if BC_MODEL is not None:
    STATE_DICT = torch.load(os.path.join(ROOT_PROJECT_PATH, "models", "%s.pt" % BC_MODEL))
elif DQN_MODEL is not None:
    STATE_DICT = torch.load(os.path.join(ROOT_PROJECT_PATH, "models", "%s.pt" % DQN_MODEL))
print ("\n\nLOADED!\n")


# Adding game state to a pytorch state tensor:
def _normMiddlePeak(v):
    return 4/(1 + np.exp(-v)) - 2

def add_ball_state(state, phys):
    state.append(phys.location.x / libPreprocess.POS_X_MAX)
    state.append(phys.location.y / libPreprocess.POS_Y_MAX)
    state.append(phys.location.z / libPreprocess.POS_Z_MAX)

def add_player_state(state, phys, boost):
    # Note: Need to keep in sync with libPreprocess
    state.append(phys.location.x / libPreprocess.POS_X_MAX)
    state.append(phys.location.y / libPreprocess.POS_Y_MAX)
    state.append(phys.location.z / libPreprocess.POS_Z_MAX)
    state.append(_normMiddlePeak(phys.velocity.x / libPreprocess.VEL_X_MAX))
    state.append(_normMiddlePeak(phys.velocity.y / libPreprocess.VEL_Y_MAX))
    state.append(_normMiddlePeak(_normMiddlePeak(phys.velocity.z / libPreprocess.VEL_Z_MAX)))
    state.append(_normMiddlePeak(phys.rotation.pitch / (np.pi / 2)))
    state.append(phys.rotation.yaw / np.pi)
    state.append(_normMiddlePeak(phys.rotation.roll / np.pi))
    state.append(phys.angular_velocity.x / libPreprocess.ANG_VEL_MAX)
    state.append(phys.angular_velocity.y / libPreprocess.ANG_VEL_MAX)
    state.append(phys.angular_velocity.z / libPreprocess.ANG_VEL_MAX)
    state.append(boost / libPreprocess.BOOST_MAX)

# Use RLBot's debugging tool to print the action output above the car
def debugTxt(R, car, msg):
    R.begin_rendering()
    R.draw_string_3d(car.physics.location, 2, 2, msg, R.white())
    R.end_rendering()

# Sigmoid in numpy rather than pytorch
def npSigmoid(X):
    return 1/(1+np.exp(-X))

# Actual bot code starts here.
class MyBot(BaseAgent):
    def initialize_agent(self):
        print( "\n\n\n***\n\nCREATING BOT\n\n***\n\n")
        # This runs once before the bot starts up
        self.controller_state = SimpleControllerState()

        nPerTeam = 1
        self.stateDim = 3 + 2 * nPerTeam * (4*3 + 1)
        if BC_MODEL is not None:
            self.model = StoA_Model(self.stateDim, 3, DEVICE)
        else:
            self.model = SAtoV_Model(self.stateDim, 3, DEVICE)
        self.model.load(STATE_DICT)
        self.model.eval()
        print ("Model loaded!")

    # Create state tensor from RLBot state:
    def packet_to_state_tensor(self, packet: GameTickPacket):
        state = []
        add_ball_state(state, packet.game_ball.physics) # b_
        my_car = packet.game_cars[self.index]
        enemy_car = packet.game_cars[1 - self.index]
        add_player_state(state, my_car.physics, my_car.boost) # me_
        add_player_state(state, enemy_car.physics, enemy_car.boost) # e0_
        assert (len(state) == self.stateDim)
        return torch.from_numpy(np.array(state)).float().unsqueeze(0)

    # Convert game state to pytorch state to model actions to game actions
    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # Output actions
        aThrottle, aSteer, aBoost = 0, 0, False

        state_tensor = self.packet_to_state_tensor(packet)
        if BC_MODEL is not None:
            # BC, so use model to predict A from Q
            action_tensor = self.model(state_tensor).squeeze()
            action = action_tensor.detach().numpy()
            assert (len(action) == 3)

            # And convert A into two [-1, 1] and one boolean
            aAnalog = 2 * npSigmoid(action[:2]) - 1
            aThrottle, aSteer, aBoost = aAnalog[0], aAnalog[1], (action[2] > 0)

        else:
            # DQN, so find best action by using many to get the best Q:
            _, aMax = libRewards.bestQ(self.model, state_tensor, returnAction=True)
            (aThrottle, aSteer, aBoost) = aMax

        my_car = packet.game_cars[self.index]
        debugTxt(self.renderer, my_car, "(%.2f %.2f %.2f)" % (aThrottle, aSteer, float(aBoost)))

        # throttle, steer, boost
        self.controller_state.throttle = aThrottle
        self.controller_state.steer = aSteer
        self.controller_state.boost = aBoost
        return (self.controller_state)
