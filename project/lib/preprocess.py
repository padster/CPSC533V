# Preprocess replay data:
#   1) Filter to only the values we want, and split to state vs action
#   2) Impute missing values
#   3) Normalize each to desired ranges

import numpy as np
import pandas as pd
import random
import torch

import lib.rewards as libRewards

# Normalization constants to convert units to [-1, 1]
POS_X_MAX, POS_Y_MAX, POS_Z_MAX = 4000, 5500, 2000
VEL_X_MAX, VEL_Y_MAX, VEL_Z_MAX = 22000, 22000, 16000
ANG_VEL_MAX = 6000
BOOST_MAX = 256


# Filter of keys we want for player state, ball state, and player actions
PLAYER_STATE_KEYS = (
    ['pos_x', 'pos_y', 'pos_z'] +              # x \
    ['vel_x', 'vel_y', 'vel_z'] +              # dx/xt \
    ['rot_x', 'rot_y', 'rot_z'] +              # q \
    ['ang_vel_x', 'ang_vel_y', 'ang_vel_z'] +  # dq/dt \
    ['boost']
)
BALL_STATE_KEYS = (
    ['pos_x', 'pos_y', 'pos_z']
)
PLAYER_ANALOG_ACTION_KEYS = ['throttle', 'steer']
PLAYER_DIGITAL_ACTION_KEYS = ['boost']
PLAYER_ACTION_KEYS = PLAYER_ANALOG_ACTION_KEYS + PLAYER_DIGITAL_ACTION_KEYS

# For those with a large middle peak, transfrom to be more precise at the centre.
def _normMiddlePeak(v):
    return 4/(1 + np.exp(-v)) - 2

# Note: Missing values come from start of game or after explosion, in which case,
# should be an okay approximation to impute by forward then backfilling.
def _imputePlayerState(stateDF):
    return stateDF.fillna(method='ffill').fillna(method='bfill')

# Normalize each value of the player state.
def _normPlayerState(stateDF):
    normedDF = pd.DataFrame()
    normedDF['pos_x'] = stateDF['pos_x'] / POS_X_MAX
    normedDF['pos_y'] = stateDF['pos_y'] / POS_Y_MAX
    normedDF['pos_z'] = stateDF['pos_z'] / POS_Z_MAX
    normedDF['vel_x'] = _normMiddlePeak(stateDF['vel_x'] / VEL_X_MAX)
    normedDF['vel_y'] = _normMiddlePeak(stateDF['vel_y'] / VEL_Y_MAX)
    normedDF['vel_z'] = _normMiddlePeak(_normMiddlePeak(stateDF['vel_z'] / VEL_Z_MAX))
    normedDF['rot_x'] = _normMiddlePeak(stateDF['rot_x'] / (np.pi / 2))
    normedDF['rot_y'] = stateDF['rot_y'] / np.pi
    normedDF['rot_z'] = _normMiddlePeak(stateDF['rot_z'] / np.pi)
    normedDF['ang_vel_x'] = stateDF['ang_vel_x'] / ANG_VEL_MAX
    normedDF['ang_vel_y'] = stateDF['ang_vel_y'] / ANG_VEL_MAX
    normedDF['ang_vel_z'] = stateDF['ang_vel_z'] / ANG_VEL_MAX
    normedDF['boost'] = stateDF['boost'] / BOOST_MAX
    assert normedDF.shape[1] == stateDF.shape[1], "Columns are missing normalization"
    return normedDF

# Fill missing actions by replacing with inaction
def _imputePlayerActions(actionDF):
    actionDF = actionDF.fillna({
        'throttle': 0,
        'steer': 0,
        'boost': False,
    })
    return actionDF

# Normalize action values - NOTE: already normed by ControlsCreator on load.
def _normPlayerActions(actionDF):
    normedDF = actionDF.copy()
    assert normedDF.shape[1] == actionDF.shape[1], "Columns are missing normalization"
    return normedDF

# Normalize ball state.
def _normBallState(stateDF):
    normedDF = pd.DataFrame()
    normedDF['pos_x'] = stateDF['pos_x'] / POS_X_MAX
    normedDF['pos_y'] = stateDF['pos_y'] / POS_Y_MAX
    normedDF['pos_z'] = stateDF['pos_z'] / POS_Z_MAX
    assert normedDF.shape[1] == stateDF.shape[1], "Columns are missing normalization"
    return normedDF

# Filter, impute and normalize player state values
def cleanPlayerStates(playerDF):
    stateDF = playerDF[PLAYER_STATE_KEYS]
    stateDF = _imputePlayerState(stateDF)
    stateDF = _normPlayerState(stateDF)
    return stateDF

# Filter, impute and normalize player action values
def cleanPlayerActions(controlDF):
    actionDF = controlDF[PLAYER_ACTION_KEYS]
    actionDF = _imputePlayerActions(actionDF)
    actionDF = _normPlayerActions(actionDF)
    return actionDF

# Filter, impute and normalize ball state values
def cleanBallStates(ballDF):
    stateDF = ballDF[BALL_STATE_KEYS] * 1.0 # force int -> float
    stateDF = _normBallState(stateDF)
    return stateDF

###
# Per-player state merging
##

# Utility to copy a column from one DF to another, and add a prefix to its name
def _copyIntoPrefixed(toDF, fromDF, prefix):
    for column in list(fromDF):
        toDF[prefix + column] = fromDF[column]

# Given a player, list the IDs of the other players on their team,
#    then the IDs of the players on the other team.
def _teamBreakdown(gameData, playerIdx):
    nPlayers = len(gameData.players)
    orangeIdx = [i for i, p in enumerate(gameData.players) if p.is_orange]
    blueIdx = [i for i, p in enumerate(gameData.players) if not p.is_orange]

    isOrange = gameData.players[playerIdx].is_orange
    teamIdx = [i for i in range(nPlayers) if i != playerIdx and gameData.players[i].is_orange == isOrange]
    enemyIdx = blueIdx if isOrange else orangeIdx
    return teamIdx, enemyIdx

# Given a game, and a particular player, merge ball and all player states
#     into a single dataframe correctly named.
# For orange players, mirror along the Y axis to fake them being on blue too.
def stateAndActionsForPlayer(game, playerIdx):
    gameData = game['data']
    stateDF = pd.DataFrame(index=game['ballStates'].index)
    _copyIntoPrefixed(stateDF, game['ballStates'], "b_")
    _copyIntoPrefixed(stateDF, game['playerStates'][playerIdx], "me_")

    teamIdx, enemyIdx = _teamBreakdown(gameData, playerIdx)
    for i, idx in enumerate(teamIdx):
        _copyIntoPrefixed(stateDF, game['playerStates'][idx], "t%d_" % i)
    for i, idx in enumerate(enemyIdx):
        _copyIntoPrefixed(stateDF, game['playerStates'][idx], "e%d_" % i)
    assert max(stateDF.isna().sum()) == 0, "NA state values not successfully removed?"

    actionDF = game['playerActions'][playerIdx]
    assert max(actionDF.isna().sum()) == 0, "NA action values not successfully removed?"

    endOverhang = actionDF.shape[0] - stateDF.shape[0]
    actionDF = actionDF[endOverhang:]
    assert actionDF.shape[0] == stateDF.shape[0], "Matching number of timepoints"

    # If orange, flip the entire game along the middle of the pitch:
    # This allows us to use the same logic for orange and blue players,
    #   despite them going in opposite directions
    if gameData.players[playerIdx].is_orange:
        oldStateDF = stateDF.copy()
        for yColumn in list(oldStateDF):
            if yColumn.endswith("_y"):
                stateDF[yColumn] = -1 * oldStateDF[yColumn]

    return stateDF, actionDF

###
# Converting states and actions into batches, for training
##
def dataToBatches(states, actions, batchSz, splitActions=False, includeRewards=False, seed=1234):
    random.seed(seed)
    nRows = states.shape[0]
    stateSz = states.shape[1]
    actionAnalogSz = len(PLAYER_ANALOG_ACTION_KEYS)
    actionDigitalSz = len(PLAYER_DIGITAL_ACTION_KEYS)
    actionSz = actionAnalogSz + actionDigitalSz

    print ("%d rows, %d state dim, %d action dim, into %d batches of size %d" % (
        nRows, stateSz, actionSz, (nRows + batchSz - 1) // batchSz, batchSz
    ))

    tOrder = list(range(0, nRows - 1))
    random.shuffle(tOrder)

    dataBatches = []
    for i in range(0, len(tOrder), batchSz):
        nInBatch = min(batchSz, len(tOrder) - i)

        s = np.zeros((nInBatch, stateSz))
        if splitActions:
            aAnalog = np.zeros((nInBatch, actionAnalogSz))
            aDigital = np.zeros((nInBatch, actionDigitalSz))
        else:
            a = np.zeros((nInBatch, actionSz))

        r = None
        if includeRewards:
            assert not splitActions, "Having both artificial rewards and split actions is unsupported - pick one."
            r = np.zeros((nInBatch))
            sPrime = np.zeros((nInBatch, stateSz))

        for j in range(nInBatch):
            t = tOrder[i + j]

            s[j, :] = states.iloc[t, :].values
            if splitActions:
                aAnalog[j, :] = actions.iloc[t, :actionAnalogSz].values.astype(np.float32)
                aDigital[j, :] = actions.iloc[t, actionAnalogSz:].values.astype(np.bool)
            else:
                a[j, :] = actions.iloc[t, :].values
                if includeRewards:
                    r[j] = libRewards.artificialReward(states.iloc[t, :], a[j, :])
                    sPrime[j, :] = states.iloc[t+1, :].values

        if splitActions:
            dataBatches.append({
                's': torch.from_numpy(s).float(),
                'aAnalog': torch.from_numpy(aAnalog).float(),
                'aDigital': torch.from_numpy(aDigital).float()
            })
        else:
            dataBatches.append({
                's': torch.from_numpy(s).float(),
                'a': torch.from_numpy(a).float(),
            })
            if includeRewards:
                dataBatches[-1]['r'] = torch.from_numpy(r).float()
                dataBatches[-1]['sPrime'] = torch.from_numpy(sPrime).float()

    if splitActions:
        return dataBatches, stateSz, actionAnalogSz, actionDigitalSz
    else:
        return dataBatches, stateSz, actionSz
