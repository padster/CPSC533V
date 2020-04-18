# Define artificial rewards for each (s, a) pair.
#     Used to reward agents even before they can score a goal.

import numpy as np
import torch

DEFAULT_COEF = {
    # S
    'nearBall'  : 3.0,
    'nearGoal'  : 0.3,
    'hasBoost'  : 0.1,
    'behindBall': 0.2,
    # A
    'forwards'  : 0.6,
    'straight'  : 0.6,
    'saveBoost' : 0.1,
    # Combine
    'state'     : 0.1,
    'action'    : 0.2,
    # Constant to pad upwards if needed
    'const'     : 0.0,
}

def _pos(s, key, typ='pos'):
    return np.array([s['%s_%s_%s' % (key, typ, d)] for d in ['x', 'y', 'z']])

def _dist3D(v1, v2):
    return np.sqrt(np.sum(np.square((v1 - v2))))

# s = dataframe with all named state values.
# a = list of [throttle, steer, boost]
def artificialReward(s, a, coefOverrides={}):
    coef = {}
    coef.update(DEFAULT_COEF)
    coef.update(coefOverrides)

    ballAt = _pos(s, 'b')
    meAt = _pos(s, 'me')

    nearBall = 3.0 - 2 * _dist3D(ballAt, meAt)
    nearGoal = (ballAt[1] + 1) / 2
    hasBoost = s['me_boost']
    behindBall = 1 if ballAt[1] > meAt[1] else 0
    stateReward = \
              coef['nearBall'] * nearBall \
            + coef['nearGoal'] * nearGoal \
            + coef['hasBoost']  * hasBoost \
            + coef['behindBall'] * behindBall

    goForwards = (a[0] + 1.0) / 2
    stayStraight = 1.0 - np.abs(a[1])
    noBoost = 1 - float(a[2])
    actionReward = \
              coef['forwards'] * goForwards \
            + coef['straight'] * stayStraight \
            + coef['saveBoost'] * noBoost

    return coef['state'] * stateReward + coef['action'] * actionReward + coef['const']

# Continuous 3D action space, so we can't find the maximum easily.
# Instead, sample over a small subset of actions
#  * Throttle takes values [100% back, nothing, 100% forwards]
#  * Steer takes values [100% left, nothing, 100% right]
def bestQ(model, state, nT=3, nS=3, returnAction=False):
    qMax = None
    if returnAction:
        # Note: by default, this is done in batch, with multiple states.
        # During real-world interaction, we do only one at a time.
        assert state.shape[0] == 1, "Need single state for single action"
        aMax = (0, 0, 0)

    # Try all action combinations:
    for tValue in range(nT):
        throttle = 2 * tValue / (nT - 1) - 1 # [-1, 1]
        for sValue in range(nS):
            steer = 2 * sValue / (nS - 1) - 1 # [-1, 1]
            for boost in [False, True]:
                aArray = np.repeat(np.array([[tValue, sValue, boost]]), state.shape[0], axis=0)
                sa = torch.cat((state, torch.from_numpy(aArray).float()), dim=1)
                q = model(sa).detach().numpy()
                if returnAction:
                    q = q[0]
                    if qMax is None or qMax < q:
                        qMax = q
                        aMax = (throttle, steer, boost)
                else:
                    if qMax is None:
                        qMax = q
                    else:
                        # no need to store action, so element-wise max in parallel.
                        qMax = np.maximum(qMax, q)

    if returnAction:
        return qMax, aMax
    else:
        return qMax
