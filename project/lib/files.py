# Code for dealing with replay files.


# Load the library locally
import os
import sys
sys.path.append('D:/projects/carball')

import carball
from carball.controls.controls import ControlsCreator
from carball.json_parser import Game


# Given the name of a replay in the replays folders,
#   load and preprocess with carball
def parseReplayToGameData(replayID):
    replayPath = os.path.join("replays", "%s.replay" % replayID)
    print ("Loading...\n\t%s" % replayPath)
    json = carball.decompile_replay(replayPath)
    game = Game()
    game.initialize(loaded_json=json)

    # Parses action-related properties into their actual actions
    ControlsCreator().get_controls(game)
    return game


# Given a loaded game, trim extra rows, and print some details for debugging
def cleanAndDisplayGameData(gameData):
    nPlayers = len(gameData.players)
    assert nPlayers == 2, "Only 1v1 modes supported, this has %d players" % nPlayers
    print ("%d players loaded!" % nPlayers)

    # For now, rename to 'Expert' and 'Bot'
    expertIdx = 0 if gameData.players[0].name == "padster" else 1
    gameData.players[  expertIdx].name = 'expert'
    gameData.players[1-expertIdx].name = 'bot'

    orangeIdx = [i for i, p in enumerate(gameData.players) if p.is_orange]
    blueIdx = [i for i, p in enumerate(gameData.players) if not p.is_orange]

    print ("\nOrange team:")
    for i in orangeIdx:
        print ("\t%s" % gameData.players[i].name)
    print ("\nBlue team:")
    for i in blueIdx:
        print ("\t%s" % gameData.players[i].name)

    nTimepoints = len(gameData.ball)
    for p in gameData.players:
        assert len(p.data) >= nTimepoints - 20, \
            "Players (%d) need the same number of time points (%d), no leaves allowed" % (len(p.data), nTimepoints)
        nTimepoints = min(nTimepoints, len(p.data))
    print ("\n%d data points acquired" % nTimepoints)

    # if not all the same, trim
    gameData.ball = gameData.ball.tail(nTimepoints)
    for p in gameData.players:
        p.data = p.data.tail(nTimepoints)

    print ("====\n\n")
