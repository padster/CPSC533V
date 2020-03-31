import os
import sys
sys.path.append('D:/projects/carball')

import carball

def replayToJSON(replayID):
  replayPath = os.path.join('replays', "%s.replay" % replayID)
  jsonPath = os.path.join('json', "%s.json" % replayID)
  return carball.analyze_replay_file(replayPath, output_path=jsonPath, overwrite=True)


gameProto = replayToJSON("191A953940B2B3282A6AD6B5D35911F5").get_protobuf_data()
print (gameProto)
