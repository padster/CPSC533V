# Neural network model for function f : S -> A
#   Used for behaviouralCloning

import os
import torch
import torch.nn as nn

class StoA_Model(nn.Module):
    def __init__(self, state_size, action_size, device):
        super(StoA_Model, self).__init__()

        # |S| -> 32 -> 8 -> |A|
        self.dimIn         = state_size
        self.dimH1         = 32
        self.dimH2         = 8
        self.dimOut        = action_size

        self.model = nn.Sequential(
            nn.Linear(self.dimIn, self.dimH1),
            nn.BatchNorm1d(self.dimH1),
            nn.ReLU(),
            nn.Linear(self.dimH1, self.dimH2),
            nn.BatchNorm1d(self.dimH2),
            nn.ReLU(),
            nn.Linear(self.dimH2, self.dimOut),
        )
        self.model.to(device)

    def forward(self, x):
        return self.model(x.float())

    def save(self, modelID):
        path = os.path.join("models", "%s.pt" % modelID)
        torch.save(self.state_dict(), path)
        print('Saved model!\n\t%s' % path)

    def load(self, state=None, path=None):
        if state is None:
            state = torch.load(path)
        self.load_state_dict(state)
        print('Loaded model!')
