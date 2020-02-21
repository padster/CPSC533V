import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(MyModel, self).__init__()
        
        # Start with a super simple multi-layer perceptron, one hidden layer 
        self.dimH   = 10 # hidden layer has 10 dimensions
        self.dimIn  = state_size
        self.dimOut = action_size
 
        self.model = torch.nn.Sequential(
            nn.Linear(self.dimIn, self.dimH),
            nn.ReLU(),
            nn.Linear(self.dimH, self.dimOut)
        )

    def forward(self, x):
        # input data type needs to be converted to float
        return self.model(x.float())

    def select_action(self, state):
        self.eval()
        x = self.forward(state)
        self.train()
        return x.max(1)[1].view(1, 1).to(torch.long)
