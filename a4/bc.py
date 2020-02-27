import gym
import torch
from eval_policy import eval_policy, device
from model import MyModel
from dataset import Dataset

# Turn off logging for now
gym.logger.set_level(40)

BATCH_SIZE = 64
TOTAL_EPOCHS = 100
LEARNING_RATE = 10e-4
PRINT_INTERVAL = 500
TEST_INTERVAL = 2

ENV_NAME = 'CartPole-v0'

dataset = Dataset(data_path="{}_dataset.pkl".format(ENV_NAME))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)

env = gym.make(ENV_NAME)

# State is 4D, action has two options
model = MyModel(4, 2)

def train_behavioral_cloning():
    
    # Adam optimizer usually a good default. TODO: pick learning rate:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Decision is binary 0 or 1, so cross entropy loss should work well:
    loss_function = torch.nn.CrossEntropyLoss().to(device)

    gradient_steps = 0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        for iteration, data in enumerate(dataloader):
            data = {k: v.to(device) for k, v in data.items()}

            output = model(data['state'])

            loss = loss_function(output, data["action"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if gradient_steps % PRINT_INTERVAL == 0:
                print('[epoch {:4d}/{}] [iter {:7d}] [loss {:.5f}]'
                    .format(epoch, TOTAL_EPOCHS, gradient_steps, loss.item()))
            
            gradient_steps += 1

        if epoch % TEST_INTERVAL == 0:
            score = eval_policy(policy=model, env=ENV_NAME)
            print('[Test on environment] [epoch {}/{}] [score {:.2f}]'
                .format(epoch, TOTAL_EPOCHS, score))

    model_name = "behavioral_cloning_{}.pt".format(ENV_NAME)
    print('Saving model as {}'.format(model_name))
    torch.save(model.state_dict(), model_name)


if __name__ == "__main__":
    train_behavioral_cloning()
