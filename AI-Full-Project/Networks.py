import torch
import torch.nn.functional as F
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self, input_size=6, hidden_size1=64, hidden_size2=32, output_size=1, dropout_prob=0.5):
        # STUDENTS: __init__() must initialize nn.Module and define your network's
        # custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # STUDENTS: forward() must complete a single forward pass through your network
        # and return the output which should be a tensor

        x = F.relu(self.fc1(input))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

    def evaluate(self, model, test_loader, loss_function):
        # STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
        # mind that we do not need to keep track of any gradients while evaluating the
        # model. loss_function will be a PyTorch loss function which takes as argument the model's
        # output and the desired output.
        total_loss = 0.0
        for idx, sample in enumerate(test_loader):
            input = sample['input']
            target_output = sample['label']
            network_output = model(input)
            # print(network_output,"\n ------------------------------------------------------------------")
            loss = loss_function(network_output, target_output)
            total_loss += loss.item()

        return total_loss / len(test_loader)


