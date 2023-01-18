import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers.gate_layer import GateLayer, GateLayer_mask


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.gate1 = GateLayer(200,200,[1, -1])

        self.fc2 = nn.Linear(200, 200)
        self.gate2 = GateLayer(200,200,[1, -1])

        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.gate1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.gate2(x)

        x = self.fc3(x)
        return torch.sigmoid(x)
    
    
class Net_mask(nn.Module):
    def __init__(self, input_size, units):
        super(Net_mask, self).__init__()
        self.channels = 200
        self.fc1 = nn.Linear(input_size, self.channels)

        self.gate1 = GateLayer(self.channels,self.channels,[1, -1])

        self.fc2 = nn.Linear(self.channels, self.channels)
        self.gate2 = GateLayer(self.channels,self.channels,[1, -1])
        
        self.fc3 = nn.Linear(self.channels, 1)
        self.units = units

#         self.mask = torch.ones(200).cuda()
#         self.mask[1] = 0

    def forward(self, x):
        self.gate_mask = GateLayer_mask(self.channels,self.channels,[1, -1], self.units)
        self.gate_mask = self.gate_mask.cuda()
        x = self.fc1(x)
        x = F.relu(x)
#         x = self.gate_mask(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.gate_mask(x)
#         x = self.gate2(x)
#         print(x.size())
#         x = x.mul(self.mask)
        
        x = self.fc3(x)
        return torch.sigmoid(x)

class NetRaw(nn.Module):
    def __init__(self, input_size):
        super(NetRaw, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)

        self.fc2 = nn.Linear(200, 200)

        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return torch.sigmoid(x)

class NetRawAdv(NetRaw):
    def forward_fair(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x1 = F.relu(x)

        x = self.fc3(x1)
        return torch.sigmoid(x),x1
    
class NetSmall(nn.Module):
    def __init__(self, input_size):
        super(NetSmall, self).__init__()
        self.fc1 = nn.Linear(input_size, 20)

        self.fc2 = nn.Linear(20, 20)

        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return torch.sigmoid(x)
    
class NetSmall_test(nn.Module):
    def __init__(self, input_size):
        super(NetSmall_test, self).__init__()
        
        channels = 20

        self.fc1 = nn.Linear(input_size, channels)

        self.fc2 = nn.Linear(channels, channels)

        self.fc3 = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return torch.sigmoid(x)
    
class NetRawAdv(NetRaw):
    def forward_fair(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x1 = F.relu(x)

        x = self.fc3(x1)
        return torch.sigmoid(x),x1



def freezen_gate(model, freeze_=True):
    for name, child in model.named_children():
        if name[:4]=="gate":
            for param in child.parameters():
                param.requires_grad = not freeze_
    
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# class myNet(nn.Module):
#     def __init__(self, input_size):
#         super(myNet, self).__init__()
#         self.fc1 = nn.Linear(input_size, 200)
#         self.fc2 = nn.Linear(200, 200)
#         self.fc3 = nn.Linear(200, 1)
#         self.fc_g = nn.Linear(200, 1)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         features = F.relu(x)
#         y_target = self.fc3(features)
#         y_gender = self.fc_g(features)
#
#         return features, torch.sigmoid(y_target), torch.sigmoid(y_gender)


if __name__=="__main__":
    input_size=120 
    model = NetRaw(input_size=input_size)
    
    
    print (model )
    print ("raw param_c",count_parameters(model) )
    
    # freezen_gate(model, freeze_=True)
    # print ("after frozen param_c",count_parameters(model) )
    #
    # freezen_gate(model, freeze_=False)
    # print ("release frozen param_c",count_parameters(model) )
    