

import torch
import torch.nn as nn

class GenericBlock(nn.Module):
    def __init__(self, backcast_length, forecast_length, units, theta_dim):
        super(GenericBlock, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.units = units 

        self.theta_dim = theta_dim
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)

        self.theta_b_fc = nn.Linear(units, theta_dim, bias=False)
        self.theta_f1_fc = nn.Linear(units, theta_dim, bias=False)
        #self.theta_f2_fc = nn.Linear(units, theta_dim, bias=False)

        self.backcast_fc = nn.Linear(theta_dim, backcast_length)
        self.forecast1_fc = nn.Linear(theta_dim, forecast_length)
        #self.forecast2_fc = nn.Linear(theta_dim, forecast_length)



    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        theta_b = self.theta_b_fc(x)
        theta_f1 = self.theta_f1_fc(x)
        #theta_f2 = self.theta_f2_fc(x)

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast1 = self.forecast1_fc(theta_f1)  # generic. 3.3.
        #forecast2 = self.forecast2_fc(theta_f2) # generic. 3.3.

        return backcast, forecast1  #, forecast2

      
class Trans(nn.Module):
    def __init__(self, backcast_length, forecast_length, theta_dim, units, stack_types, num_blocks):
        
        super(Trans, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stack_types = stack_types
        self.num_blocks = num_blocks
        self.theta_dim = theta_dim
        self.units = units


        self.stacks = nn.ModuleList()
        for stack_type in stack_types:
            stack = nn.ModuleList()
            for _ in range(num_blocks):
                stack.append(GenericBlock(backcast_length, forecast_length, units, theta_dim))
            self.stacks.append(stack)

    def forward(self, x):
        print(x)
        fc1 = torch.zeros(x.size(0), self.forecast_length)
        fc2 = torch.zeros(x.size(0),self.forecast_length)
        for stack in self.stacks:
            for block in stack:
                print("x shape",x)
                backcast, forecast = block(x)
                print("forecast ",forecast)
                x = x - backcast
                fc1 += forecast
        return backcast, fc1


    