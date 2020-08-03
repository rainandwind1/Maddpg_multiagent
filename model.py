import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import random
import collections

class Model_net(nn.Module):
    def __init__(self, name, input_size, output_size, mem_len, lr):
        super(Model_net, self).__init__()
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.q_net = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )
        self.mem = collections.deque(maxlen=mem_len)
        self.optimizer = optim.Adam(self.parameters(), lr = lr)


    
    def forward(self, inputs):
        q_value = self.q_net(inputs)
        return q_value

    def choose_action(self, inputs, epsilon):
        inputs = torch.tensor(inputs, dtype = torch.float32)
        q_val = self(inputs)
        seed = np.random.rand()
        if seed < epsilon:
            action_choice = random.sample(range(self.output_size), 1)[0]
        else:
            action_choice = torch.argmax(q_val).data.numpy()
        action_vec = np.zeros(self.output_size)
        action_vec[action_choice] = 1.
        return int(action_choice), action_vec

    def save_trans(self, transitions):
        self.mem.append(transitions)

    def sample_batch(self, batch_size):
        s_ls, a_ls, r_ls, s_next_ls, done_flag_ls = [], [], [], [], []
        trans_batch = random.sample(self.mem, batch_size)
        for trans in trans_batch:
            s, a, r, s_next, done_flag = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            done_flag_ls.append([done_flag])
        return torch.tensor(s_ls, dtype = torch.float32),\
                torch.tensor(a_ls, dtype = torch.int64),\
                torch.tensor(r_ls, dtype = torch.float32),\
                torch.tensor(s_next_ls, dtype = torch.float32),\
                torch.tensor(done_flag_ls, dtype = torch.float32)

    
    def train(self, target_model, gamma, batch_size, loss_list=None):
        s, a, r, s_next, done_flag = self.sample_batch(batch_size)
        q_val = self(s)
        a_index = torch.LongTensor(a)
        q_val = torch.gather(q_val, 1, a_index)
        q_target = r + gamma * torch.max(self(s_next).detach()) * done_flag
        td_error = q_target - q_val
        loss = F.mse_loss(q_target, q_val)
        if loss_list:
            loss_list.append(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



