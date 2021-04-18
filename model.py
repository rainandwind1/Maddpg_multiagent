import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import random
import collections
import copy


# DQL
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


# PPO
class PPO(nn.Module):
    def __init__(self, name, input_size, global_input_size, output_size, lr):
        super(PPO, self).__init__()
        self.name = name
        self.input_size = input_size
        self.global_size = global_input_size
        self.output_size = output_size
        self.actor = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.global_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.mem = []

    def forward(self, inputs, target=None, global_inputs=None):
        policy_prob = self.actor(inputs)
        if global_inputs is not None:
            critic_val = self.critic(global_inputs)
        if not target:
            return policy_prob, critic_val
        elif target == 'get critic':
            return critic_val
        elif target == 'get policy':
            return F.softmax(policy_prob, -1)


    def choose_action(self, inputs):
        inputs = torch.tensor(inputs, dtype = torch.float32)
        policy_prob = self(inputs, 'get policy')  
        action_sample = torch.distributions.Categorical(policy_prob)
        action_choice = int(action_sample.sample().numpy())
        action_vec = np.zeros(self.output_size)
        action_vec[action_choice] = 1
        return action_choice ,action_vec

    def save_trans(self, transitions):
        self.mem.append(transitions)


    def package_trans(self):
        s_ls, global_s_ls, a_ls, r_ls, s_next_ls, global_next_ls, a_prob_ls, done_flag_ls = [], [], [], [], [], [], [], []
        for trans in self.mem:
            s, global_s, a, r, s_next, global_next, a_prob, done_flag = trans
            s_ls.append(s)
            global_s_ls.append(global_s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            global_next_ls.append(global_next)
            a_prob_ls.append([a_prob])
            done_flag_ls.append([done_flag])
        return torch.tensor(s_ls, dtype = torch.float32),\
                torch.tensor(global_s_ls, dtype = torch.float32),\
                torch.tensor(a_ls, dtype = torch.int64),\
                torch.tensor(r_ls, dtype = torch.float32),\
                torch.tensor(s_next_ls, dtype = torch.float32),\
                torch.tensor(global_next_ls, dtype = torch.float32),\
                torch.tensor(a_prob_ls, dtype = torch.float32),\
                torch.tensor(done_flag_ls, dtype = torch.float32)


    def train(self, k_epo, gamma, epi_clip, Lambda, loss_list=None):
        s, global_s, a, r, s_next, global_next, a_prob, done_flag = self.package_trans()
        for epo in range(k_epo):
            td_target = r + gamma * self(s_next, 'get critic', global_next) * done_flag
            td_error = td_target - self(s, 'get critic', global_s)
            td_error = td_error.detach().numpy()

            advantage_ls = []
            advantage = 0.
            for error in td_error[::-1]:
                advantage = gamma * Lambda * advantage + error[0]
                advantage_ls.append([advantage])
            advantage_ls.reverse()
            advantage = torch.tensor(advantage, dtype = torch.float32)

            policy = self(s, 'get policy')
            policy = policy.gather(1, a)
            ratio = torch.exp(torch.log(policy) - torch.log(a_prob))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - epi_clip, 1 + epi_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self(s, 'get critic', global_s), td_target.detach())

            self.optimizer.zero_grad()
            loss.backward(retain_graph = True)
            self.optimizer.step()
        self.mem = []


class DDPG(nn.Module):
    def __init__(self, name, input_size, output_size, total_size, mem_len, lr):
        super(DDPG, self).__init__()
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.total_size = total_size
        self.actor = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.total_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.buffer = collections.deque(maxlen=mem_len)    # (s,a,r,s',done_flag)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr = lr)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr = lr)



    def get_action(self, obs_i):
        action_op = self.actor(obs_i)
        noise = torch.rand_like(action_op)
        action = F.softmax(action_op - torch.log(-torch.log(noise)), -1)
        return action, action_op

    def get_critic(self, x):
        critic = self.critic(x)
        return critic

    def save_trans(self, transitions):
        self.buffer.append(transitions)
    
    def sample_trans(self, batch_size):
        cur_s_ls, s_ls, a_ls, r_ls, s_next_ls, done_flag_ls = [], [], [], [], [], []
        trans_batch = random.sample(self.buffer, batch_size)
        for trans in trans_batch:
            cur_s, s, a, r, s_next, done_flag = trans
            cur_s_ls.append(cur_s)
            s_ls.append(s)
            a_ls.append(a)
            r_ls.append([r])
            s_next_ls.append(s_next)
            done_flag_ls.append([done_flag])
        return torch.tensor(cur_s_ls, dtype = torch.float32),\
                torch.tensor(s_ls, dtype = torch.float32),\
                a_ls,\
                torch.tensor(r_ls, dtype = torch.float32),\
                torch.tensor(s_next_ls, dtype = torch.float32),\
                torch.tensor(done_flag_ls, dtype = torch.float32)


    def train(self, models, target_model_ls, gamma, batch_size, toi, loss_list=None):
        cur_s, s, a, r, s_next, done_flag = self.sample_trans(batch_size)
        
        # 更新 critic
        init = 0
        a_ = torch.tensor([])
        for idx, model in enumerate(target_model_ls):
            action_i, _ = model.get_action(s_next[:,init:init + model.input_size])
            a_ = torch.cat((a_, action_i.detach()), -1)
            init += model.input_size
        target = r + gamma * target_model_ls[int(self.name)].get_critic(torch.cat((s_next, a_), -1)) * done_flag

        a_copy = copy.deepcopy(a)
        for idx, a_i in enumerate(a_copy):
            a_copy[idx] = torch.tensor(np.concatenate(a_i, -1), dtype = torch.float32)
        a_total = torch.stack(a_copy)
            
        loss_critic = ((target.detach() - self.get_critic(torch.cat((s, a_total), -1))) ** 2).mean()

        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.6, norm_type=2)
        self.optimizer_critic.step()


        # 更新actor
        prob, raw_op = self.get_action(cur_s)
        a_r = [torch.tensor(np.vstack(a[i][n] for i in range(batch_size)), dtype = torch.float32) for n in range(len(a[0]))]
        a_c = copy.deepcopy(a_r)
        a_r[int(self.name)] = prob
        a = torch.cat(a_r, 1)
        
        q_val = self.get_critic(torch.cat((s, a), -1))
        
        loss_actor = -q_val.mean() + 1e-3*(raw_op**2).mean()
                
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.6, norm_type=2)
        self.optimizer_actor.step()
        
        


        # 参数跟踪
        for param_t, param_o in zip(target_model_ls[int(self.name)].parameters(), self.parameters()):
            param_t.data = toi*param_o.data + (1-toi)*param_t.data 

    
    










