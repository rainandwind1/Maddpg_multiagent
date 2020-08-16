import torch
from torch import nn, optim
import torch.nn.functional as F
from make_env import make_env
import numpy as np
import random
from model import Model_net, PPO
import argparse
import os



parser = argparse.ArgumentParser(description='Base init and setup for training or display')
parser.add_argument('-scen_name', type=str, help='Choose scenarios for training or display', default='simple_tag')



if __name__ == "__main__":

    args = parser.parse_args()
    env_name = args.scen_name
    LEARNING_RATE = 1e-4
    total_step = 0
    epsilon = 0.1
    GAMMA = 0.98
    K_epo = 2
    Lambda = 0.95
    DONE_INTERVAL = 100 
    SAVE_INTERVAL = 60
    MAX_EPOCH = 2000
    render_flag = True
    train_flag = False
    LOAD_KEY = False
    TRAIN_KEY = True
    param_path = '.\param'
    log_path = '.\info'
    if not os.path.exists(param_path):
        print("创建参数文件夹")
        os.makedirs(param_path)
    if not os.path.exists(log_path):
        print("创建日志文件夹")
        os.makedirs(log_path)
    env = make_env(env_name)
    obs_ls = env.reset()        # 初始化状态

    global_input_size = 0
    for cv in obs_ls:
        global_input_size += len(cv)
    # 初始化模型
    agent_models = [PPO(str(i), len(obs_ls[i]),  global_input_size, env.action_space[i].n, LEARNING_RATE) for i in range(len(env.world.agents))]

    if LOAD_KEY:
        for idx, model in enumerate(agent_models):
            if idx == 3:
                check_point = torch.load('./param/PPOagent3_1980.pkl')
            else:
                check_point = torch.load('./param/PPOagent2_1020.pkl')
            model.load_state_dict(check_point)

    for epo_i in range(MAX_EPOCH):
        obs_ls = env.reset()
        score_ls = np.array([0. for _ in range(env.n)]) # n个代理的回合得分表
        for step in range(DONE_INTERVAL):
            total_step += 1
            if render_flag:
                env.render()
            # 动作序列
            action_ls = []
            action_vec_ls = []

            # 随机动作
            # for i, agent in enumerate(env.world.agents):
            #     agent_action_space = env.action_space[i]
            #     action = agent_action_space.sample()
            #     action_vec = np.zeros(agent_action_space.n)
            #     action_vec[action] = 1
            #     action_ls.append(action_vec)
            
            # IQL choose action
            for i, model in enumerate(agent_models):
                action_i, action_vec_i = model.choose_action(obs_ls[i])
                action_vec_ls.append(action_vec_i)
                action_ls.append(action_i)
            
            obs_next_ls, reward_ls, done_ls, info_ls = env.step(np.array(action_vec_ls))
            score_ls += reward_ls
            done_flag_ls = []
            for d in done_ls:
                if (total_step % 60 and total_step > 0) or d:
                    done_flag_ls.append(1.)
                else:
                    done_flag_ls.append(1.) 

            a_prob_ls = []
            for n in range(len(obs_ls)):
                action_p = agent_models[n](torch.tensor(obs_ls[n], dtype = torch.float32), 'get policy')
                a_prob_ls.append(action_p[action_ls[n]].detach().item())


            # save transitions
            total_s = []
            total_s_next = []
            for t in range(len(obs_ls)):
                total_s += list(obs_ls[t])
                total_s_next += list(obs_next_ls[t])

            for n in range(len(agent_models)):
                agent_models[n].save_trans((obs_ls[n], total_s, action_ls[n], reward_ls[n], obs_next_ls[n], total_s_next, a_prob_ls[n], done_flag_ls[n]))
            
            obs_ls = obs_next_ls
            
            # train agent net
            train_flag = True
            for i, model in enumerate(agent_models):
                model.train(K_epo, GAMMA, epsilon, Lambda)
            
            # ******* 打印回合结果 ********
            if step == DONE_INTERVAL - 1:
                print("Epoch:{}".format(epo_i + 1))
                for idx, score in enumerate(score_ls):
                    print("agent{} score:{} train_flag:{} epsilon:{}".format(idx, score, train_flag, epsilon))


        # 保存信息
        if (epo_i+1) % SAVE_INTERVAL == 0:
            print('save process')
            for idx, model in enumerate(agent_models):
                print('agent' + str(idx))
                torch.save(model.state_dict(), param_path + '/PPOagent' + str(idx) + '_' + str(epo_i + 1) + '.pkl')
        