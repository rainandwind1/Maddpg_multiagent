import torch
from torch import nn, optim
import torch.nn.functional as F
from make_env import make_env
import numpy as np
import random
from model import Model_net
import argparse
import os



parser = argparse.ArgumentParser(description='Base init and setup for training or display')
parser.add_argument('-scen_name', type=str, help='Choose scenarios for training or display', default='simple_tag')



if __name__ == "__main__":

    args = parser.parse_args()
    env_name = args.scen_name
    LEARNING_RATE = 1e-3
    DONE_INTERVAL = 60
    total_step = 0
    epsilon = 0.9
    GAMMA = 0.98
    BATCH_SIZE = 64
    UPDATE_INTERVAL = 50
    SAVE_INTERVAL = 500
    MAX_EPOCH = 2000
    MEM_LEN = 30000
    render_flag = True
    epsilon_flag = True
    train_flag = False
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

    # 初始化模型
    agent_models = [Model_net(str(i), len(obs_ls[i]), env.action_space[i].n, MEM_LEN, LEARNING_RATE) for i in range(len(env.world.agents))]    
    agent_target_models = [Model_net('target_' + str(i), len(obs_ls[i]), env.action_space[i].n, MEM_LEN, LEARNING_RATE) for i in range(len(env.world.agents))]    
    
    for idx, model in enumerate(agent_target_models):
        model.load_state_dict(agent_models[idx].state_dict())


    for epo_i in range(MAX_EPOCH):
        epsilon = max(0.01, epsilon * 0.999)
        obs = env.reset()
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
                action_i, action_vec_i = model.choose_action(obs_ls[i], epsilon)
                action_vec_ls.append(action_vec_i)
                action_ls.append(action_i)
            
            obs_next_ls, reward_ls, done_ls, info_ls = env.step(np.array(action_vec_ls))
            score_ls += reward_ls
            done_flag_ls = []
            for d in done_ls:
                if (total_step % 60 and total_step > 0) or d:
                    done_flag_ls.append(0.)
                else:
                    done_flag_ls.append(1.) 

            # save transitions
            for n in range(len(agent_models)):
                agent_models[n].save_trans((obs_ls[n], action_ls[n], reward_ls[n], obs_next_ls[n], done_flag_ls[n]))
            
            obs_ls = obs_next_ls
            
            # train agent net
            if total_step > 2000:
                train_flag = True
                for i, model in enumerate(agent_models):
                    model.train(agent_target_models[i], GAMMA, BATCH_SIZE)
            
            # cover traget net
            if epo_i % 50 == 0:
                for idx, model in enumerate(agent_target_models):
                    model.load_state_dict(agent_models[idx].state_dict())

            # ******* 打印回合结果 ********
            if step == 59:
                print("Epoch:{}".format(epo_i + 1))
                for idx, score in enumerate(score_ls):
                    print("agent{} score:{} train_flag:{} epsilon:{}".format(idx, score, train_flag, epsilon))

        if epo_i+1 % SAVE_INTERVAL == 0:
            for idx, model in enumerate(agent_target_models):
                torch.save(model.state_dict(), param_path + 'agent' + str(idx) + '_' + str(epo_i + 1) + '.pkl')
        