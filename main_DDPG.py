import torch
from torch import nn, optim
import torch.nn.functional as F
from make_env import make_env
import numpy as np
import random
from model import Model_net, PPO, DDPG
import argparse
import os



parser = argparse.ArgumentParser(description='Base init and setup for training or display')
parser.add_argument('-scen_name', type=str, help='Choose scenarios for training or display', default='simple_tag')



if __name__ == "__main__":

    args = parser.parse_args()
    env_name = args.scen_name
    LEARNING_RATE = 1e-3
    total_step = 0
    GAMMA = 0.98
    toi = 0.01
    score_mem = 0
    batch_size = 128
    DONE_INTERVAL = 100 
    SAVE_INTERVAL = 500
    MAX_EPOCH = 30000
    MEM_LEN = 10000
    display = True

    if display:
        render_flag, LOAD_KEY, TRAIN_KEY = [True, True, False]
    else:
        render_flag, LOAD_KEY, TRAIN_KEY = [False, False, True]


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

    global_input_size = 0
    for cv in obs_ls:
        global_input_size += len(cv)
    for action_space in env.action_space:
        global_input_size += action_space.n
    # 初始化模型
    agent_models = [DDPG(str(i), len(obs_ls[i]),  env.action_space[i].n, global_input_size,  MEM_LEN, LEARNING_RATE) for i in range(len(env.world.agents))]
    target_models = [DDPG(str(i), len(obs_ls[i]),  env.action_space[i].n, global_input_size,  MEM_LEN, LEARNING_RATE) for i in range(len(env.world.agents))]
    for idx, model in enumerate(target_models):
        model.load_state_dict(agent_models[idx].state_dict())


    if LOAD_KEY:
        for idx, model in enumerate(agent_models):
            if idx == 3:
                check_point = torch.load('./param/DDPGagent3_18000.pkl')
            else:
                check_point = torch.load('./param/DDPGagent2_18000.pkl')
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
            
            # DDPG choose action
            for i, model in enumerate(agent_models):
                action_vec_i, _ = model.get_action(torch.tensor(obs_ls[i], dtype = torch.float32)) 
                action_vec_ls.append(action_vec_i.detach().numpy())
                action_ls.append(action_vec_i.detach().numpy())
                # action_ls += [0.25 for i in range(len(action_vec_i))]
            
            obs_next_ls, reward_ls, done_ls, info_ls = env.step(action_vec_ls)
            score_ls += reward_ls
            done_flag_ls = []
            for d in done_ls:
                if (total_step % 60 and total_step > 0) or d:
                    done_flag_ls.append(1.)
                else:
                    done_flag_ls.append(1.) 


            # save transitions
            total_s = []
            total_s_next = []
            for t in range(len(obs_ls)):
                total_s += list(obs_ls[t])
                total_s_next += list(obs_next_ls[t])

            for n in range(len(agent_models)):
                agent_models[n].save_trans((obs_ls[n], total_s, action_ls, reward_ls[n], total_s_next, done_flag_ls[n]))
            
            obs_ls = obs_next_ls
            
            # train agent net
            if TRAIN_KEY:
                if total_step > 10000:
                    if total_step % 5 == 0:
                        train_flag = True
                        for i, model in enumerate(agent_models):
                            model.train(agent_models, target_models, GAMMA, batch_size, toi)

            # ******* 打印回合结果 ********
            if step == DONE_INTERVAL - 1:
                if epo_i == 0:
                    score_mem = score_ls
                else:
                    score_ls = 0.99*score_mem + 0.01*score_ls
                    score_mem = score_ls
                print("Epoch:{}".format(epo_i + 1))
                for idx, score in enumerate(score_ls):
                    print("agent{} score:{} train_flag:{}".format(idx, score, train_flag))


        # 保存信息
        if (epo_i+1) % SAVE_INTERVAL == 0:
            print('save process')
            for idx, model in enumerate(agent_models):
                print('agent' + str(idx))
                torch.save(model.state_dict(), param_path + '/DDPGagent' + str(idx) + '_' + str(epo_i + 1) + '.pkl')
        