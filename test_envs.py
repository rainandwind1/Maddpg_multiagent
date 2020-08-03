from make_env import make_env
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Base init and setup for training or display')
    parser.add_argument('-scen_name', type=str, help='Choose scenarios for training or display')
    args = parser.parse_args()
    name = args.scen_name
    print('Choose scen:{}'.format(name))

    MAX_EPOCH = 1000
    render_flag = True
    env = make_env('simple_tag')
    obs_ls = env.reset()
    for i in range(MAX_EPOCH):
        obs = env.reset()
        score_ls = np.array([0. for _ in range(env.n)]) # n个代理的回合得分表
        while True:
            if render_flag:
                env.render()
            
            # 动作序列
            action_ls = []
            for i, agent in enumerate(env.world.agents):
                agent_action_space = env.action_space[i]
                action = agent_action_space.sample()
                action_vec = np.zeros(agent_action_space.n)
                action_vec[action] = 1
                action_ls.append(action_vec)

            
            obs_next_ls, reward_ls, done_ls, info_ls = env.step(action_ls)
            score_ls += reward_ls

            obs_ls = obs_next_ls


            # ******* 打印回合结果 ******** #
            # print()



# from make_env import make_env
# import numpy as np

# env = make_env('simple_tag')

# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         agent_actions = []
#         for i, agent in enumerate(env.world.agents):
#             # This is a Discrete
#             # https://github.com/openai/gym/blob/master/gym/spaces/discrete.py
#             agent_action_space = env.action_space[i]

#             # Sample returns an int from 0 to agent_action_space.n
#             action = agent_action_space.sample()

#             # Environment expects a vector with length == agent_action_space.n
#             # containing 0 or 1 for each action, 1 meaning take this action
#             action_vec = np.zeros(agent_action_space.n)
#             action_vec[action] = 1
#             agent_actions.append(action_vec)

#         # Each of these is a vector parallel to env.world.agents, as is agent_actions
#         observation, reward, done, info = env.step(agent_actions)
#         print (observation)
#         print (reward)
#         print (done)
#         print (info)
#         print()

