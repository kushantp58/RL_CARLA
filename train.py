import argparse
import numpy as np
from parl.utils import csv_logger, logger, tensorboard, ReplayMemory
from env_utils import ParallelEnv, LocalEnv
#from torch_base import TorchModel, TorchSAC, TorchAgent  # Choose base wrt which deep-learning framework you are using
from paddle_base import PaddleModel, PaddleSAC, PaddleAgent
from env_config import EnvConfig
from parl.utils import CSVLogger

WARMUP_STEPS = 2e3
EVAL_EPISODES = 3
MEMORY_SIZE = int(1e4)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # determines the relative importance of entropy term against the reward
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4


# Runs policy for 3 episodes by default and returns average reward
def run_evaluate_episodes(agent, env, eval_episodes):
    avg_reward = 0.
    for k in range(eval_episodes):
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < env._max_episode_steps:
            steps += 1
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def main():
    logger.info("-----------------Carla_SAC-------------------")
    logger.set_dir('./{}_train'.format(args.env))

    csv_logger = CSVLogger('./{}_train/log.csv'.format(args.env))


    # Parallel environments for training
    train_envs_params = EnvConfig['train_envs_params']
    env_num = EnvConfig['env_num']
    env_list = ParallelEnv(args.env, args.xparl_addr, train_envs_params)

    # env for eval
    eval_env_params = EnvConfig['eval_env_params']
    eval_env = LocalEnv(args.env, eval_env_params)

    obs_dim = eval_env.obs_dim
    action_dim = eval_env.action_dim

    # Initialize model, algorithm, agent, replay_memory
    if args.framework == 'torch':
        CarlaModel, SAC, CarlaAgent = TorchModel, TorchSAC, TorchAgent
    elif args.framework == 'paddle':
        CarlaModel, SAC, CarlaAgent = PaddleModel, PaddleSAC, PaddleAgent
    model = CarlaModel(obs_dim, action_dim)
    algorithm = SAC(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    agent = CarlaAgent(algorithm)
    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)

    total_steps = 0
    last_save_steps = 0
    test_flag = 0
    st = 0.1
    ft = 0.1
    sr = 0

    obs_list = env_list.reset()

    while total_steps < args.train_total_steps:
        # Train episode
        if rpm.size() < WARMUP_STEPS:
            action_list = [
                np.random.uniform(-50, 50, size=action_dim)
                for _ in range(env_num)
            ]
        else:
            action_list = [agent.sample(obs) for obs in obs_list]
        next_obs_list, reward_list, done_list, info_list = env_list.step(
            action_list)


        # Store data in replay memory
        for i in range(env_num):
            rpm.append(obs_list[i], action_list[i], reward_list[i],
                       next_obs_list[i], done_list[i])
            if(done_list[i]==True):
                st = 0
            else:
                ft = 0
                
                       
        obs_list = env_list.get_obs()
        total_steps = env_list.total_steps
        # Train agent after collecting sufficient data
        if rpm.size() >= WARMUP_STEPS:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                BATCH_SIZE)
            #agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,batch_terminal)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

        # Save agent
        if total_steps > int(1e5) and total_steps > last_save_steps + int(1e4):
            agent.save('./{}_model/step_{}_model.ckpt'.format(
                args.framework, total_steps))
            last_save_steps = total_steps

        # Evaluate episode
        if (total_steps + 1) // args.test_every_steps >= test_flag:
            while (total_steps + 1) // args.test_every_steps >= test_flag:
                test_flag += 1
            avg_reward = run_evaluate_episodes(agent, eval_env, EVAL_EPISODES)    
            sr = st/(st+ft)
            csv_logger.log_dict({'Epochs': total_steps,'Evaluation over episodes':EVAL_EPISODES,'Average Reward per episode': avg_reward,'success rate':sr})
            tensorboard.add_scalar('eval/episode_reward', avg_reward,
                                   total_steps)
            logger.info(
                'Total steps {}, Evaluation over {} episodes, Average reward: {}'
                .format(total_steps, EVAL_EPISODES, avg_reward))
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xparl_addr",
        default='localhost:8080',
        help='xparl address for parallel training')
    parser.add_argument("--env", default="carla-v0")
    parser.add_argument(
        '--framework',
        default='paddle',
        help='choose deep learning framework: torch or paddle')
    parser.add_argument(
        "--train_total_steps",
        default=5e5,
        type=int,
        help='max time steps to run environment')
    parser.add_argument(
        "--test_every_steps",
        default=1e3,
        type=int,
        help='the step interval between two consecutive evaluations')
    args = parser.parse_args()

    main()
