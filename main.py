"""
Main script to run DRL algorithms with diffusion-based reward shaping.
"""

import argparse
import gymnasium as gym
from drl_algorithms.DRL_SAC import SAC
from critic_scaling.utils import robotics_env_maker, classic_control_env_maker
from critic_scaling.networks import (BasicActorSAC, BasicQNetwork, ReaCritic)
import torch
import os
import wandb
#from gymnasium.wrappers import Monitor
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    # Environment settings
    parser.add_argument("--env-id", type=str, default="HeteroHighDimMEC-v3", help="Environment ID")#Ant-v4 ,HumanoidStandup-v4, MountainCarContinuous-v0, HeteroHighDimMEC-v3
    parser.add_argument('--algorithm', type=str, default='sac', choices=['sac', 'ddpg', 'td3', 'ppo', 'a3c'],
                        help='DRL algorithm to use')
    parser.add_argument("--render", type=bool, default=False)

    # Experiment settings
    parser.add_argument("--exp-name", type=str, default="CTSDRL")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=2)  # GPU
    # parser.add_argument("--cuda", type=int, default=-1)  # Use CPU by default

    # SAC parameters
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--alpha-autotune", type=bool, default=True)

    ########## CTS parameters
    parser.add_argument("--critic-scaling", type=bool, default=True)
    parser.add_argument("--HRstep", type=int, default=8)
    parser.add_argument("--VRstep", type=int, default=1)
    

    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--learning-starts", type=int, default=1e4)
    # parser.add_argument("--learning-starts", type=int, default=1000)

    # Logging parameters
    parser.add_argument("--log_dir", type=str, default="runs", help="Directory to save logs")
    parser.add_argument("--write-frequency", type=int, default=100)
    parser.add_argument("--save-frequency", type=int, default=10000)
    parser.add_argument("--plot-rewards", action="store_true", help="Plot rewards after training")

    return parser.parse_args()


def main():
    args = parse_args()
    wandb.init(
        project="CTS",
        config=vars(args),
        name=f"{args.algorithm}-{args.env_id}-{args.HRstep}-{args.VRstep}"
    )

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    # env = gym.make(args.env_id)
    env = robotics_env_maker(env_id=args.env_id, seed=args.seed, render=args.render) if args.env_id.startswith(
        "My") else classic_control_env_maker(env_id=args.env_id, seed=args.seed, render=args.render)
    
    # Setup device
    device = f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    log_dir = os.path.join(args.log_dir, f"{args.exp_name}-{args.env_id}-{args.seed}")
    os.makedirs(log_dir, exist_ok=True)

    # critic_scaling = args.critic_scaling
    # if critic_scaling:
    #     critic = ReaCritic
    # else:
    #     critic = BasicQNetwork

    def init_drl_model(algorithm_name, env, args, device, log_dir): 
        
    
        def get_critic(algo):
            return {
                'sac': ReaCritic if args.critic_scaling else BasicQNetwork,
        
            }[algo]
    
        critic_class = get_critic(algorithm_name)
    
        # -----------------------------
        
        def build_critic(observation_space, action_space=None):
            HRstep = args.HRstep
            VRstep = args.VRstep
        
            if args.critic_scaling:
                if algorithm_name in ['sac', 'ddpg', 'td3']:
                    return critic_class(observation_space, action_space, HRstep, VRstep).to(device)
                elif algorithm_name == 'ppo':
                    return critic_class(observation_space, HRstep, VRstep).to(device)
                elif algorithm_name == 'a3c':
                    return critic_class(observation_space, HRstep, VRstep).to(device)
            else:
                if algorithm_name in ['sac', 'ddpg', 'td3']:
                    return critic_class(observation_space, action_space).to(device)
                elif algorithm_name == 'ppo':
                    return critic_class(observation_space).to(device)
                elif algorithm_name == 'a3c':
                    return critic_class(observation_space).to(device)

    
        
        critic_creator = lambda: build_critic(env.observation_space, getattr(env, 'action_space', None))
        critic = get_critic(algorithm_name)
        algorithms = {
            'sac': lambda: SAC(
                env=env,
                actor_class=BasicActorSAC,
                HRstep=args.HRstep,
                VRstep=args.VRstep,
                critic_class=critic,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                alpha_lr=args.alpha_lr,
                tau=args.tau,
                gamma=args.gamma,
                alpha=args.alpha,
                alpha_autotune=args.alpha_autotune,
                device=device,
                seed=args.seed,
                save_folder=log_dir,
                write_frequency=args.write_frequency
            ),
            'ddpg': lambda: DDPG(
                env=env,
                actor_class=BasicActorDDPG,
                HRstep=args.HRstep,
                VRstep=args.VRstep,
                critic=build_critic(env.observation_space, env.action_space),
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                tau=args.tau,
                gamma=args.gamma,
                device=device,
                seed=args.seed,
                save_folder=log_dir,
                write_frequency=args.write_frequency
            ),
            'td3': lambda: TD3(
                env=env,
                actor_class=BasicActorDDPG,
                HRstep=args.HRstep,
                VRstep=args.VRstep,
                critic=build_critic(env.observation_space, env.action_space),
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                tau=args.tau,
                gamma=args.gamma,
                device=device,
                seed=args.seed,
                save_folder=log_dir,
                write_frequency=args.write_frequency
            ),
            'ppo': lambda: PPO(
                env=env,
                actor_class=BasicActorPPO,
                HRstep=args.HRstep,
                VRstep=args.VRstep,
                #actor=BasicActorPPO(env.observation_space, env.action_space),
                critic = build_critic(env.observation_space, env.action_space),
                batch_size=args.batch_size,
                timesteps_per_batch=1,
                update_epochs=10,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                gamma=args.gamma,
                clip_range=0,
                device=device,
                seed=args.seed,
                save_folder=log_dir,
                write_frequency=args.write_frequency
            ),
            'a3c': lambda: A3C(
                env_name=args.env_id,
                actor_class=BasicActorA3C,
                HRstep=args.HRstep,
                VRstep=args.VRstep,
                critic_creator=build_critic(env.observation_space),
                num_processes=4,
                gamma=args.gamma,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                device=device,
                seed=args.seed,
                save_folder=log_dir,
                write_frequency=args.write_frequency
            )
        }
        
        
    
        
        if algorithm_name not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}. Available options: {list(algorithms.keys())}")
    
        return algorithms[algorithm_name]()

    # Usage in main:
    algorithm_name = args.algorithm
    drlmodel = init_drl_model(algorithm_name, env, args, device, log_dir)
    drlmodel.learn(args.total_timesteps)

    print("Starting training...")
    drlmodel.learn(
        total_timesteps=args.total_timesteps,
        learning_starts=args.learning_starts,
        save_frequency=args.save_frequency
    )
    drlmodel.save("final")
    print(f"Training finished. Logs and models are saved in {log_dir}")

if __name__ == "__main__":
    main()