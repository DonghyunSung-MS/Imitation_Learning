import argparse
import wandb
import agents.ppo.rl_agent as rl_agent
from imit_configs import IMIT_CONFIGS
from tasks.humanoid_Tocabi import Tocabi_v1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="sub8_walk1")
    parser.add_argument("--seed",type=int,default=1)
    args = parser.parse_args()
    wandb.init(project="imitation-learning-walk")

    configs = IMIT_CONFIGS[args.env] #presetting prameters for each enviroment.
    wandb.config.update(configs)
    env = Tocabi_v1.walk()
    env._task.__init__(move_speed=0, random= args.seed)
    env._task.set_referencedata(env, configs.filename, configs.max_num_frames)


    imit_agent = rl_agent.PPOAgent(env, configs, args.seed)
    imit_agent.train()

if __name__=="__main__":
    main()
