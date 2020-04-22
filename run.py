import argparse
import agents.ppo.rl_agent as rl_agent
from imit_configs import IMIT_CONFIGS
from tasks.humanoid_CMU import humanoid_CMU_imitation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="sub7_walk1")
    args = parser.parse_args()

    configs = IMIT_CONFIGS[args.env] #presetting prameters for each enviroment.

    env = humanoid_CMU_imitation.walk()
    env._task.set_referencedata(env, configs.filename, configs.max_num_frames)


    imit_agent = rl_agent.PPOAgent(env, configs)
    imit_agent.train()

if __name__=="__main__":
    main()
