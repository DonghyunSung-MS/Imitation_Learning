import argparse
import agents.ppo.rl_agent as rl_agent
from imit_configs import IMIT_CONFIGS
from tasks.humanoid_CMU import humanoid_CMU_imitation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="sub7_walk1")
    parser.add_argument("--seed",type=int,default=1)
    args = parser.parse_args()

    configs = IMIT_CONFIGS[args.env] #presetting prameters for each enviroment.

    env = humanoid_CMU_imitation.walk()
    env._task.__init__(move_speed=0, random= args.seed)
    env._task.set_referencedata(env, configs.filename, configs.max_num_frames)

    model_path = configs.model_dir+'190th_model_a.pth.tar'
    imit_agent = rl_agent.PPOAgent(env, configs, args.seed)
    imit_agent.test_interact(model_path)

if __name__=="__main__":
    main()
