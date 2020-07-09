import os
import gym
import numpy as np

env = gym.make("gym_imit:DyrosRed-v1")


#Simple Dyamics randomization test
print("body mass",sum(env.model.body_mass))

body_name_list = env.get_body_name()
print("body(joint) length : ",len(body_name_list))
print(body_name_list)

act_name_list = env.get_act_name()
print("motor(actuator) length : ",len(act_name_list))
print(act_name_list)

_OUT_DIR = os.path.join(os.getcwd()+"/../randXml/test3/")
#print(_OUT_DIR)
xml_name = 'random.xml'
env.domain_randomizer(xml_name, _OUT_DIR)
print()
print("After randomization")
print("body mass",sum(env.model.body_mass))

body_name_list = env.get_body_name()
print("body(joint) length : ",len(body_name_list))
print(body_name_list)

act_name_list = env.get_act_name()
print("motor(actuator) length : ",len(act_name_list))
print(act_name_list)

while True:
    env.render()
