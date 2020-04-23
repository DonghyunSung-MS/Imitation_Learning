import math
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(input_size, hidden_size, output_size, layer_size, act,out_act=nn.Identity):
    layers=[]
    for i in range(layer_size):
        ac = act if i<layer_size-1 else out_act
        if i ==0:
            layers += [nn.Linear(input_size, hidden_size[i]),ac()]

        elif i==layer_size-1:
            layers += [nn.Linear(hidden_size[i-1], output_size),ac()]
        else:
            layers += [nn.Linear(hidden_size[i-1], hidden_size[i]),ac()]

    return nn.Sequential(*layers)

def calculate_gae(masks,rewards,old_values, args):

    previous_advantage = 0
    previsous_rewards2go = 0
    target = 0

    advantages = torch.zeros_like(masks)
    rewards2go = torch.zeros_like(masks)
    old_values = old_values.squeeze(1)

    for i in reversed(range(0,len(masks))):
        if masks[i]==0:
            target = rewards[i]

            advantages[i] = rewards[i] - old_values.data[i]
            rewards2go[i] = rewards[i]

            previous_advantage = advantages[i]
            previsous_rewards2go = rewards2go[i]
        else:
            target = rewards[i] + args.gamma*old_values.data[i+1]
            td_residual = target - old_values.data[i]

            advantages[i] = td_residual + args.gamma * args.lamda * previous_advantage
            rewards2go[i] = rewards[i] + args.gamma * previsous_rewards2go

            previous_advantage = advantages[i]
            previsous_rewards2go = rewards2go[i]

    return rewards2go, advantages


def surrogate_loss(actor, old_policy_log,
                   advantages_samples, states_samples,  actions_samples,
                   mini_batch_index):
    mu, std = actor(states_samples)
    new_policy_samples = actor.get_log_prob(actions_samples, mu, std).squeeze(1)
    #print(new_policy_samples.shape)
    old_policy_samples = old_policy_log[mini_batch_index]
    ratio = torch.exp(new_policy_samples - old_policy_samples)
    #print(new_policy_samples.shape, ratio.shape, advantages_samples.shape)
    surrogate_loss = ratio * advantages_samples

    return surrogate_loss, ratio
