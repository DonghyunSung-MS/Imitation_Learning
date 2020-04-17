
class PPOActor(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        self.a_net = mlp(input_dim, args.hidden_size, output_dim, args.layer_size, nn.Tanh)
        #print(self.a_net)

    def forward(self, x):
        # input -> output(mean of torque+std(constant))
        mu = self.a_net(x)
        log_std = torch.zeros_like(mu)
        std = torch.exp(log_std)

        return mu, std

    def get_action(self, mu, std):
        normal = Normal(mu, std)
        action = normal.sample()
        return action.data.numpy()

    def get_log_prob(self, actions, mu, std):
        normal = Normal(mu, std)
        log_prob = normal.log_prob(actions) #log_probability of policy
        return log_prob

class PPOCritic(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        self.c_net = mlp(input_dim, args.hidden_size, 1, args.layer_size, nn.Tanh)
        #print(self.c_net)


    def forward(self, x):
        return self.c_net(x)
