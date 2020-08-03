import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(MLPBlock, self).__init__()
        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.relu)

    def forward(self, x):
        out = self.net(x)
        res = x
        return self.relu(out + res)


class MultiPerceptronNet(nn.Module):
    def __init__(self, num_inputs, num_channels):
        super(MultiPerceptronNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [MLPBlock(in_channels, out_channels)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, num_channels):
        super(MLP, self).__init__()
        self.mlp = MultiPerceptronNet(input_size, num_channels)
        num_levels = len(num_channels)
        num_neurons = 2 ** num_levels
        self.linear = nn.Linear(num_neurons, num_neurons)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.mlp(x)
        return self.linear(y1)


class MLPDiscriminator(nn.Module):
    """Discriminator using simple MLP, outputs a probability for each time step

    Args:
        input_size (int): dimensionality (channels) of the input
        n_layers (int): number of hidden layers
        n_channels (int): number of channels in the hidden layers (it's always the same)

    Input: (batch_size, seq_len, input_size)
    Output: (batch_size, seq_len, 1)
    """

    def __init__(self, input_size, n_layers, n_channel):
        super().__init__()
        # Assuming same number of channels layerwise
        num_channels = [n_channel] * n_layers
        self.mlp = MLP(input_size, 1, num_channels)

    def forward(self, x):
        return torch.sigmoid(self.mlp(x))


class MLPGenerator(nn.Module):
    """Generator using simple MLP, expecting a noise vector for each timestep as input

    Args:
        noise_size (int): dimensionality (channels) of the input noise
        output_size (int): dimenstionality (channels) of the output sequence
        n_layers (int): number of hidden layers
        n_channels (int): number of channels in the hidden layers (it's always the same)

    Input: (batch_size, seq_len, input_size)
    Output: (batch_size, seq_len, output_size)
    """

    def __init__(self, noise_size, output_size, n_layers, n_channel):
        super().__init__()
        num_channels = [n_channel] * n_layers
        self.mlp = MLP(noise_size, output_size, num_channels)

    def forward(self, x):
        return torch.tanh(self.mlp(x))



