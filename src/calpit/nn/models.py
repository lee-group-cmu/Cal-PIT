import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=6, output_dim=1, hidden_layers=[512, 512, 512], sigmoid=False):
        super().__init__()
        self.all_layers = [input_dim]
        self.all_layers.extend(hidden_layers)
        self.all_layers.append(output_dim)
        self.layer_list = []

        for i in range(len(self.all_layers) - 1):
            self.layer_list.append(nn.Linear(self.all_layers[i], self.all_layers[i + 1]))
            self.layer_list.append(nn.PReLU())
        self.layer_list.pop()
        if sigmoid:
            self.layer_list.append(nn.Sigmoid())
        self.layers = nn.Sequential(*self.layer_list)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)
