import torch.nn as nn


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)
    elif type(m) == nn.Conv1d:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)

class TextMLP(nn.Module):
    def __init__(self, config):
        super(TextMLP,self).__init__()
        self.config = config
        self.text_module = nn.Sequential(
                nn.Linear(config['text_dim'], config['hidden_dim'], bias=True),
                nn.BatchNorm1d(config['hidden_dim']),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2, bias=True),
                nn.BatchNorm1d(config['hidden_dim'] // 2),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(config['hidden_dim'] // 2, config['hidden_dim'] // 4, bias=True),
                nn.BatchNorm1d(config['hidden_dim'] // 4),
                nn.ReLU(True),
                nn.Dropout(0.5))
        self.hash_layer = nn.Linear()
        self.apply(weights_init)


    def forward(self, input):
        feature = self.text_module(input)
        hash_feature = self.hash_layer(feature)
        return hash_feature
