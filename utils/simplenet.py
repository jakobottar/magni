from torch import nn


class SimpleMLP(nn.Module):

    def __init__(self, num_classes, input_dim: int = 4096):
        super(SimpleMLP, self).__init__()

        self.fcblock = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 16),
        )
        self.out = nn.Linear(16, num_classes)

    def forward(self, x, return_feature=False):
        t = self.fcblock(x)
        if return_feature:
            return self.out(t), t
        return self.out(t)
