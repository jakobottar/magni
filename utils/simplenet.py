from torch import nn


class SimpleMLP(nn.Module):

    def __init__(self, num_classes, feature_dim=16, input_dim: int = 4096):
        super(SimpleMLP, self).__init__()

        self.fcblock = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, feature_dim),
        )
        self.out = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim

        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feature=False):
        t = self.fcblock(x)
        if return_feature:
            return self.out(t), t
        return self.out(t)
