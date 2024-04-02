from torchvision import models
from torchvision.models.convnext import CNBlockConfig


class ConvNeXt(models.ConvNeXt):
    """
    ConvNeXt model architecture
    Custom version with feature extraction support
    """

    def __init__(
        self,
        layer_scale: float = 1e-6,
        num_classes: int = 1000,
        block=None,
        norm_layer=None,
        **kwargs,
    ) -> None:

        # convnext_small
        block_setting = [
            CNBlockConfig(96, 192, 3),
            CNBlockConfig(192, 384, 3),
            CNBlockConfig(384, 768, 27),
            CNBlockConfig(768, None, 3),
        ]

        stochastic_depth_prob = 0.4

        super(ConvNeXt, self).__init__(
            block_setting=block_setting,
            stochastic_depth_prob=stochastic_depth_prob,
            layer_scale=layer_scale,
            num_classes=num_classes,
            block=block,
            norm_layer=norm_layer,
            **kwargs,
        )

    def forward(self, x, return_feature=False):
        f = self.features(x)
        f = self.avgpool(f)
        x = self.classifier(f)

        # flatten empty dimensions
        f = f.flatten(1, -1)

        if return_feature:
            return x, f
        return x
