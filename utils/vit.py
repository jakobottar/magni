import torch
from torchvision import models


class ViT(models.VisionTransformer):
    """
    Vision Transformer model architecture
    Custom version with feature extraction support
    """

    def __init__(self):

        # ViT-L/16
        image_size = 224
        patch_size = 16
        num_layers = 24
        num_heads = 16
        hidden_dim = 1024
        mlp_dim = 4096

        super().__init__(
            image_size,
            patch_size,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            # dropout: float = 0.0,
            # attention_dropout: float = 0.0,
            # num_classes: int = 1000,
            # representation_size: Optional[int] = None,
            # norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # conv_stem_configs: Optional[List[ConvStemConfig]] = None,)
        )

    def forward(self, x, return_feature=False):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        f = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        f = f[:, 0]

        x = self.heads(f)

        if return_feature:
            return x, f

        return x
