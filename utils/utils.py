import math

import torch


def combine_features(method):
    match method.lower():
        case "concat":

            return lambda sem, xrd: torch.cat((sem, xrd), dim=1)

        case "max":

            def max_join(sem, xrd):
                assert sem.shape == xrd.shape
                return torch.max(sem, xrd)

            return max_join

        case "add":

            def add(sem, xrd):
                assert sem.shape == xrd.shape
                return sem + xrd

            return add


def repeat_and_reshape(x: torch.Tensor, width: int):
    """
    repeat and reshape "1-D" tensor of shape (N, 1, A) to (N, 3, B, W)
    """
    if x.dim() == 2:
        x.unsqueeze_(1)
    _, _, a = x.shape
    rows = math.ceil(a / width)
    zeroes = torch.zeros((x.shape[0], 1, rows * width - a), device=x.device)
    x = torch.cat((x, zeroes), dim=2)
    return x.reshape(x.shape[0], 1, rows, width).repeat(1, 3, 1, 1)


if __name__ == "__main__":

    original = torch.rand(size=(4, 1, 4096))

    new = repeat_and_reshape(original, (4, 1, 224, 224))

    print(new.shape)
