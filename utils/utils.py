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
