import torch.nn as nn

__all__ = ["initialize_decoder", "initialize_head"]


def initialize_decoder(module: nn.Module, activation: str = "relu") -> None:
    """Initialize a decoder module."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            if activation == "relu":
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            else:
                nn.init.kaiming_normal_(m.weight, mode="fan_in")

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.Conv1d):
            if activation == "relu":
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            else:
                nn.init.kaiming_normal_(m.weight, mode="fan_in")

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module: nn.Module):
    """Initialize a segmentation head module."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
