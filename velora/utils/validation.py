import torch


def device_validation(device: str | torch.device) -> torch.device:
    if isinstance(device, str):
        device = torch.device(device)

    return device
