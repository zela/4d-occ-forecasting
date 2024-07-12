import torch


def set_device():
    """
    Set the device to MPS or CUDA if available.
    :return: device, device_count
    """
    if torch.backends.mps.is_available():
        print("MPS device found.")
        return torch.device("mps"), 0
    elif torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA devices found. Device count: {device_count}")
        return torch.device("cuda:0"), device_count
    else:
        print("No GPU found. Using CPU.")
        return torch.device("cpu"), 0
