import torch
from dotenv import load_dotenv
import os

def test_env():
    """
    Test if the environment variables from .env are loaded.
    """
    load_dotenv()
    path = os.environ.get("NUSCENES")
    print(path)


def test_mps():
    """
    Test if MPS device is available.

    Usage of the device:
        mps_device = torch.device("mps")
        x = torch.ones(5, device=mps_device)
        model = YourModel()
        model.to(mps_device)
    """
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")


if __name__ == "__main__":
    test_env()
    test_mps()

