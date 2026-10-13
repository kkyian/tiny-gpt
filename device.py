"""Select an accelerator without requiring one."""
import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    x = torch.ones(4, device=device, requires_grad=True)
    x.square().sum().backward()
    assert torch.isfinite(x.grad).all().item()
    print(f"PyTorch {torch.__version__}; device: {device}; forward/backward: PASS")
