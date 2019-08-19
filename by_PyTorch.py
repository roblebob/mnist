import torch
import torchvision
from pathlib import Path


hyperparams = dict(batch_size_train=64,
                   batch_size_test=1000,
                   learning_rate=0.01,
                   learning_momentum=0.5,
                   n_epochs=3,
                   log_interval=10)

# to guarantee different multiple randoms in multiple instances
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


_path = Path.cwd().joinpath('data').as_posix()
_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(_path, train=True, download=True, transform=_transform),
    batch_size=hyperparams["batch_size_train"],
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(_path, train=False, download=True, transform=_transform),
    batch_size=hyperparams["batch_size_test"],
    shuffle=True
)



