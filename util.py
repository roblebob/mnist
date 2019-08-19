import torch
import torchvision
from matplotlib import pyplot as plt


def printer(data):

    # if using pytorch's data pipeline/sserver
    if isinstance(data, torch.utils.data.DataLoader):

        data_iter = enumerate(data)
        batch_idx, (example_data, example_targets) = next(data_iter)

        fig = plt.figure()
        for i in range(6):
          plt.subplot(2,3,i+1)
          plt.tight_layout()
          plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
          plt.title("Ground Truth: {}".format(example_targets[i]))
          plt.xticks([])
          plt.yticks([])

        plt.show()

