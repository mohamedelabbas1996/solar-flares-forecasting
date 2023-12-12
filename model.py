import torch
import torch.nn as nn
from transformers import (
    ViTConfig,
    ViTImageProcessor,
    ViTForImageClassification,
)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.ModuleList(
            [cnn_block(11, 1), cnn_block(11, 64)]
            + [cnn_block(3, 64), cnn_block(3, 64), cnn_block(3, 64)]
        )
        self.linear = nn.Linear(64, 128)
        self.bn = nn.BatchNorm1d(128)
        self.drop_out = nn.Dropout(0.5)
        self.linear2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop_out2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(64, 2)

    def forward(self, x):
        print(x.shape)
        for idx, l in enumerate(self.cnn):
            print(idx, x.shape)
            x = l(x)
        print(x.shape)
        x = x.squeeze(2).squeeze(2).reshape(-1, 64)
        x = self.linear(x)
        print(x.shape)
        x = self.bn(x)
        print(x.shape)
        x = self.drop_out(x)
        print(x.shape)
        x = self.linear2(x)
        print(x.shape)
        x = self.bn2(x)
        print(x.shape)
        x = self.drop_out2(x)
        print(x.shape)
        x = self.linear3(x)
        return x


class cnn_block(nn.Module):
    def __init__(self, filter_size, in_channels):
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=filter_size
        )
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        return x


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=4, hidden_size=64, num_layers=2, batch_first=True
        )

    def forward(self, x):
        return self.lstm(x)


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        config = ViTConfig(num_channels=1)
        self.image_processor = ViTImageProcessor(
            do_normalize=False, do_rescale=False, do_resize=False
        )
        self.vit = ViTForImageClassification(config)

    def forward(self, x):
        model_inputs = self.image_processor(x, return_tensors="pt")
        outputs = self.vit(**model_inputs)
        return outputs.logits


if __name__ == "__main__":
    x = torch.rand(1, 4, 15)
    y = torch.tensor([1]).reshape(1, 1)
    magnetogram = torch.rand(1, 1, 224, 224)
    cnn = CNN()

    # print(magnetogram.shape)
    # logits = cnn(magnetogram)
    vit = ViT()
    labels = torch.randint(0, 2, (32,), dtype=torch.float32)
    print(vit(magnetogram))
# print(logits)

# x = torch.rand(3, 15, 4)
# lstm = LSTM()
# output = lstm(x)
# print(output[0][:, -1, :].shape)
