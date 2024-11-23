import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, timesteps=1000, classes=10):
        super().__init__()
        self._embeddings_time = nn.Embedding(timesteps, 32)
        self._linear_time = nn.Linear(32, 32)
        self._embeddings_class = nn.Embedding(classes, 32)
        self._linear_class = nn.Linear(32, 32)

        self._encoder_00 = nn.Conv2d(1 + 32, 64, kernel_size=3, stride=1, padding=1,)
        self._encoder_01 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,)
        self._encoder_10 = nn.Conv2d(64 + 32, 64, kernel_size=3, stride=1, padding=1,)
        self._encoder_11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,)
        self._encoder_20 = nn.Conv2d(64 + 32, 128, kernel_size=3, stride=1, padding=2,)
        self._encoder_21 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,)
        self._encoder_30 = nn.Conv2d(128 + 32, 128, kernel_size=3, stride=1, padding=1,)
        self._encoder_31 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,)

        self._pooling_0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._pooling_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._pooling_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self._decoder_00 = nn.ConvTranspose2d(128 + 128, 128, kernel_size=3, stride=1, padding=1,)
        self._decoder_01 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1,)
        self._decoder_10 = nn.ConvTranspose2d(128 + 64, 64, kernel_size=3, stride=1, padding=1,)
        self._decoder_11 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1,)
        self._decoder_20 = nn.ConvTranspose2d(64 + 64, 64, kernel_size=3, stride=1, padding=1,)
        self._decoder_21 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1,)
        self._decoder_30 = nn.ConvTranspose2d(64 + 1, 32, kernel_size=3, stride=1, padding=1,)
        self._decoder_31 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1,)

        self._upsampling_0 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self._upsampling_1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1)
        self._upsampling_2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self._upsampling_3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self._upsampling_4 = nn.ConvTranspose2d(1, 1, kernel_size=1, stride=1)

        self._conv_out = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x, t, label):
        emb_time = self._linear_time(F.relu(self._embeddings_time(t)))
        emb_class = self._linear_class(F.relu(self._embeddings_class(label)))

        emb_time_class = emb_time + emb_class
        
        emb_time_class_0 = emb_time_class.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, x.shape[2], x.shape[3])
        e0 = torch.cat([x, emb_time_class_0], dim=1)
        e0 = F.leaky_relu_(self._encoder_00(e0))
        e0 = F.leaky_relu_(self._encoder_01(e0))
        e0 = self._pooling_0(e0)
        # print("0e", e0.shape)

        emb_time_class_1 = emb_time_class.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, e0.shape[2], e0.shape[3])
        e1 = torch.cat([e0, emb_time_class_1], dim=1)
        e1 = F.leaky_relu_(self._encoder_10(e1))
        e1 = F.leaky_relu_(self._encoder_11(e1))
        e1 = self._pooling_1(e1)
        # print("1e", e1.shape)

        emb_time_class_2 = emb_time_class.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, e1.shape[2], e1.shape[3])
        e2 = torch.cat([e1, emb_time_class_2], dim=1)
        e2 = F.leaky_relu_(self._encoder_20(e2))
        e2 = F.leaky_relu_(self._encoder_21(e2))
        e2 = self._pooling_2(e2)
        # print("2e", e2.shape)

        emb_time_class_3 = emb_time_class.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, e2.shape[2], e2.shape[3])
        e3 = torch.cat([e2, emb_time_class_3], dim=1)
        e3 = F.leaky_relu_(self._encoder_30(e3))
        e3 = F.leaky_relu_(self._encoder_31(e3))
        e3 = self._pooling_3(e3)
        # print("3e", e3.shape)

        d0 = self._upsampling_0(e3)
        d0 = torch.cat([d0, e2], dim=1)
        d0 = F.leaky_relu_(self._decoder_00(d0))
        d0 = F.leaky_relu_(self._decoder_01(d0))
        # print("0d", d0.shape)
        
        d1 = self._upsampling_1(d0)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = F.leaky_relu_(self._decoder_10(d1))
        d1 = F.leaky_relu_(self._decoder_11(d1))
        # print("1d", d1.shape)

        d2 = self._upsampling_2(d1)
        d2 = torch.cat([d2, e0], dim=1)
        d2 = F.leaky_relu_(self._decoder_20(d2))
        d2 = F.leaky_relu_(self._decoder_21(d2))
        # print("2d", d2.shape)

        d3 = self._upsampling_3(d2)
        d3 = torch.cat([d3, x], dim=1)
        d3 = F.leaky_relu_(self._decoder_30(d3))
        d3 = F.leaky_relu_(self._decoder_31(d3))
        # print("3d", d3.shape)

        out = self._conv_out(d3)

        return out


if __name__ == "__main__":
    model = UNet()

    total_params = sum(p.numel() for p in model.parameters())
    print("Model size: {}".format(total_params))

    random_input = torch.randn(4, 1, 28, 28)
    timestep = torch.LongTensor([0, 1, 2, 3])
    label = torch.LongTensor([0, 1, 2, 3])
    print(model(random_input, timestep, label).shape)
