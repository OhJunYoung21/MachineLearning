## 3dCNN모델을 정의한다.

class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3))

    def forward(self, x):
        return self.conv3d(x)
