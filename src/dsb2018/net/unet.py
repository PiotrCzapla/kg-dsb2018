from fastai.imports import *
from fastai.torch_imports import *

def center_crop_tensor(tensor, new_width, new_height):
    _,_,orig_h,orig_w = tensor.shape
    x = (orig_h - new_height) // 2
    y = (orig_w - new_width) // 2
    return tensor[:, :, x:(x+new_height), y:(y+new_height)]

class Collector(nn.Module):
    def forward(self, input):
        self.tensor = input
        return input


class Link(nn.Module):
    def __init__(self):
        super(Link, self).__init__()
        self.collector = [Collector()]

    def capture_above(self):
        return self.collector[0]

    def concat_with_above(self):
        return self

    def forward(self, input):
        _,_,h,w = input.shape
        concat = torch.cat([center_crop_tensor(self.collector[0].tensor, h, w), input], 1)
        return concat

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #print("Init", classname)
        nn.init.xavier_normal(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class StdConv(nn.Conv2d):
    def __init__(self,  num_in, num_out, kernel_size, stride, padding, dilation, relu, bn):
        super().__init__(num_in, num_out, kernel_size, stride, padding, dilation)
        self.bn = nn.BatchNorm2d(num_out) if bn else lambda x: x
        self.relu = nn.ReLU() if relu else lambda x: x


    def forward(self, x):
        return self.relu(self.bn(super().forward(x)))

class UNet(nn.Module):
    """Class representing an unet model."""

    def __init__(self, m=1, bn=False):
        super().__init__()

        self.size_calc=[]
        self.m = m
        self.bn = bn
        l1 = Link()
        l2 = Link()
        l3 = Link()
        l4 = Link()

        self.unet = nn.Sequential(
            self.conv_down(l1, 3, 8*m),
            self.conv_down(l2, 8*m, 16*m),
            self.conv_down(l3, 16*m, 32*m),
            self.conv_down(l4, 32*m, 64*m),
            self.conv_bottom(64*m, 128*m),
            self.conv_up(l4, 128*m, 64*m),
            self.conv_up(l3, 64*m, 32*m),
            self.conv_up(l2, 32*m, 16*m),
            self.conv_up(l1, 16*m, 8*m),
            self.conv2d(8*m, 1, 1, relu=False),
            nn.Sigmoid(),
        )

        self.allowed_sizes = {}
        for x in range(200, 1024):
            self.allowed_sizes.setdefault(int(self.calculate_output_size(x)), x)

        #self.unet.apply(kaiming_normal)
        self.unet.apply(weights_init)

    #res = [nn.BatchNorm1d(num_features=ni)]
    #if p: res.append(nn.Dropout(p=p))

    def conv2d(self, num_in, num_out, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        self.size_calc.append(
            lambda sz_in: np.floor((sz_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
        return StdConv(num_in, num_out, kernel_size, stride, padding, dilation, relu, self.bn)

    def max_pool(self, kernel_size, stride=1, padding=0, dilation=1):
        self.size_calc.append(
            lambda sz_in: np.floor((sz_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
        return nn.MaxPool2d(kernel_size, stride, padding, dilation)

    def conv_transpose2d(self, num_in, num_out, kernel_size, stride=1, padding=0, output_padding=0):
        self.size_calc.append(
            lambda sz_in:  (sz_in - 1) * stride - 2 * padding + kernel_size + output_padding)
        return nn.ConvTranspose2d(num_in, num_out, kernel_size, stride, padding, output_padding)

    def conv_down(self, link, num_in, num_out):
        return nn.Sequential(
            self.conv2d(num_in,  num_out, kernel_size=3),
            self.conv2d(num_out, num_out, kernel_size=3),
            link.capture_above(),
            self.max_pool(2, stride=2))

    def conv_bottom(self, num_in, num_out):
        return nn.Sequential(
            self.conv2d(num_in,  num_out, kernel_size=3),
            self.conv2d(num_out, num_out, kernel_size=3))

    def conv_up(self, link, num_in, num_out):
        return nn.Sequential(
            self.conv_transpose2d(num_in, num_out, 2, stride=2),
            link.concat_with_above(),
            self.conv2d(num_out*2,  num_out, kernel_size=3),
            self.conv2d(num_out, num_out, kernel_size=3))

    def calculate_output_size(self, sz_in):
        sz = sz_in
        for f in self.size_calc:
            sz = f(sz)
        return sz

    def forward(self, input):
        sz = input.shape[2]
        in_sz = self.allowed_sizes.get(sz)
        padding = int((in_sz - sz)/2)
        input = F.pad(input, tuple([padding] * 4), 'reflect')
        out = self.unet(input)
        return out
