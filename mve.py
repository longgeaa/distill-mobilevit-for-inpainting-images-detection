
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Reduce
import numpy as np
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


      
class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(64, 64)
        self.reduce4 = Conv1x1(64, 64)
        self.block = nn.Sequential(
            ConvBNR(64 + 64, 64, 3),
            ConvBNR(64, 64, 3),
            nn.Conv2d(64, 1, 1))

    def forward(self, x4, x1):
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out

class CustomizedConv(nn.Module):
    def __init__(self, channels=1, choice='similarity'):
        super(CustomizedConv, self).__init__()
        self.channels = channels
        self.choice = choice
        kernel = [[0.03598, 0.03735, 0.03997, 0.03713, 0.03579],
                  [0.03682, 0.03954, 0.04446, 0.03933, 0.03673],
                  [0.03864, 0.04242, 0.07146, 0.04239, 0.03859],
                  [0.03679, 0.03936, 0.04443, 0.03950, 0.03679],
                  [0.03590, 0.03720, 0.04003, 0.03738, 0.03601]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.kernel = nn.modules.utils._pair(3)
        self.stride = nn.modules.utils._pair(1)
        self.padding = nn.modules.utils._quadruple(0)
        self.same = False

    def __call__(self, x):
        if self.choice == 'median':
            x = F.pad(x, self._padding(x), mode='reflect')
            x = x.unfold(2, self.kernel[0], self.stride[0]).unfold(3, self.kernel[1], self.stride[1])
            x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        else:
            x = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

def get_pf_list():
    pf1 = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0]]).astype('float32')

    pf2 = np.array([[0, 0, 0],
                    [0, -1, 1],
                    [0, 0, 0]]).astype('float32')

    pf3 = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]]).astype('float32')

    return [torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone(),
            torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone(),
            torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone()
            ]


def constrained_weights(weights):
    weights = weights.permute(2, 3, 0, 1)
    # Scale by 10k to avoid numerical issues while normalizing
    weights = weights * 10000

    # Set central values to zero to exlude them from the normalization step
    weights[2, 2, :, :] = 0

    # Pass the weights
    filter_1 = weights[:, :, 0, 0]
    filter_2 = weights[:, :, 0, 1]
    filter_3 = weights[:, :, 0, 2]

    # Normalize the weights for each filter.
    # Sum in the 3rd dimension, which contains 25 numbers.
    filter_1 = filter_1.reshape(1, 1, 1, 25)
    filter_1 = filter_1 / filter_1.sum(3).reshape(1, 1, 1, 1)
    filter_1[0, 0, 0, 12] = -1

    filter_2 = filter_2.reshape(1, 1, 1, 25)
    filter_2 = filter_2 / filter_2.sum(3).reshape(1, 1, 1, 1)
    filter_2[0, 0, 0, 12] = -1

    filter_3 = filter_3.reshape(1, 1, 1, 25)
    filter_3 = filter_3 / filter_3.sum(3).reshape(1, 1, 1, 1)
    filter_3[0, 0, 0, 12] = -1

    # Prints are for debug reasons.
    # The sums of all filter weights for a specific filter
    # should be very close to zero.
    # print(filter_1)
    # print(filter_2)
    # print(filter_3)
    # print(filter_1.sum(3).reshape(1,1,1,1))
    # print(filter_2.sum(3).reshape(1,1,1,1))
    # print(filter_3.sum(3).reshape(1,1,1,1))

    # Reshape to original size.
    filter_1 = filter_1.reshape(1, 1, 5, 5)
    filter_2 = filter_2.reshape(1, 1, 5, 5)
    filter_3 = filter_3.reshape(1, 1, 5, 5)

    # Pass the weights back to the original matrix and return.
    weights[:, :, 0, 0] = filter_1
    weights[:, :, 0, 1] = filter_2
    weights[:, :, 0, 2] = filter_3

    weights = weights.permute(2, 3, 0, 1)
    return weights


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input

def get_sobel(in_chan, out_chan):
    '''
    filter_x = np.array([
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3],
    ]).astype(np.float32)
    filter_y = np.array([
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3],
    ]).astype(np.float32)
    '''
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y

# helpers

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """Transformer block described in ViT.
    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MV2Block(nn.Module):
    """MV2 block described in MobileNetV2.
    Paper: https://arxiv.org/pdf/1801.04381
    Based on: https://github.com/tonylins/pytorch-mobilenet-v2
    """

    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out

class Spade(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(Spade, self).__init__()
        self.param_free_norm = nn.BatchNorm2d(out_channels, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.mlp_gamma = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        

    def forward(self, x, edge):
        normalized = self.param_free_norm(x)

        edge = F.interpolate(edge, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(edge)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d',
                      ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    """MobileViT.
    Paper: https://arxiv.org/abs/2110.02178
    Based on: https://github.com/chinhsuanwu/mobilevit-pytorch
    """

    def __init__(
        self,
        image_size,
        dims,
        channels,
        num_classes,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3)
    ):
        super().__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'

        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, last_dim = channels

        self.conv1 = conv_nxn_bn(3, init_dim, stride=2)
        self.pf_conv = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1, bias=False)
        self.bayar_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        
        self.stem = nn.ModuleList([])
        self.stem.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.stem.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        
        self.conv11 = conv_nxn_bn(12, init_dim, stride=2)
        self.stem1 = nn.ModuleList([])
        self.stem1.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.stem1.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.stem1.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.stem1.append(MV2Block(channels[2], channels[3], 1, expansion))

        self.hor=HOR()

        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[3], channels[4], 2, expansion),
            MobileViTBlock(dims[0], depths[0], channels[5],
                           kernel_size, patch_size, int(dims[0] * 2))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[5], channels[6], 2, expansion),
            MobileViTBlock(dims[1], depths[1], channels[7],
                           kernel_size, patch_size, int(dims[1] * 4))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[7], channels[8], 2, expansion),
            MobileViTBlock(dims[2], depths[2], channels[9],
                           kernel_size, patch_size, int(dims[2] * 4))
        ]))

        self.sobel_x1, self.sobel_y1 = get_sobel(64, 1)
        self.sobel_x2, self.sobel_y2 = get_sobel(64, 1)
        self.eam = EAM()

        self.decision = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(channels[-2], channels[-2], 3, stride=1, padding=1),
            nn.BatchNorm2d(channels[-2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[-2], channels[-2], 3, stride=1, padding=1),
            nn.BatchNorm2d(channels[-2]),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(channels[-2], channels[-2], 3, stride=1, padding=1),
            nn.BatchNorm2d(channels[-2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[-2], channels[-2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[-2]),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(channels[-2], 1, 3, stride=1, padding=1),
        )
        self.median = CustomizedConv(choice='median')
        self.pf_list = get_pf_list()
        self.reset_pf()

    def forward(self, x):
        b,c,H,W=x.shape
        
        self.bayar_conv.weight.data = constrained_weights(self.bayar_conv.weight.data)
        bayar_x = self.bayar_conv(x)
        pf_x = self.pf_conv(x)
        noise_x = torch.cat([pf_x,bayar_x],dim=1)

        x = self.conv1(x)      
        noise_x = self.conv11(noise_x)
        
        for conv in self.stem:
            x = conv(x)
        for conv in self.stem1:
            noise_x = conv(noise_x)

        
        s1 = run_sobel(self.sobel_x1, self.sobel_y1, x)
        s2 = run_sobel(self.sobel_x2, self.sobel_y2, noise_x)
        edge = self.eam(s2, s1)
        edge_att = torch.sigmoid(edge)

        x = self.hor(noise_x,x)
        for conv, attn in self.trunk:
            x = conv(x)
            edge_att = F.interpolate(edge_att, x.size()[2:], mode='bilinear', align_corners=False)
            x = attn(x, edge_att)

        x = self.decision(x)
        x = self.median(x)
        x = F.interpolate(x, (H, W))
        x = nn.Sigmoid()(x)
        oe = F.interpolate(torch.sigmoid(edge), (64,64), mode='bilinear', align_corners=False)
        return x, oe

    def reset_pf(self):
        for idx, pf in enumerate(self.pf_list):
            self.pf_conv.weight.data[idx, :, :, :] = pf
            
class MobileViT32(nn.Module):
    """MobileViT.
    Paper: https://arxiv.org/abs/2110.02178
    Based on: https://github.com/chinhsuanwu/mobilevit-pytorch
    """

    def __init__(
        self,
        image_size,
        dims,
        channels,
        num_classes,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3)
    ):
        super().__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'

        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, last_dim = channels

        self.conv1 = conv_nxn_bn(12, init_dim, stride=2)
        self.pf_conv = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1, bias=False)
        self.bayar_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        
        self.stem = nn.ModuleList([])
        self.stem.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.stem.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))


        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[3], channels[4], 2, expansion),
            MobileViTBlock(dims[0], depths[0], channels[5],
                           kernel_size, patch_size, int(dims[0] * 2)),
            MV2Block(channels[4], channels[4], 1, expansion),
            MobileViTBlock(dims[1], depths[1], channels[5],
                           kernel_size, patch_size, int(dims[1] * 4)),
            MV2Block(channels[4], channels[4], 1, expansion),
            MobileViTBlock(dims[2], depths[2], channels[5],
                           kernel_size, patch_size, int(dims[2] * 4))
        ]))


        self.decision = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(channels[-6], channels[-6], 3, stride=1, padding=1),
            nn.BatchNorm2d(channels[-6]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[-6], channels[-6], 3, stride=1, padding=1),
            nn.BatchNorm2d(channels[-6]),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(channels[-6], channels[-6], 3, stride=1, padding=1),
            nn.BatchNorm2d(channels[-6]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[-6], channels[-6], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[-6]),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(channels[-6], 1, 3, stride=1, padding=1),
        )
        self.median = CustomizedConv(choice='median')
        self.pf_list = get_pf_list()
        self.reset_pf()

    def forward(self, x):
        b,c,H,W=x.shape
        
        self.bayar_conv.weight.data = constrained_weights(self.bayar_conv.weight.data)
        bayar_x = self.bayar_conv(x)
        pf_x = self.pf_conv(x)
        noise_x = torch.cat([pf_x,bayar_x],dim=1)

        x = self.conv1(noise_x)      
        
        for conv in self.stem:
            x = conv(x)

        for conv in self.trunk:
            for i in conv:
                x = i(x)


        x = self.decision(x)
        x = self.median(x)
        x = F.interpolate(x, (H, W))
        x = nn.Sigmoid()(x)
        return x

    def reset_pf(self):
        for idx, pf in enumerate(self.pf_list):
            self.pf_conv.weight.data[idx, :, :, :] = pf
            
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class aggregation_add(nn.Module):
    def __init__(self, channel):
        super(aggregation_add, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(x1) + x2
        x3_1 = self.conv_upsample2(x1) \
               + self.conv_upsample3(x2) + x3

        x3_2 = torch.cat((x3_1, self.conv_upsample4(x1_1), self.conv_upsample5(x2_1)), 1)
        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x            
class aggregation_add2(nn.Module):
    def __init__(self, channel):
        super(aggregation_add2, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) + x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               + self.conv_upsample3(self.upsample(x2)) + x3

        x3_2 = torch.cat((x3_1, self.conv_upsample4(self.upsample(self.upsample(x1_1))), self.conv_upsample5(self.upsample(x2_1))), 1)
        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x
        
class MobileViT32FPN(nn.Module):
    """MobileViT.
    Paper: https://arxiv.org/abs/2110.02178
    Based on: https://github.com/chinhsuanwu/mobilevit-pytorch
    """

    def __init__(
        self,
        image_size,
        dims,
        channels,
        num_classes,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3)
    ):
        super().__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'

        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, last_dim = channels

        self.conv1 = conv_nxn_bn(12, init_dim, stride=2)
        self.pf_conv = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1, bias=False)
        self.bayar_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        
        self.stem = nn.ModuleList([])
        self.stem.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.stem.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))

        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[3], channels[4], 2, expansion),
            MobileViTBlock(dims[0], depths[0], channels[5],
                           kernel_size, patch_size, int(dims[0] * 2)),            
        ]))
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[4], channels[4], 1, expansion),
            MobileViTBlock(dims[1], depths[1], channels[5],
                           kernel_size, patch_size, int(dims[1] * 4)),
        ]))
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[4], channels[4], 1, expansion),
            MobileViTBlock(dims[2], depths[2], channels[5],
                           kernel_size, patch_size, int(dims[2] * 4))
        ]))

        self.sig = nn.Sigmoid()
        self.rfb2_1 = nn.Conv2d(channels[4], channels[1], 1, padding=0)
        self.rfb3_1 = nn.Conv2d(channels[4], channels[1], 1, padding=0)
        self.rfb4_1 = nn.Conv2d(channels[4], channels[1], 1, padding=0)
        self.att_1 = nn.Conv2d(channels[1], 1, 3, padding=1)
        self.att_2 = nn.Conv2d(channels[1], 1, 3, padding=1)
        self.att_3 = nn.Conv2d(channels[1], 1, 3, padding=1)
        self.agg1 = aggregation_add(channels[1])

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.median = CustomizedConv(choice='median')
        self.pf_list = get_pf_list()
        self.reset_pf()

    def forward(self, x):
        b,c,H,W=x.shape
        
        self.bayar_conv.weight.data = constrained_weights(self.bayar_conv.weight.data)
        bayar_x = self.bayar_conv(x)
        pf_x = self.pf_conv(x)
        noise_x = torch.cat([pf_x,bayar_x],dim=1)

        x = self.conv1(noise_x)      
        
        for conv in self.stem:
            x = conv(x)
        features = []
        for conv,attn in self.trunk:
            x = conv(x)
            x = attn(x)
            features.append(x)

        h3,h4,h5=features
        x3 = self.rfb2_1(h3)
        att_3 = self.att_1(x3)
        x3 = x3 * self.sig(att_3)
        x4 = self.rfb3_1(h4)
        att_4 = self.att_2(x4)
        x4 = x4 * self.sig(att_4)
        x5 = self.rfb4_1(h5)
        att_5 = self.att_3(x5)
        
        x5 = x5 * self.sig(att_5)
        
        detection_map = self.agg1(x5, x4, x3)
        x = self.upsample(detection_map)
        x = self.median(x)
        x = F.interpolate(x, (H, W))
        x = nn.Sigmoid()(x)
        return  x, att_3, att_4, att_5, x5


    def reset_pf(self):
        for idx, pf in enumerate(self.pf_list):
            self.pf_conv.weight.data[idx, :, :, :] = pf


    
class MobileViTFPN(nn.Module):
    """MobileViT.
    Paper: https://arxiv.org/abs/2110.02178
    Based on: https://github.com/chinhsuanwu/mobilevit-pytorch
    """

    def __init__(
        self,
        image_size,
        dims,
        channels,
        num_classes,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3)
    ):
        super().__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'

        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, last_dim = channels

        self.conv1 = conv_nxn_bn(12, init_dim, stride=2)
        self.pf_conv = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1, bias=False)
        self.bayar_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        
        self.stem = nn.ModuleList([])
        self.stem.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.stem.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))

        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[3], channels[4], 2, expansion),
            MobileViTBlock(dims[0], depths[0], channels[5],
                           kernel_size, patch_size, int(dims[0] * 2)),            
        ]))
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[5], channels[6], 2, expansion),
            MobileViTBlock(dims[1], depths[1], channels[7],
                           kernel_size, patch_size, int(dims[1] * 4))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[7], channels[8], 2, expansion),
            MobileViTBlock(dims[2], depths[2], channels[9],
                           kernel_size, patch_size, int(dims[2] * 4))
        ]))


        self.rfb2_1 = nn.Conv2d(48, 16, 1, padding=0)
        self.rfb3_1 = nn.Conv2d(64, 16, 1, padding=0)
        self.rfb4_1 = nn.Conv2d(80, 16, 1, padding=0)
        self.att_1 = nn.Conv2d(16, 1, 3, padding=1)
        self.att_2 = nn.Conv2d(16, 1, 3, padding=1)
        self.att_3 = nn.Conv2d(16, 1, 3, padding=1)
        self.agg1 = aggregation_add2(16)
        self.sig = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.median = CustomizedConv(choice='median')
        self.pf_list = get_pf_list()
        self.reset_pf()

    def forward(self, x):
        b,c,H,W=x.shape
        
        self.bayar_conv.weight.data = constrained_weights(self.bayar_conv.weight.data)
        bayar_x = self.bayar_conv(x)
        pf_x = self.pf_conv(x)
        noise_x = torch.cat([pf_x,bayar_x],dim=1)

        x = self.conv1(noise_x)      
        
        for conv in self.stem:
            x = conv(x)
        features = []
        for conv,attn in self.trunk:
            x = conv(x)
            x = attn(x)
            features.append(x)

        h3,h4,h5=features
        x3 = self.rfb2_1(h3)
        att_3 = self.att_1(x3)
        x3 = x3 * self.sig(att_3)
        x4 = self.rfb3_1(h4)
        att_4 = self.att_2(x4)
        x4 = x4 * self.sig(att_4)
        x5 = self.rfb4_1(h5)
        att_5 = self.att_3(x5)
        x5 = x5 * self.sig(att_5)
        
        detection_map = self.agg1(x5, x4, x3)
        # print(x5.shape, x4.shape, x3.shape,detection_map.shape)
        x = self.upsample(detection_map)
        x = self.median(x)
        x = F.interpolate(x, (H, W))
        x = nn.Sigmoid()(x)
        return  x, self.upsample(att_3), self.upsample1(att_4), self.upsample2(att_5), h5


    def reset_pf(self):
        for idx, pf in enumerate(self.pf_list):
            self.pf_conv.weight.data[idx, :, :, :] = pf





class HOR(nn.Module):
    def __init__(self):
        super(HOR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=160, out_channels=160, kernel_size=1)
        self.high = nn.Conv2d(in_channels=160, out_channels=80, kernel_size=1)
        self.low = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=1)

        self.value = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x_latter, x):
        b, c, h, w = x.shape
        _, c_, _, _ = x.shape
        x_latter_ = self.high(self.conv1(x_latter)).reshape(b, c, h * w).contiguous()
        x_ = self.low(x).reshape(b, c_, h * w).permute(0, 2, 1).contiguous()

        p = torch.bmm(x_, x_latter_).contiguous()
        p = self.softmax(p).contiguous()

        e_ = torch.bmm(p, self.value(x).reshape(b, c, h * w).permute(0, 2, 1)).contiguous()
        e = e_ + x_
        e = e.permute(0, 2, 1).contiguous()
        
        return e, x_latter_

import torch.nn as nn
 
 
class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
 
    def forward(self, featmap):
        n, c, h, w = featmap.shape
        featmap = featmap.reshape((n, c, -1))
        featmap = featmap.softmax(dim=-1)
        return featmap
 
 
class CriterionCWD(nn.Module):
 
    def __init__(self, norm_type='none', divergence='mse', temperature=1.0):
 
        super(CriterionCWD, self).__init__()
 
        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type == 'spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x: x.view(x.size(0), x.size(1), -1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type
 
        self.temperature = 1.0
 
        # define loss function
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
        self.divergence = divergence
 
    def forward(self, preds_S, preds_T):
 
        n, c, h, w = preds_S.shape
        # import pdb;pdb.set_trace()
        if self.normalize is not None:
            norm_s = self.normalize(preds_S / self.temperature)
            norm_t = self.normalize(preds_T.detach() / self.temperature)
        else:
            norm_s = preds_S#[0]
            norm_t = preds_T#[0].detach()
 
        if self.divergence == 'kl':
            norm_s = norm_s.log()

        loss = self.criterion(norm_s, norm_t)
        
        # item_loss = [round(self.criterion(norm_t[0][0].log(),norm_t[0][i]).item(),4) for i in range(c)]
        # import pdb;pdb.set_trace()
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
            # loss /= n * h * w
        else:
            loss /= n * h * w
 
        return loss * (self.temperature ** 2)