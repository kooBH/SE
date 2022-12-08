import torch
import torch.nn as nn

class ComplexConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, complex=False, padding_mode="zeros"):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding
            
        if complex:
            conv = ComplexConv2d
            bn = ComplexBatchNorm2d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d

        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding,padding=(0, 0), complex=False):
        super().__init__()
        if complex:
            tconv = ComplexConvTranspose2d
            bn = ComplexBatchNorm2d
        else:
            tconv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d
        
        self.transconv = tconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=output_padding,padding=padding)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DCUNET(nn.Module):
    def __init__(self, input_channels=1,
                 complex=True,
                 model_complexity=45,
                 model_depth=20,
                 padding_mode="zeros",
                 dropout=0.0):
        super().__init__()

        #if complex:
        #    model_complexity = int(model_complexity // 1.414)

        if not complex:
            input_channels *=2
        else  :
            model_complexity = int(model_complexity // 1.414)

        self.set_size(model_complexity=model_complexity, input_channels=input_channels, model_depth=model_depth)
        self.encoders = []
        self.model_length = model_depth // 2
        self.dropout = dropout

        for i in range(self.model_length):
            module = Encoder(self.enc_channels[i], self.enc_channels[i + 1], kernel_size=self.enc_kernel_sizes[i],
                             stride=self.enc_strides[i], padding=self.enc_paddings[i], complex=complex, padding_mode=padding_mode)
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        self.decoders = []

        for i in range(self.model_length):
            module = Decoder(self.dec_channels[i] + self.enc_channels[self.model_length - i], self.dec_channels[i + 1], kernel_size=self.dec_kernel_sizes[i],
                             stride=self.dec_strides[i], padding=self.dec_paddings[i], output_padding=self.dec_output_paddings[i],complex=complex)
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

        if complex:
            conv = ComplexConv2d
            linear = conv(self.dec_channels[-1], 1, 1)
        else:
            conv = nn.Conv2d
            linear = conv(self.dec_channels[-1], 2, 1)


        self.add_module("linear", linear)
        self.complex = complex
        self.padding_mode = padding_mode

        self.dr = nn.Dropout(self.dropout)

        self.decoders = nn.ModuleList(self.decoders)
        self.encoders = nn.ModuleList(self.encoders)


    def forward(self, input):        
        # ipnut : [ Batch Channel Freq Time 2]

        if self.complex:
            x = input
        else:
            #raise Exception('Unsupported type for input')
            tmp = input.permute(0,1,4,2,3)
            x = torch.reshape(tmp,(tmp.shape[0],tmp.shape[1]*tmp.shape[2],tmp.shape[3],tmp.shape[4]))
            pass

        # Encoders
        x_skip = []
        for i, encoder in enumerate(self.encoders):
            x_skip.append(x)
            x = encoder(x)
            x = self.dr(x)
           # print("x{}".format(i), x.shape)
        # x_skip : x0=input x1 ... x9

        #print("fully encoded ",x.shape)
        p = x
        
        # Decoders
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            p = self.dr(p)
            if i == self.model_length - 1:
                break
            #print(f"p{i}, {p.shape} + x{self.model_length - 1 - i}, {x_skip[self.model_length - 1 -i].shape}, padding {self.dec_paddings[i]}")
            
            p = torch.cat([p, x_skip[self.model_length - 1 - i]], dim=1)

        #:print(p.shape)
        mask = self.linear(p)
        mask = torch.tanh(mask)
        mask = torch.squeeze(mask,1)
        
        #return real_spec*mask[:,:,:,0], imag_spec*mask[:,:,:,1]

        if self.complex :
            return mask[:,:,:,0], mask[:,:,:,1]
        else :
            return torch.squeeze(mask[:,0,:,:]), torch.squeeze(mask[:,1,:,:])

    def set_size(self, model_complexity, model_depth=20, input_channels=1):
        if model_depth == 10:
            self.enc_channels = [input_channels,
                                 model_complexity,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 ]
            self.enc_kernel_sizes = [(7, 5),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]
            self.enc_strides = [(2, 2),
                                (2, 2),
                                (2, 2),
                                (2, 2),
                                (2, 1)]
            self.enc_paddings = [(2, 1),
                                 None,
                                 None,
                                 None,
                                 None]

            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2]

            self.dec_kernel_sizes = [(4, 3),
                                     (4, 4),
                                     (6, 4),
                                     (6, 4),
                                     (7, 5)]

            self.dec_strides = [(2, 1),
                                (2, 2),
                                (2, 2),
                                (2, 2),
                                (2, 2)]

            self.dec_paddings = [(1, 1),
                                 (1, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1)]

        elif model_depth == 20:
            self.enc_channels = [input_channels,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 128]

            self.enc_kernel_sizes = [(7, 1),
                                     (1, 7),
                                     (7, 5),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]

            self.enc_strides = [(1, 1),
                                (1, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1)]

            self.enc_paddings = [(3, 0),
                                 (0, 3),
                                 (3, 2),
                                 (3, 2),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),]
                              
                                 

            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2]

            self.dec_kernel_sizes = [(5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3), 
                                     (7, 5), 
                                     (7, 5), 
                                     (1, 7),
                                     (7, 1)]

            self.dec_strides = [(2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (1, 1),
                                (1, 1)]

            self.dec_paddings = [(2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (3, 2),
                                 (3, 2),
                                 (0, 3),
                                 (3, 0)]
            self.dec_output_paddings = [(0,0),
                                        (0,1),
                                        (0,0),
                                        (0,1),
                                        (0,0),
                                        (0,1),
                                        (0,0),
                                        (0,1),
                                        (0,0),
                                        (0,0)]
        else:
            raise ValueError("Unknown model depth : {}".format(model_depth))