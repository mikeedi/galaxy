import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """Encoder implementation"""

    def __init__(self, code_size):
        super(Encoder, self).__init__()
        self.code_size = code_size

        self.conv_1 = self.convolution_block(3, 16, kernel_size=5, padding=2)
        self.conv_2 = self.convolution_block(16, 32, kernel_size=3, padding=1)
        self.conv_3 = self.convolution_block(32, 64, kernel_size=3, padding=1)
        self.conv_4 = self.convolution_block(64, 128, kernel_size=3, padding=1)
        self.conv_5 = self.convolution_block(128, 256, kernel_size=3, padding=1)
        self.conv_6 = self.convolution_block(256, 512, kernel_size=3, padding=1)
        self.GAP = nn.AvgPool2d(kernel_size=4)
        self.dense = self.dense_block(512, self.code_size)

    def convolution_block(self, input_size, output_size, kernel_size, mp_kernel_size=2, 
                            padding=1, dilation=1, stride=1, bias=True):
        """
        input_size and output_size :: num of channel 
        """

        return nn.Sequential(
                    nn.Conv2d(input_size, output_size, kernel_size=kernel_size, 
                        padding=padding, dilation=dilation, stride=stride, bias=bias),
                    nn.BatchNorm2d(output_size),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=mp_kernel_size)
                )

    def dense_block(self, input_unit, output_unit):
        """
        dense block with tanh activation
        """
        return nn.Sequential(
                    nn.Linear(input_unit, output_unit),
                    nn.Tanh()
                )


    def convolution_block2(self, x, input_size, output_size, kernel_size, mp_kernel_size=2, 
                            padding=1, dilation=1, stride=1, bias=True):
        """
        input_size and output_size :: num of channel 
        """

        return nn.Sequential(
                    nn.Conv2d(input_size, output_size, kernel_size=kernel_size, 
                        padding=padding, dilation=dilation, stride=stride, bias=bias),
                    nn.ELU(),
                    nn.BatchNorm2d(output_size),
                    nn.MaxPool2d(kernel_size=mp_kernel_size)
                )(x)

    def forward(self, x):

        # 6 convolution blocks on tensor with shape 256x256x3
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        
        # global average pooling on tensor with shape 4x4x512
        x = self.GAP(x)
        
        # flatten tensor with shape 1x1x1024
        x = x.view(x.size(0), -1)

        # last fc layer
        x = self.dense(x)

        return x


class Decoder(nn.Module):
    """Decoder implementation"""
   
    def __init__(self, code_size):
        super(Decoder, self).__init__()
        self.code_size = code_size

        self.dense = self.dense_block(self.code_size, 512)
        self.deconv_1 = self.deconvolution_block(512, 512, kernel_size=4, padding=0, stride=1, output_padding=0)
        self.deconv_2 = self.deconvolution_block(512, 256, kernel_size=3, padding=0, stride=2, output_padding=0)
        self.deconv_3 = self.deconvolution_block(256, 256, kernel_size=3, padding=0, stride=2, output_padding=0)
        self.deconv_4 = self.deconvolution_block(256, 128, kernel_size=3, padding=0, stride=2, output_padding=0)
        self.deconv_5 = self.deconvolution_block(128, 64, kernel_size=3, padding=0, stride=2, output_padding=0)
        self.deconv_6 = self.deconvolution_block(64, 32, kernel_size=3, padding=0, stride=2, output_padding=0)
        self.deconv_7 = self.deconvolution_block(32, 16, kernel_size=3, padding=0, stride=2, output_padding=0)

        self.conv_out = nn.Conv2d(16, 3, kernel_size=5, padding=2)

    def deconvolution_block(self, input_size, output_size, kernel_size, 
                            padding=0, output_padding=0, dilation=1, stride=1, bias=True):
        """
        input_size and output_size :: num of channel 
        """

        return nn.Sequential(
                    nn.ConvTranspose2d(input_size, output_size, kernel_size=kernel_size,
                     padding=padding, output_padding=output_padding, 
                     stride=stride, dilation=dilation),
                    nn.BatchNorm2d(output_size),
                    nn.ELU(),
                )

    def dense_block(self, input_unit, output_unit):
        """
        dense block with elu activation
        """
        return nn.Sequential(
                    nn.Linear(input_unit, output_unit),
                    nn.BatchNorm1d(output_unit),
                    nn.ELU()
                )

    def forward(self, x):
        x = self.dense(x)
        x = x.view(x.size(0), 512, 1, 1)
 
        x = self.deconv_1(x) 
        x = self.deconv_2(x) 
        x = self.deconv_3(x) 
        x = self.deconv_4(x) 
        x = self.deconv_5(x) 
        x = self.deconv_6(x)
        x = self.deconv_7(x)

        x = F.interpolate(x, size=(256,256))

        x = self.conv_out(x)

        return x

class Autoencoder(nn.Module):
    """Autoencoder implementation"""

    def __init__(self, code_size):
        super(Autoencoder, self).__init__()
        self.code_size = code_size
        self.encoder = Encoder(code_size)
        self.decoder = Decoder(code_size)

    def forward(self, x):
        code = self.encoder(x)
        decode = self.decoder(code)
        return decode  

if __name__ == '__main__':

    from torchsummary import summary

    au = Autoencoder(128)

    # summary(input_size=(3, 256, 256), model=Encoder(128))
    # summary(input_size=(128,), model=Decoder(128))
    
    summary(input_size=(3, 256, 256), model=au)
