import torch
import torch.nn as nn
import torch.nn.functional as F
import binarizePM1
import quantization

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

class Scale(nn.Module):
    def __init__(self, init_value=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class Quantization1:
    def __init__(self, method):
        self.method = method
    def applyQuantization(self, input):
        return self.method(input)
    
class Quantization2:
    def __init__(self, method, bits=None, unsign=1):
        self.method = method
        self.bits = bits
        self.unsigned = unsign # 0: use signed, 1: use unsigned
    def applyQuantization(self, input):
        return self.method(input,
         input.min().item(), input.max().item(), self.bits, self.unsigned)

binarizepm1 = Quantization1(binarizePM1.binarize)

q_train = True # quantization during training
q_eval = True # quantization during evaluation

# B1 and Q1 means quantization isn't separated into weight and input. Whereas B2 and Q2 means quantization is separated. 

class VGG3_Test_B1(nn.Module):
    def __init__(self, quantize_train=True, quantize_eval=True):
        super(VGG3_Test_B1, self).__init__()
        self.htanh = nn.Hardtanh()
        self.relu = nn.ReLU()
        self.name = "VGG3_Test_B1"
        self.q_train = quantize_train
        self.q_eval = quantize_eval


        self.conv1 = QuantizedConv2d(1, 64, kernel_size=3, padding=1, stride=1, quantization=binarizepm1, quantize_train=self.q_train, quantize_eval=self.q_eval)
        self.bn1 = nn.BatchNorm2d(64)
        self.qact1 = QuantizedActivation(quantization=binarizepm1, quantize_train=self.q_train, quantize_eval=self.q_eval)

        self.conv2 = QuantizedConv2d(64, 64, kernel_size=3, padding=1, stride=1, quantization=binarizepm1, quantize_train=self.q_train, quantize_eval=self.q_eval)
        self.bn2 = nn.BatchNorm2d(64)
        self.qact2 = QuantizedActivation(quantization=binarizepm1, quantize_train=self.q_train, quantize_eval=self.q_eval)

        self.fc1 = QuantizedLinear(7*7*64, 2048, quantization=binarizepm1, quantize_train=self.q_train, quantize_eval=self.q_eval)
        self.bn3 = nn.BatchNorm1d(2048)
        self.qact3 = QuantizedActivation(quantization=binarizepm1, quantize_train=self.q_train, quantize_eval=self.q_eval)

        self.fc2 = QuantizedLinear(2048, 10, quantization=binarizepm1, quantize_train=self.q_train, quantize_eval=self.q_eval)
        self.scale = Scale()

    def forward(self, x):
        #print(self)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = self.htanh(x)
        # x = self.relu(x)
        x = self.qact1(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.htanh(x)
        # x = self.relu(x)
        x = self.qact2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.htanh(x)
        # x = self.relu(x)
        x = self.qact3(x)

        x = self.fc2(x)
        x = self.scale(x)
        # output = F.log_softmax(x, dim=1)
        return x


class VGG3_Test_B2(nn.Module):
    def __init__(self, weightBits=binarizepm1, inputBits=binarizepm1, quantize_train=True, quantize_eval=True):
        super(VGG3_Test_B2, self).__init__()
        self.htanh = nn.Hardtanh()
        self.name = "VGG3_Test_B2"
        self.q_train = quantize_train
        self.q_eval = quantize_eval
        self.weight = weightBits
        self.input = inputBits

        #CNN
        # block 1
        self.conv1 = QuantizedConv2d(1, 64, kernel_size=3, padding=1, stride=1, quantization=self.weight, quantize_train=self.q_train, quantize_eval=self.q_eval)
        self.bn1 = nn.BatchNorm2d(64)
        self.qact1 = QuantizedActivation(quantization=self.input, quantize_train=self.q_train, quantize_eval=self.q_eval)

        # block 2
        self.conv2 = QuantizedConv2d(64, 64, kernel_size=3, padding=1, stride=1, quantization=self.weight, quantize_train=self.q_train, quantize_eval=self.q_eval)
        self.bn2 = nn.BatchNorm2d(64)
        self.qact2 = QuantizedActivation(quantization=self.input, quantize_train=self.q_train, quantize_eval=self.q_eval)

        # block 3
        self.fc1 = QuantizedLinear(7*7*64, 2048, quantization=self.weight, quantize_train=self.q_train, quantize_eval=self.q_eval)
        self.bn3 = nn.BatchNorm1d(2048)
        self.qact3 = QuantizedActivation(quantization=self.input, quantize_train=self.q_train, quantize_eval=self.q_eval)

        self.fc2 = QuantizedLinear(2048, 10, quantization=self.weight, quantize_train=self.q_train, quantize_eval=self.q_eval)
        self.scale = Scale()

    def forward(self, x):

        # block 1
        #x *= 100
        #x -= torch.mean(x)
        #print("\n\n\n --Binarization-- \n\n\nAt the very beginning: \n", x)
        x = self.conv1(x)
        #print("After convolution: \n", x)
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = self.htanh(x)
        #print("Before activation: \n", x)
        x = self.qact1(x)
        #print("After activation: \n", x)

        # block 2
        #print("\n\nBeginning of second block: \n", x)
        x = self.conv2(x)
        #print("After conv in second block: \n", x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.htanh(x)
        #print("Before activation in second block: \n", x)
        x = self.qact2(x)
        #print("After activation in second block: \n", x)

        # block 3
        x = torch.flatten(x, 1)
        #print("Before linear in third block: \n", x)
        x = self.fc1(x)
        #print("After linear in third block: \n", x)
        x = self.bn3(x)
        x = self.htanh(x)
        #print("Before quantization in third block: \n", x)
        x = self.qact3(x)
        #print("After quantization in third block: \n", x)

        #print("Before last linear: \n", x)
        x = self.fc2(x)
        #print("After last linear: \n", x)
        x = self.scale(x)
        #print("result: ", x)

        return x
    
class VGG3_Test_Q1(nn.Module):
    def __init__(self, quant_method=None, quantize_train=True, quantize_eval=True):
        super(VGG3_Test_Q1, self).__init__()
        self.relu = nn.ReLU()
        self.name = "VGG3_Test_Q1"
        self.q_train = quantize_train
        self.q_eval = quantize_eval
        self.quant = quant_method

        #CNN
        # block 1
        self.conv1 = QuantizedConv2d(1, 64, kernel_size=3, padding=1, stride=1, quantization=self.quant, quantize_train=self.q_train, quantize_eval=self.q_eval)
        self.bn1 = nn.BatchNorm2d(64)
        self.qact1 = QuantizedActivation(quantization=self.quant, quantize_train=self.q_train, quantize_eval=self.q_eval)

        # block 2
        self.conv2 = QuantizedConv2d(64, 64, kernel_size=3, padding=1, stride=1, quantization=self.quant, quantize_train=self.q_train, quantize_eval=self.q_eval)
        self.bn2 = nn.BatchNorm2d(64)
        self.qact2 = QuantizedActivation(quantization=self.quant, quantize_train=self.q_train, quantize_eval=self.q_eval)

        # block 3
        self.fc1 = QuantizedLinear(7*7*64, 2048, quantization=self.quant, quantize_train=self.q_train, quantize_eval=self.q_eval)
        self.bn3 = nn.BatchNorm1d(2048)
        self.qact3 = QuantizedActivation(quantization=self.quant, quantize_train=self.q_train, quantize_eval=self.q_eval)

        self.fc2 = QuantizedLinear(2048, 10, quantization=self.quant, quantize_train=self.q_train, quantize_eval=self.q_eval)
        self.scale = Scale()

    def forward(self, x):

        # block 1
        #x *= 100
        #x -= torch.mean(x)
        #print("\n\n\n --Quantization-- \n\n\nAt the very beginning: \n", x)
        x = self.conv1(x)
        #print("After convolution: \n", x)
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = self.relu(x)
        #print("Before activation: \n", x)
        #x = self.qact1(x)
        #print("After activation: \n", x)

        # block 2
        #print("\n\nBeginning of second block (repetition of same values): \n", x)
        x = self.conv2(x)
        #print("After conv in second block: \n", x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.relu(x)
        #print("Before activation in second block: \n", x)
        #x = self.qact2(x)
        #print("After activation in second block: \n", x)

        # block 3
        x = torch.flatten(x, 1)
        #print("Before linear in third block: \n", x)
        x = self.fc1(x)
        #print("After linear in third block: \n", x)
        x = self.bn3(x)
        x = self.relu(x)
        #print("Before quantization in third block: \n", x)
        #x = self.qact3(x)
        #print("After quantization in third block: \n", x)

        #print("Before last linear (repetition of same values): \n", x)
        x = self.fc2(x)
        #print("After last linear: \n", x)
        x = self.scale(x)
        #print("result: ", x)

        return x
    
class VGG3_Test_Q2(nn.Module):
    def __init__(self, weightBits=None, inputBits=None, quantize_train=True, quantize_eval=True):
        super(VGG3_Test_Q2, self).__init__()
        self.relu = nn.ReLU()
        self.name = "VGG3_Test_Q2"
        self.q_train = quantize_train
        self.q_eval = quantize_eval
        self.weight = weightBits
        self.input = inputBits

        #CNN
        # block 1
        self.conv1 = QuantizedConv2d(1, 64, kernel_size=3, padding=1, stride=1, quantization=self.weight, quantize_train=self.q_train, quantize_eval=self.q_eval)
        self.bn1 = nn.BatchNorm2d(64)
        self.qact1 = QuantizedActivation(quantization=self.input, quantize_train=self.q_train, quantize_eval=self.q_eval)

        # block 2
        self.conv2 = QuantizedConv2d(64, 64, kernel_size=3, padding=1, stride=1, quantization=self.weight, quantize_train=self.q_train, quantize_eval=self.q_eval)
        self.bn2 = nn.BatchNorm2d(64)
        self.qact2 = QuantizedActivation(quantization=self.input, quantize_train=self.q_train, quantize_eval=self.q_eval)

        # block 3
        self.fc1 = QuantizedLinear(7*7*64, 2048, quantization=self.weight, quantize_train=self.q_train, quantize_eval=self.q_eval)
        self.bn3 = nn.BatchNorm1d(2048)
        self.qact3 = QuantizedActivation(quantization=self.input, quantize_train=self.q_train, quantize_eval=self.q_eval)

        self.fc2 = QuantizedLinear(2048, 10, quantization=self.weight, quantize_train=self.q_train, quantize_eval=self.q_eval)
        self.scale = Scale()

    def forward(self, x):

        # block 1
        #x *= 100
        #x -= torch.mean(x)
        #print("\n\n\n --Quantization-- \n\n\nAt the very beginning: \n", x)
        x = self.conv1(x)
        #print("After convolution: \n", x)
        x = F.max_pool2d(x, 2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.qact1(x)
        #print("Before activation: \n", x)
        #print("After activation: \n", x)

        # block 2
        #print("\n\nBeginning of second block (repetition of same values): \n", x)
        x = self.conv2(x)
        #print("After conv in second block: \n", x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.qact2(x)
        #print("Before activation in second block: \n", x)
        #print("After activation in second block: \n", x)

        # block 3
        x = torch.flatten(x, 1)
        #print("Before linear in third block: \n", x)
        x = self.fc1(x)
        #print("After linear in third block: \n", x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.qact3(x)

        #print("Before quantization in third block: \n", x)
        #print("After quantization in third block: \n", x)

        #print("Before last linear (repetition of same values): \n", x)
        x = self.fc2(x)
        #print("After last linear: \n", x)
        x = self.scale(x)
        #print("result: ", x)

        return x