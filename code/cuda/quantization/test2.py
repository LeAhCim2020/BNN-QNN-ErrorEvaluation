import torch
import quantization
import binarizePM1

class Quantization2:
    def __init__(self, method, bits=None, unsign=1):
        self.method = method
        self.bits = bits
        self.unsigned = unsign # 0: use signed, 1: use unsigned
    def applyQuantization(self, input):
        return self.method(input,
         input.min().item(), input.max().item(), self.bits, self.unsigned)

tensor = torch.rand(size=(2,2,3,3), dtype=torch.float).cuda()
tensor *= 100
tensor -= torch.mean(tensor)
print("random tensor", tensor)
quant1 = Quantization2(quantization.quantize, 4, 1)
quant2 = Quantization2(quantization.quantize, 8, 1)
a = quant1.applyQuantization(tensor)
print("Applying quant1 on tensor, gets: ", a)
b = quant2.applyQuantization(tensor)
print("Applying quant2 on tensor, gets: ", b)
print("Reapplying quant1 on tensor, with naive approach, gets: ", a)
a = quant1.applyQuantization(tensor)
print("On the other hand, after recalling quant1 on tensor, gets: ", a, "\nSo, we get the desired result.")
