import math

kernel_size = 3
pool_size = 2
stride = 2
numLayer = 5
width=320   # 352   # 320  1088
height=240  # 288  # 240  608
in_padding = 0
dilation = 1

outW, outH = 0, 0

### convolution
for x in range(numLayer):
    outW = math.floor((width+2*in_padding-kernel_size)/stride)+1
    outH = math.floor((height + 2 * in_padding - kernel_size) / stride) + 1
    width = outW
    height = outH
print("(widht , height) = ({},{})".format(outW, outH))


### deconvolution
in_padding = [0, 0, 0, 0, 0]
out_pedding = [0, 0, 0, 0, 0]


for x in range(numLayer):
    outW = (outW-1)*stride-2*in_padding[x]+dilation*(kernel_size-1) + out_pedding[x] + 1
    outH = (outH-1)*stride-2*in_padding[x]+dilation*(kernel_size-1) + out_pedding[x] + 1
print("(widht , height) = ({},{})".format(outW, outH))