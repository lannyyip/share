import torch
from torch.autograd import Variable
import torch.onnx as torch_onnx
import os

from networks.ResNet import *

def main_resnet():
    #input_shape = (1, 1024, 650)
    input_shape = (3, 224, 224)
    _, h, w = input_shape

    root_dir = 'd:/trt_models'
    model_name = f'resnet'

    model_onnx_path = os.path.join(root_dir, f"{model_name}_output_{w}.onnx")

    resnet = resnet152(num_classes=1)
    #resnet.load_state_dict(torch.load('D:/trt_models/res152_panda_wr_cyclegan_oe_fulltrain_best.pth'))
    #model = RankingNetworkOutput(resnet)
    #model = model.cpu()

    model = resnet.cpu()
    
    dummy_input1 = Variable(torch.randn(16, *input_shape))
    model.train(False)
    
    inputs = ['input0']
    outputs = ['output0', 'output1']
    #dynamic_axes = {'input0': {0: 'batch', 3:'width'}, 
    dynamic_axes = {'input0': {}, 
                    #'input1': {0: 'batch'}, 
                    'output0':{}, 
                    'output1':{}
                    }
    out = torch.onnx.export(model, dummy_input1, model_onnx_path, input_names=inputs,
        output_names=outputs, dynamic_axes=dynamic_axes, opset_version=11)




if __name__=='__main__':
    main_resnet()
