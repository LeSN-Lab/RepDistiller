

from onnx import shape_inference
#필요한 import문
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import onnx
import torch.nn as nn
import torch.nn.init as init
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resnet import resnet8  # Assume the provided code is in 'model_code.py'

# 정의된 모델을 사용하여 초해상도 모델 생성
# torch_model = SuperResolutionNet(upscale_factor=3)


# Instantiate the model and load the pretrained weights (if available)
torch_model = resnet8(num_classes=10)

# 미리 학습된 가중치를 읽어옵니다

batch_size = 1    # 임의의 수

# 모델을 미리 학습된 가중치로 초기화합니다
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
# Assuming torch_model is already defined and is an instance of the model you wish to load the weights into
checkpoint = torch.load('/home/shared/AI2/AI2_MODELS/RepDistiller/save/models/resnet8_cifar10_lr_0.05_decay_0.0005_trial_0/resnet8_best.pth', map_location="cpu")
torch_model.load_state_dict(checkpoint['model'])

# 모델을 추론 모드로 전환합니다
torch_model.eval()

# 모델에 대한 입력값
x = torch.randn(1, 3, 32, 32, requires_grad=True)
torch_out = torch_model(x)

# 모델 변환
torch.onnx.export(torch_model,               # 실행될 모델
                  x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  "resnet8.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  )
                #   dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                #                 'output' : {0 : 'batch_size'}})
path = "./resnet8.onnx"
onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), path)
