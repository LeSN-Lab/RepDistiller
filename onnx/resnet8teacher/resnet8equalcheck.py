from onnx import shape_inference
import onnx




#필요한 import문
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

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
checkpoint = torch.load('/home/shared/AI2/AI2_MODELS/RepDistiller/save/models/resnet8_cifar10_lr_0.05_decay_0.0005_trial_0/resnet8_best.pth', map_location)
torch_model.load_state_dict(checkpoint['model'])

# 모델을 추론 모드로 전환합니다
torch_model.eval()

# 모델에 대한 입력값
x = torch.randn(1, 3, 32, 32, requires_grad=True)
torch_out = torch_model(x)
#layer 간의 입출력 크기를 확인하기 위해서 저장된 ONNX를 다시 불러와서 아래와 같은 방식으로 shape 정보를 저장하는 과정이 필요






def saveShape():
    path = "./resnet8.onnx"
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), path)
  
def checkValidSchema(): 
    path = "./resnet8.onnx"
    model = onnx.load(path)
    return onnx.checker.check_model(model)

def compareOutput():
  import onnxruntime

  ort_session = onnxruntime.InferenceSession("resnet8.onnx")

  def to_numpy(tensor):
      return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

  # ONNX 런타임에서 계산된 결과값
  ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
  ort_outs = ort_session.run(None, ort_inputs)

  # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
  np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

  print("Exported model has been tested with ONNXRuntime, and the result looks good!")

if __name__ == "__main__":
    
    compareOutput()