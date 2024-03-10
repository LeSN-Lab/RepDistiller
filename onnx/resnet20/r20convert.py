import onnx
import torch
import pandas as pd
from onnx2torch import convert
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from AI2_MODELS.RepDistiller.onnx.resnet import resnet20, resnet32x4  # Assume the provided code is in 'model_code.py'
from cifar100 import get_cifar100_dataloaders

from PIL import Image
# 1단계: 데이터셋 로딩 및 전처리

class CIFAR100Dataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #normalize the image about cifar10 mean and standard deviation
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        label = self.data_frame.iloc[idx, 0]
        image = self.data_frame.iloc[idx, 1:].values.astype(np.uint8).reshape(32, 32, 3)
        image = Image.fromarray(image)  # PIL 이미지로 변환
        image = self.transform(image)
        image = image.unsqueeze(0)  # 배치 차원 추가
        return image, label

# CSV 파일에서 CIFAR-100 데이터셋 로드
# test_loader = CIFAR100Dataset(csv_file='/home/shared/AI2/ERAN/data/cifar100_test.csv')

#새로 만든 cifar100csv
test_loader = CIFAR100Dataset(csv_file='/home/shared/AI2/RepDistiller/onnx/resnet20/data/cifar100_samples.csv')

# 데이터로더 정의
# cifar100_data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
# train_loader, test_loader = get_cifar100_dataloaders(batch_size=2, num_workers=8, is_instance=False)




#convert onnx model to torch model
def convertModel(onnx_model_path):
    # Load the ONNX model
    onnx_model = onnx.load(onnx_model_path)
    # Convert the ONNX model to a PyTorch model
    pytorch_model = convert(onnx_model)
    return pytorch_model

#run the model with pandas dataframe
def run_model(model):
    #load dataset with batch size 1 from pandas dataframe
    
    #run the model with the dataset
    
    sum = 0
    for i, (inputs, labels) in enumerate(test_loader):
        
        
        outputs = model(inputs)
        #get max label from outputs
        _, predicted = torch.max(outputs, 1)
        #print accuracy
        if predicted == labels:
            sum += 1
    print(f'Accuracy: {sum/len(test_loader)}, Total: {len(test_loader)}')
        
        
        
        
    
    

def onnxModelAccuracy():
  #convert onnx model to torch model
  model = convertModel('/home/shared/AI2/RepDistiller/onnx/resnet20/resnet20.onnx')
  model.eval()
  
  #run the model with pandas dataframe
  run_model(model)


def torchModelAccuracy():
    # Instantiate the model and load the pretrained weights (if available)
    torch_model = resnet20(num_classes=100)
    # torch_model = resnet32x4(num_classes=100)

    # 미리 학습된 가중치를 읽어옵니다

    batch_size = 1    # 임의의 수

    # 모델을 미리 학습된 가중치로 초기화합니다
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    # Assuming torch_model is already defined and is an instance of the model you wish to load the weights into
    checkpoint = torch.load('/home/shared/AI2/RepDistiller/save/student_model/S:resnet20_T:resnet56_cifar100_kd_r:0.1_a:0.9_b:0.0_1/resnet20_best.pth', map_location="cpu")
    # checkpoint = torch.load('/home/shared/AI2/RepDistiller/save/models/resnet32x4_vanilla/ckpt_epoch_240.pth', map_location="cpu")
    torch_model.load_state_dict(checkpoint['model'])
    torch_model.eval()
    run_model(torch_model)

if __name__ == "__main__":
    # train_loader, test_loader = get_cifar100_dataloaders(batch_size=2, num_workers=8, is_instance=False)
    # onnxModelAccuracy()
    torchModelAccuracy()