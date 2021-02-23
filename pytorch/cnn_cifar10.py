import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms

def get_device(gpu_name):
    device = gpu_name if torch.cuda.is_available() else 'cpu'
    return device


def set_seed(seed_value, device):
    torch.manual_seed(seed_value)
    if 'cuda' in device:
        torch.cuda.manual_seed_all(seed_value)


def get_loaders(file_path, transform, batch_size, shuffle):
    train_dataset = dataset.CIFAR10(root=file_path, train=True, transform=transform, download=True)
    test_dataset = dataset.CIFAR10(root=file_path, train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader

## make CNN model
class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()

        ## shape input = (batch_size, 3, 32, 32)
        ##       Conv  = (batch_size, 32, 32, 32)
        ##       Pool  = (batch_size, 32, 16, 16)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        ## shape input = (batch_size, 32, 16, 16)
        ##       Conv  = (batch_size, 64, 16, 16)
        ##       Pool  = (batch_size, 64, 8, 8)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        ## shape input = (batch_size, 64, 8, 8)
        ##       Conv  = (batch_size, 128, 8, 8)
        ##       Pool  = (batch_size, 128, 4, 4)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.dropout_prob = 0.5
        ## shape input = (batch_size, 128, 4, 4)
        ##       FC1   = (batch_size, 128)
        ##       FC2   = (batch_size, 10)
        self.layer4 = torch.nn.Sequential(
            nn.Linear(4 * 4 * 128, 128, bias=True),
            nn.ReLU(),
            nn.Dropout(p = 1 - self.dropout_prob),
            nn.Linear(128, 10, bias=True))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1) # batch_size 차원을 제외한 데이터를 1차원으로 변환
        out = self.layer4(out)
        return out


def train(device, model, epochs, train_loader, criterion, optimizer, batch_size, printable=True):
    model.train()
    for epoch in range(1, epochs+1):
        train_loss = 0
        train_accuracy = 0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)

            output = model(data)    # batch_size 차원의 데이터를 model에 투입하여 훈련
            loss = criterion(output, label)   # 손실 함수의 값 계산
            optimizer.zero_grad()   # pytorch의 변화도를 0으로 설정한다. (기존 변화도에 누적되는 방식이기 때문)
            loss.backward()         # 역전파를 수행하여 가중치의 변화량을 계산
            optimizer.step()        # 가중치의 변화량을 적용하여 가중치 업데이트

            train_loss += loss.item()
            predict = output.max(1)[1]  # (batch_size, classes) 데이터에서 가장 큰 값을 가진 class 노드의 index 추출
            train_accuracy += predict.eq(label).sum().item()   # batch_size 데이터 중 정답과 일치한 개수
        else:
            train_loss /= len(train_loader)  # len(train_loader) = (전체 훈련 데이터 수 / batch_size)
            train_accuracy *= (100 / len(train_loader.dataset))  # len(train_loader.dataset) = 전체 훈련 데이터 수
            if printable:
                print('Train Result Epoch = {}, Loss = {:.4f}, Accuracy = {:.4f}%)'.format(epoch, train_loss, train_accuracy))
    else:
        return train_loss, train_accuracy


def test(device, model, test_loader, criterion, printable=True):
    model.eval()            # 평가 모드 적용 - 드롭아웃, 배치정규화 비활성화
    with torch.no_grad():   # 역전파 비활성화 -> 메모리 절약 -> 연산 속도 상승
        test_loss = 0
        test_accuracy = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += criterion(output, target).item()
            predict = output.max(1)[1]  # (batch_size, classes) 데이터에서 가장 큰 값을 가진 class 노드의 index 추출
            test_accuracy += predict.eq(target).sum().item()   # batch_size 데이터 중 정답과 일치한 개수
        else:
            test_loss /= len(test_loader)  # len(test_loader) = (전체 시험 데이터 수 / batch_size)
            test_accuracy *= (100 / len(test_loader.dataset)) # len(test_loader.dataset) = 전체 시험 데이터 수
            if printable:
                print('Test Result: Loss = {:.4f}, Accuracy = {:.4f}%)'.format(test_loss, test_accuracy))
        return test_loss, test_accuracy


def run(parallel_train=False, gpu_name="cuda", seed_value=1216, file_path="./downloads/", batch_size=100, shuffle=False, learning_rate=0.001, training_epochs=100, printable=True):
    
    # gpu 또는 cpu 장치 설정
    device = get_device(gpu_name)

    # 랜덤 시드 설정
    set_seed(seed_value, device)

    transform = transforms.Compose([
        transforms.ToTensor(),  # 데이터 타입을 Tensor로 변형
        # 정규화 값의 출처 = https://www.programmersought.com/article/5163444351/
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 데이터의 (평균 데이터 값, 표준 편차 값) 설정
    ])
    train_loader, test_loader = get_loaders(file_path, transform, batch_size, shuffle)

    model = CNN().to(device)  # 모델 생성
    if parallel_train:
        model = nn.DataParallel(model)  # 데이터 병렬 처리                             
    criterion = nn.CrossEntropyLoss().to(device)  # 손실 함수 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 최적화 설정

    # 훈련
    lastest_train_loss, lastest_train_accuracy = train(device, model, training_epochs, train_loader, criterion, optimizer, batch_size, printable)
    if not printable:
        print('Lastest Train Result: Loss = {:.4f}, Accuracy = {:.4f}%)'.format(lastest_train_loss, lastest_train_accuracy))

    # 시험
    test_loss, test_accuracy = test(device, model, test_loader, criterion, printable)
    if not printable:
        print('Test Result: Loss = {:.4f}, Accuracy = {:.4f}%)'.format(test_loss, test_accuracy))


if __name__ == "__main__":
    parallel_train = False    # GPU 병렬 사용 여부
    gpu_name = 'cuda' if parallel_train else 'cuda:0'
    seed_value = 1216         # 랜덤 시드 값
    file_path="./downloads/"  # 데이터 파일 저장 경로
    batch_size = 10000        # 배치 크기
    if parallel_train and torch.cuda.available():
        batch_size *= torch.cuda.device_count()
    shuffle = True  # DataLoader의 데이터 shuffle
    learning_rate = 0.0005    # 학습률 설정
    training_epochs = 5       # 훈련 횟수 설정
    printable = False         # train, test 함수에서 출력 활성화

    # 동작
    run(parallel_train, gpu_name, seed_value, file_path, batch_size, shuffle, learning_rate, training_epochs, printable)