import matplotlib.pyplot as plt

import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split

from DataSource import StudentStressDataSet

def split_train_and_test():
    data = pd.read_csv('./data/StressLevelDataset.csv')
    # 按 8:2 比例分割数据
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # 将数据分别保存为 train.csv 和 test.csv
    train_data.to_csv('./data/train.csv', index=False)
    test_data.to_csv('./data/test.csv', index=False)


class StressLevelDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, delimiter=',', quotechar='"')
        numpy_data = self.df.to_numpy(dtype='float32') #转numpy
        self.features = torch.from_numpy(numpy_data[:, 0:-1])
        self.labels = torch.from_numpy(numpy_data[:, -1:])
        self.len = numpy_data.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.len

class MLPModel(torch.nn.Module): #One Layer Softmax Model，
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(20, 40)
        self.linear2 = torch.nn.Linear(40, 20)
        self.linear3 = torch.nn.Linear(20, 10)
        self.linear4 = torch.nn.Linear(10, 5)
        self.linear5 = torch.nn.Linear(5, 3)

    def forward(self, x): #x.shape = (N, 20)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return self.linear5(x)

def train(epoch_size, device):
    # 0 Pre-define parameters
    if device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device("cpu")
    batch_size = 256
    # 1 Prepare data
    train_dataset = StressLevelDataset('./data/train.csv')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = StressLevelDataset('./data/test.csv')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 2 Prepare Model
    model = MLPModel()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()  # 多分类交叉熵损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    loss_list = list()
    test_acc_list = list()

    # 3 train
    def batch_train(epoch_num):  # 在一个epoch的测试
        accumulated_loss, num = 0, 0
        for batch_id, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # 让数据加载到相应的cpu或gpu内存上
            labels = labels.view(inputs.size(0)) #criterion接受的labels是要求一维的
            labels = labels.to(torch.int64) #nvidia需要labels为int，不是float，否则会报错
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)

            accumulated_loss += loss.item()
            num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("[%d, %5d] loss: %.3f" % (epoch_num+1, batch_id + 1, loss.item()))
        loss_list.append(accumulated_loss / num)

    # 4 test and evaluate
    def test():
        correct = 0  # 正确预测个数
        total = 0  # 样本总数
        with torch.no_grad():  # 不要构成计算图
            for (inputs, labels) in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.view(inputs.size(0))
                labels = labels.to(torch.int64) #nvidia需要labels为int，不是float，否则会报错
                y_pred = model(inputs)
                _, predicted = torch.max(y_pred, 1)  # 沿着第一个纬度（行）取max，_是最大值，predicted是下标
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            return 100 * correct / total
            #print("Accuracy of the network on test set: %d %%" % (100 * correct / total))

    # train and test
    for epoch in range(epoch_size):
        batch_train(epoch)
        test_acc = test()
        test_acc_list.append(test_acc)
        if epoch % (epoch_size // 10) == 0 or epoch == epoch_size - 1:  # 进行到1/10进行一次test
            print("--------epcho {} testing on test set--------".format(epoch))
            #test()
            print("Accuracy of the network on test set: %d %%" % test_acc)
            print("--------testing finished--------")

    torch.save(model.state_dict(), './final_mlp_model.pth')
    print("Model saved to ./final_mlp_model.pth")

    #plot
    # 创建折线图
    plt.plot([i for i in range(1,epoch_size+1)], loss_list, marker='o', linestyle='-', color='r', label='Empirical Risk')

    # 添加标题和标签
    plt.title("Empirical Risk during training")  # 标题
    plt.xlabel("Epoch")  # x轴标签
    plt.ylabel("Empirical Risk")  # y轴标签
    # 显示图例
    plt.legend()

    plt.show()

    plt.plot([i for i in range(1, epoch_size+1)], test_acc_list, marker='o', linestyle='-', color='r', label='Test Accuracy')

    # 添加标题和标签
    plt.title("Test Accuracy during training")  # 标题
    plt.xlabel("Epoch")  # x轴标签
    plt.ylabel("Accuracy on Test Set")  # y轴标签
    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()


def load_and_evaluate_BestMLP_model(device):
    if device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device("cpu")

    filepath = './final_mlp_model.pth'
    # 加载模型
    model = MLPModel()
    model.load_state_dict(torch.load(filepath))
    model.to(device)
    model.eval()  # 设置为评估模式

    # 评估
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()
    X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)

    with torch.no_grad():  # 不构建计算图
        inputs, labels = X_test.to(device), y_test.to(device)
        inputs = inputs.float()  # 转换为浮动类型，linear层必须要求输入为float
        labels = labels.view(inputs.size(0))
        labels = labels.to(torch.int64)
        y_pred = model(inputs)
        #scores for each class = y_pred.detach().numpy()
        _, predicted = torch.max(y_pred, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)

    print("Accuracy of the loaded model on test set: %.2f%%" % (100 * correct / total))


if __name__ == '__main__':
    #train(100, 'gpu')
    load_and_evaluate_BestMLP_model('gpu')