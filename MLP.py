import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score


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

class BestMLPModel(object):
    def __init__(self, device, filepath='./final_mlp_model.pth'):
        if device == 'gpu':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device("cpu")

        # 加载模型
        model = MLPModel()
        model.load_state_dict(torch.load(filepath))
        model.to(device)
        model.eval()  # 设置为评估模式

        self.model = model
        self.device = device

    def score_matrix(self, X_test):
        """

        :param X_test: numpy array
        :return:
        """
        X_test = torch.from_numpy(X_test)
        with torch.no_grad():  # 不构建计算图
            inputs = X_test.to(self.device)
            inputs = inputs.float()  # 转换为浮动类型，linear层必须要求输入为float
            y_pred = self.model(inputs)
            return y_pred.detach().numpy()

def load_and_evaluate_BestMLP_model(device):
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    model = BestMLPModel(device, filepath='./final_mlp_model.pth')
    score_function = model.score_matrix(X_test)
    predicted = np.argmax(score_function, 1)
    correct = (predicted == y_test).sum()
    total = y_test.shape[0]
    print("Accuracy of the loaded model on test set: %.2f%%" % (100 * correct / total))

def get_model_merged_roc_curve_parameters(X_test, y_test ,model, num_classes=3):
    # 获取模型的预测概率
    y_score = model.score_matrix(X_test)

    # One-hot encode the true labels for multiclass ROC
    y_test_one_hot = np.zeros((len(y_test), num_classes))
    for i, label in enumerate(y_test):
        y_test_one_hot[i, label] = 1

    # Calculate macro-average ROC curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute ROC curve and ROC area for each class
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_score[:, i])
        roc_auc[i] = roc_auc_score(y_test_one_hot[:, i], y_score[:, i])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Average it and compute the macro-average ROC curve
    mean_tpr /= num_classes

    # Compute macro-average AUC
    macro_auc = np.mean(list(roc_auc.values()))  # 直接计算各个类别的AUC的平均值

    return all_fpr, mean_tpr, macro_auc


def plot_roc_auc():
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()
    model = BestMLPModel('cpu', filepath='./final_mlp_model.pth')
    all_fpr, mean_tpr, macro_auc = get_model_merged_roc_curve_parameters(X_test, y_test, model, num_classes=3)

    # 绘制合并的 ROC 曲线
    plt.plot(all_fpr, mean_tpr, label=f"MLP (Macro AUC={macro_auc:.2f})")

    # 绘制随机猜测的对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Chance Level (AUC=0.50)")

    # 设置图表
    plt.title("Merged ROC Curve for MLP (Combined Classes)", fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def get_MLP_merged_roc_curve_parameters():
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()
    model = BestMLPModel('cpu', filepath='./final_mlp_model.pth')
    return get_model_merged_roc_curve_parameters(X_test, y_test, model, num_classes=3)

if __name__ == '__main__':
    #train(100, 'gpu')
    #load_and_evaluate_BestMLP_model('cpu')
    plot_roc_auc()