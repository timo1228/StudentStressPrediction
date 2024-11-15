import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from abc import ABC, abstractmethod

class Dataset(ABC):
    @abstractmethod
    def train_and_test(self):
        pass

class StudentStressDataSet(Dataset):
    def __init__(self):
        data = pd.read_csv('./data/StressLevelDataset.csv')
        self.inited = False
        #init the data

        # Rename columns to remove spaces
        data.columns = data.columns.str.replace(' ', '')

        # Feature Engineer
        data.dropna(inplace=True)

        self.data = data

        # Train-Test Split and Scaling
        # Define target variable and features
        X = data.drop(['stress_level'], axis=1)  # Ensure 'Close' is dropped to create the feature set
        y = data['stress_level']  # Target variable is 'Close' price

        # Step 1: Replace infinite values with NaN
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Step 2: Check for NaN values and handle them
        # Using forward fill to handle NaN values (you can adjust this as needed)
        X.fillna(method='ffill', inplace=True)

        self.X = X
        self.y = y

        # Step 3: Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  #random split the data, not appropriate for time series data

        """
        # Step 4: Standardize the data
        # 标准化数据：x'=(x-u)/sigma, u is the mean, sigma is the standard deviation
        scaler = StandardScaler()
        # fit_transform() 方法用于在训练集 X_train 上计算标准化所需的均值和标准差，并使用这些统计量对 X_train 进行标准化。
        # fit() 计算训练集的均值和标准差。transform() 根据计算的均值和标准差将训练数据转换为标准化后的数据。
        X_train_scaled = scaler.fit_transform(X_train)
        # transform() 方法用于使用在 X_train 上计算的均值和标准差来标准化测试集 X_test。注意，这里没有调用 fit()，因为测试集的标准化应该基于训练集的统计信息，而不是重新计算。
        X_test_scaled = scaler.transform(X_test)
        print("Data scaling successful!")
        """

        self.X_train = X_train.to_numpy()
        self.X_test = X_test.to_numpy()
        self.y_train = y_train.to_numpy()
        self.y_test = y_test.to_numpy()

        self.inited = True
        print("initializing successfully!")


    def EDA(self): #展示数据的特征
        pass

    def train_and_test(self):
        if self.inited:
            return self.X_train, self.X_test, self.y_train, self.y_test
        else:
            raise RuntimeError('Dataset not yet initialized!')


def test():
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


if __name__ == "__main__":
    test()