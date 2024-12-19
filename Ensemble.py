from numpy.ma.core import multiply
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from DataSource import StudentStressDataSet

"""
    using GridSearch to find optimal parameters
"""
def Best_RFParam_Search():
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    # 定义随机森林模型
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 500, 1000, 1500, 2000],
        'max_depth': [3, 4, 5, 7, 8, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        #'max_features': ['sqrt', 'log2', None],
        #'bootstrap': [True, False]
    }

    # 使用 GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

    # 运行网格搜索
    grid_search.fit(X_train, y_train)

    # 打印最佳参数
    print(f"Best Parameters: {grid_search.best_params_}")

    # 使用最佳参数进行预测
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # 打印准确率
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    #Best Parameters: {'bootstrap': True, 'max_depth': 4, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 10000}
    #Accuracy: 0.90

def Best_RandomForest_model():
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    # 创建并训练 Random Forest 分类器,max_feature是每个base learner用于训练的的feature数量（为了减少模型见的correlation）
    rf_clf = RandomForestClassifier(n_estimators=1000, max_depth=4, min_samples_leaf=1, min_samples_split=10, max_features='sqrt', bootstrap=True, criterion='entropy', random_state=42)
    rf_clf.fit(X_train, y_train)

    # 进行预测
    y_pred = rf_clf.predict(X_test)

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best Random Forest Accuracy: {accuracy:.6f}") #89.54545454545455%
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 查看特征重要性
    feature_labels = ['anxiety_level', 'self_esteem', 'mental_health_history', 'depression', 'headache', 'blood_pressure',
                      'sleep_quality', 'breathing_problem', 'noise_level', 'living_conditions', 'safety', 'basic_needs',
                      'academic_performance', 'study_load', 'teacher_student_relationship', 'future_career_concerns',
                      'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']
    #通过评估每个特征对模型预测性能的贡献来计算。这是通过所有树（1000个）的加权平均得到的
    feature_importances = rf_clf.feature_importances_
    # 对特征重要性排序
    sorted_indices = np.argsort(feature_importances)[::-1]  # 按重要性从高到低排序
    sorted_feature_labels = [feature_labels[i] for i in sorted_indices]
    sorted_feature_importances = feature_importances[sorted_indices]

    # 绘制排序后的特征重要性图
    plt.figure(figsize=(15, 8))
    plt.barh(sorted_feature_labels, sorted_feature_importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance in Random Forest (Sorted)')
    plt.gca().invert_yaxis()  # 将y轴顺序颠倒，以便最高的重要性在顶部
    plt.show()

    all_fpr, mean_tpr, macro_auc = get_model_merged_roc_curve_parameters(X_test=X_test, y_test=y_test,
                                                                         model=rf_clf)

    # 绘制合并的 ROC 曲线
    plt.plot(all_fpr, mean_tpr, label=f"Random Forest (Macro AUC={macro_auc:.2f})")

    # 绘制随机猜测的对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Chance Level (AUC=0.50)")

    # 设置图表
    plt.title("Merged ROC Curve for Random Forest (Combined Classes)", fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def Best_XGBoost_model():

    data = pd.read_csv('./data/StressLevelDataset.csv')

    # Rename columns to remove spaces
    data.columns = data.columns.str.replace(' ', '')
    data.dropna(inplace=True)

    X = data.drop(['stress_level'], axis=1)  # Ensure 'Close' is dropped to create the feature set
    y = data['stress_level']  # Target variable is 'Close' price

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(method='ffill', inplace=True)

    # Calculate correlations between features
    correlation_matrix = X.corr()

    # Set a threshold to remove features with a correlation greater than
    threshold = 0.75
    drop_columns = set()

    # Traverse the correlation coefficient matrix to remove highly correlated features
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:  # The correlation is greater than the threshold
                colname = correlation_matrix.columns[i]
                drop_columns.add(colname)

    print("Dropped columns due to high correlation:", drop_columns)

    # Remove highly relevant features
    X = X.drop(columns=drop_columns)

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # random split the data, not appropriate for time series data

    # X_train, X_test, y_train, y_test = dataset.train_and_test()
    # 初始化 XGBoost 分类器
    xgb_classifier = XGBClassifier(
        max_depth=5,  # 树的最大深度
        learning_rate=1,  # 学习率
        n_estimators=150,  # 树的数量
        objective='multi:softmax',  # 多分类问题
        num_class=3,  # 类别数量（需根据数据调整）
        eval_metric='mlogloss'  # 评估指标
    )

    # 训练模型
    xgb_classifier.fit(X_train, y_train)

    # 使用模型进行预测
    y_pred = xgb_classifier.predict(X_test)

    # 计算并打印准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best XGBoost Accuracy: {accuracy}")

    # 绘制特征重要性
    feature_importances = xgb_classifier.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]  # 按重要性从高到低排序
    sorted_feature_labels = [X.columns[i] for i in sorted_indices]
    sorted_feature_importances = feature_importances[sorted_indices]

    plt.figure(figsize=(15, 8))
    plt.barh(sorted_feature_labels, sorted_feature_importances, color='skyblue')
    plt.xlabel("Feature Importance", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.title("Feature Importance in XGBoost", fontsize=16)
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.show()

    all_fpr, mean_tpr, macro_auc =  get_model_merged_roc_curve_parameters(X_test=X_test, y_test=y_test, model=xgb_classifier)

    # 绘制合并的 ROC 曲线
    plt.plot(all_fpr, mean_tpr, label=f"XGBoost (Macro AUC={macro_auc:.2f})")

    # 绘制随机猜测的对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Chance Level (AUC=0.50)")

    # 设置图表
    plt.title("Merged ROC Curve for XGBoost (Combined Classes)", fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def plot_roc_curve(y_test, y_score, n_classes=3):
    """
    绘制多类分类问题的 ROC 曲线。

    @param y_test: 真实标签
    @param y_score: 模型的预测概率
    @param n_classes: 类别数目
    """
    # 将标签二值化
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    plt.figure(figsize=(10, 7))

    # 计算每个类别的 ROC 曲线
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        auc_score = roc_auc_score(y_test_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc_score:.2f})")

    # 绘制随机猜测的对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Chance Level (AUC=0.50)")

    # 图表设置
    plt.title("ROC Curves", fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.show()


def get_model_merged_roc_curve_parameters(X_test, y_test ,model, num_classes=3):
    # 获取模型的预测概率
    y_score = model.predict_proba(X_test)

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



def get_RF_ROC_AUC_parameters():
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    # 创建并训练 Random Forest 分类器,max_feature是每个base learner用于训练的的feature数量（为了减少模型见的correlation）
    rf_clf = RandomForestClassifier(n_estimators=1000, max_depth=4, min_samples_leaf=1, min_samples_split=10,
                                    max_features='sqrt', bootstrap=True, criterion='entropy', random_state=42)
    rf_clf.fit(X_train, y_train)

    return get_model_merged_roc_curve_parameters(X_test=X_test, y_test=y_test,
                                                                         model=rf_clf)


def get_XGBoost_ROC_AUC_parameters():
    data = pd.read_csv('./data/StressLevelDataset.csv')

    # Rename columns to remove spaces
    data.columns = data.columns.str.replace(' ', '')
    data.dropna(inplace=True)

    X = data.drop(['stress_level'], axis=1)  # Ensure 'Close' is dropped to create the feature set
    y = data['stress_level']  # Target variable is 'Close' price

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(method='ffill', inplace=True)

    # Calculate correlations between features
    correlation_matrix = X.corr()

    # Set a threshold to remove features with a correlation greater than
    threshold = 0.75
    drop_columns = set()

    # Traverse the correlation coefficient matrix to remove highly correlated features
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:  # The correlation is greater than the threshold
                colname = correlation_matrix.columns[i]
                drop_columns.add(colname)

    print("Dropped columns due to high correlation:", drop_columns)

    # Remove highly relevant features
    X = X.drop(columns=drop_columns)

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)  # random split the data, not appropriate for time series data

    # X_train, X_test, y_train, y_test = dataset.train_and_test()
    # 初始化 XGBoost 分类器
    xgb_classifier = XGBClassifier(
        max_depth=5,  # 树的最大深度
        learning_rate=1,  # 学习率
        n_estimators=150,  # 树的数量
        objective='multi:softmax',  # 多分类问题
        num_class=3,  # 类别数量（需根据数据调整）
        eval_metric='mlogloss'  # 评估指标
    )

    # 训练模型
    xgb_classifier.fit(X_train, y_train)

    return get_model_merged_roc_curve_parameters(X_test=X_test, y_test=y_test, model=xgb_classifier)



if __name__ == '__main__':
    #Best_RFParam_Search()
    #Best_RandomForest_model()
    Best_XGBoost_model()
