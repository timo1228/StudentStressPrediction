from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt

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
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 查看特征重要性
    feature_labels = ['anxiety_level', 'self_esteem', 'mental_health_history', 'depression', 'headache', 'blood_pressure',
                      'sleep_quality', 'breathing_problem', 'noise_level', 'living_conditions', 'safety', 'basic_needs',
                      'academic_performance', 'study_load', 'teacher_student_relationship', 'future_career_concerns',
                      'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']
    #通过评估每个特征对模型预测性能的贡献来计算。这是通过所有树（1000个）的加权平均得到的
    feature_importances = rf_clf.feature_importances_
    plt.figure(figsize=(15, 8))
    plt.barh(feature_labels, feature_importances)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Random Forest')
    plt.show()



if __name__ == '__main__':
    Best_RandomForest_model()
    #Best_RFParam_Search()