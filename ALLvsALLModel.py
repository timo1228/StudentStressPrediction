import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from DataSource import StudentStressDataSet
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

class AllVsAllClassifier(BaseEstimator, ClassifierMixin):
    """
    All-vs-all classifier
    We assume that the classes will be the integers 0,..,(n_classes-1).
    We assume that the estimator provided to the class, after fitting, has a "decision_function" that
    returns the score for the positive class.
    """

    def __init__(self, estimator, n_classes):
        """
        Constructed with the number of classes and an estimator (e.g. an
        SVM estimator from sklearn)
        @param estimator : binary base classifier used
        @param n_classes : number of classes
        """
        self.n_classes = n_classes
        #using map tp store estimators that distinguish from class i and class j, key = ij, i<j(reqired)
        self.estimators = dict()
        for class_i in range(n_classes):
            for class_j in range(class_i+1, n_classes, 1):
                estimator_ij = clone(estimator) #this is the estimator to distinguish from class i and class j
                self.estimators[str(class_i)+str(class_j)] = estimator_ij
        self.fitted = False

    def fit(self, X, y=None):
        """
        This should fit one classifier for each class.
        self.estimators[i] should be fit on class i vs rest
        @param X: array-like, shape = [n_samples,n_features], input data
        @param y: array-like, shape = [n_samples,] class labels
        @return returns self
        """
        # Your code goes here
        if X is None or y is None or X.size == 0 or y.size == 0:
            return self
        # train each estimator
        for class_i in range(self.n_classes):
            for class_j in range(class_i + 1, self.n_classes, 1):
                #train estimator split class i and class j (i<j)
                estimator_ij = self.estimators[str(class_i)+str(class_j)]
                #mask is a numpy array with boolean value that if y == class_i) | (y == class_j), it is true
                mask = (y == class_i) | (y == class_j)
                X_subset = X[mask]
                y_subset = y[mask]
                # need to refactor y to binary, for y[i] == class_i, y[i] = 1; for y[i] != class_i, y[i] = 0
                y_subset = np.where(y_subset == class_i, 1, 0)
                #so the positive class is class i
                estimator_ij.fit(X_subset, y_subset)

        self.fitted = True
        return self

    def get_decision_scores(self, estimator, X, neg_score=False):
        """
        Unified interface to get decision scores from the estimator.
        If `decision_function` is available, use it. Otherwise, use `predict_proba`.
        neg_score means when neg_score=True, return the score of negative class.
        e.g. score of positive class is estimator.decision_function, when neg_score=True, we want to return the
        score of negative class, which is -estimator.decision_function(for linear decision plane model like svm and logistic regression).
        """
        if hasattr(estimator, "decision_function"):
            if neg_score:
                # w^Tx+b is the prediction of class j, -(w^Tx+b) is class i
                return -estimator.decision_function(X)
            else:
                return estimator.decision_function(X)
        elif hasattr(estimator, "predict_proba"):
            # Use probability of the positive class
            proba = estimator.predict_proba(X)
            if neg_score:
                return proba[:, 0]  # Negative class probabilities
            else:
                return proba[:, 1]  # Positive class probabilities
        else:
            raise AttributeError(
                "The provided estimator must have either 'decision_function' or 'predict_proba'."
            )

    def decision_function(self, X):
        """
        Returns the score of each input for each class. Assumes
        that the given estimator also implements the decision_function method (which sklearn SVMs do),
        and that fit has been called.
        @param X : array-like, shape = [n_samples, n_features] input data
        @return array-like, shape = [n_samples, n_classes]
        """
        if not self.fitted:
            raise RuntimeError("You must train classifer before predicting data.")

        # Replace the following return statement with your code
        scores = []
        # In sklearn, decision_function returns the signed functional margin of input X, w^Tx+b
        for class_i in range(self.n_classes):
            scores_i = []
            for class_j in range(0, self.n_classes):
                if class_i == class_j:
                    continue
                elif class_i < class_j:
                    estimator_ij = self.estimators[str(class_i)+str(class_j)]
                    score_ij = self.get_decision_scores(estimator_ij, X, neg_score=False)
                    scores_i.append(score_ij.reshape(-1, 1))
                else: #class_i > class_j
                    estimator_ji = self.estimators[str(class_j)+str(class_i)]
                    #in this case, score of decision_function is class_j, wo need negative score
                    score_ij = self.get_decision_scores(estimator_ji, X, neg_score=True)
                    scores_i.append(score_ij.reshape(-1, 1))
            scores_i = np.hstack(scores_i)
            #score i is the sum of the score on i from other (k-1) classifiers
            scores_i = np.sum(scores_i, axis=1)
            scores.append(scores_i.reshape(-1, 1))
        decision_matrix = np.hstack(scores)
        return decision_matrix

    def predict(self, X):
        """
        Predict the class with the highest score.
        @param X: array-like, shape = [n_samples,n_features] input data
        @returns array-like, shape = [n_samples,] the predicted classes for each input
        """
        # Replace the following return statement with your code
        decision_matrix = self.decision_function(X)
        return np.argmax(decision_matrix, axis=1)

def SVM():
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    svm_estimator = SVC(kernel='poly', degree=4, C=0.09, coef0=3, gamma='scale') #0.8954545454545455
    #svm_estimator = SVC(kernel='rbf', C=1, gamma=0.1) #88.6%, best we can find
    clf_allvsall = AllVsAllClassifier(svm_estimator, n_classes=3)
    clf_allvsall.fit(X_train, y_train)

    y_pred = clf_allvsall.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Model Accuracy: {accuracy}")


def LogistiRegression():
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    lr_estimator = LogisticRegression(random_state=42, penalty='l2', C=2, max_iter=10000) #0.8909090909090909
    #lr_estimator = LogisticRegression(random_state=42, penalty='l1', solver='saga', C=8, max_iter=10000) #82.7%
    clf_allvsall = AllVsAllClassifier(lr_estimator, n_classes=3)
    clf_allvsall.fit(X_train, y_train)

    y_pred = clf_allvsall.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy}")

def NaiveBayes():
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    #这个要求feature也都是二值化的，binarize=1意思是对于连续的feature，其值>1,设为1;<1,设为0。将其二值化
    bnb_estimator = BernoulliNB(alpha=1, binarize=1, fit_prior=True) #87.7%
    clf_allvsall = AllVsAllClassifier(bnb_estimator, n_classes=3)
    clf_allvsall.fit(X_train, y_train)

    y_pred = clf_allvsall.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Bernoulli Naive Bayes Accuracy: {accuracy}")

def plot_seperate_auc_curves():
    # Assuming StudentStressDataSet is a class that prepares the dataset
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    models = {
        "SVM": SVC(kernel='poly', degree=4, C=0.09, coef0=3, gamma='scale'),
        "Logistic Regression": LogisticRegression(random_state=42, penalty='l2', C=2, max_iter=10000),
        "Naive Bayes": BernoulliNB(alpha=1, binarize=1, fit_prior=True),
    }

    # Prepare plot
    plt.figure(figsize=(10, 7))

    for name, model in models.items():
        # Initialize AllVsAllClassifier
        clf_allvsall = AllVsAllClassifier(model, n_classes=3)
        clf_allvsall.fit(X_train, y_train)

        # Get decision function or probabilities for test set
        y_score = clf_allvsall.decision_function(X_test)

        # Calculate AUC for all classes and average them (Macro-average AUC)
        auc_scores = []
        for i in range(3):  # Assuming 3 classes
            y_test_binary = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_test_binary, y_score[:, i])
            auc_score = roc_auc_score(y_test_binary, y_score[:, i])
            auc_scores.append(auc_score)

            # Plot each ROC curve for individual classes (if needed)
            plt.plot(fpr, tpr, label=f"{name} - Class {i} (AUC={auc_score:.2f})")

        # Calculate macro-average AUC
        macro_auc = np.mean(auc_scores)

        # Plot the ROC curve for the macro-averaged AUC
        #plt.plot(fpr, tpr, label=f"{name} (Macro AUC={macro_auc:.2f})")

    # Plot settings
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Chance Level (AUC=0.50)")
    plt.title("AUC-ROC Curves for Different Models (Macro-Averaged)", fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def plot_merged_auc_curves():
    """
    Plot merged ROC curves for multiple models (SVM, Logistic Regression, Naive Bayes).
    This function will display a combined ROC curve for each model with macro-average AUC.
    """
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    models = {
        "SVM": SVC(kernel='poly', degree=4, C=0.09, coef0=3, gamma='scale'),
        "Logistic Regression": LogisticRegression(random_state=42, penalty='l2', C=2, max_iter=10000),
        "Naive Bayes": BernoulliNB(alpha=1, binarize=1, fit_prior=True),
    }

    # Prepare plot
    plt.figure(figsize=(10, 7))

    # To collect all fpr, tpr and AUC scores for plotting later
    all_fpr = np.linspace(0, 1, 100)  # Common FPR to interpolate all curves
    for name, model in models.items():
        # Initialize AllVsAllClassifier
        clf_allvsall = AllVsAllClassifier(model, n_classes=3)
        clf_allvsall.fit(X_train, y_train)

        # Get decision function or probabilities for test set
        y_score = clf_allvsall.decision_function(X_test)

        # Calculate AUC for all classes and average them (Macro-average AUC)
        auc_scores = []
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(3):  # Assuming 3 classes
            y_test_binary = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_test_binary, y_score[:, i])
            auc_score = roc_auc_score(y_test_binary, y_score[:, i])
            auc_scores.append(auc_score)

            # Ensure the starting point is (0,0) for each curve
            if fpr[0] > 0:
                fpr = np.insert(fpr, 0, 0)  # Insert 0 at the start of fpr
                tpr = np.insert(tpr, 0, 0)  # Insert 0 at the start of tpr

            # Interpolate each ROC curve at common FPR
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        # Average it to get the macro-average TPR
        mean_tpr /= 3

        # Calculate macro-average AUC
        macro_auc = np.mean(auc_scores)

        # Plot the ROC curve for the macro-averaged AUC
        plt.plot(all_fpr, mean_tpr, label=f"{name} (Macro AUC={macro_auc:.2f})")

    # Plot settings
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Chance Level (AUC=0.50)")
    plt.title("Merged ROC Curves for Different Models (Macro-Averaged)", fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def get_AVA_ROC_AUC_parameters():
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    models = {
        "SVM": SVC(kernel='poly', degree=4, C=0.09, coef0=3, gamma='scale'),
        "Logistic Regression": LogisticRegression(random_state=42, penalty='l2', C=2, max_iter=10000),
        "Naive Bayes": BernoulliNB(alpha=1, binarize=1, fit_prior=True),
    }

    # Prepare plot
    plt.figure(figsize=(10, 7))

    # To collect all fpr, tpr and AUC scores for plotting later
    all_fpr = np.linspace(0, 1, 100)  # Common FPR to interpolate all curves

    AvA_params = dict()
    for name, model in models.items():
        # Initialize AllVsAllClassifier
        clf_allvsall = AllVsAllClassifier(model, n_classes=3)
        clf_allvsall.fit(X_train, y_train)

        # Get decision function or probabilities for test set
        y_score = clf_allvsall.decision_function(X_test)

        # Calculate AUC for all classes and average them (Macro-average AUC)
        auc_scores = []
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(3):  # Assuming 3 classes
            y_test_binary = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_test_binary, y_score[:, i])
            auc_score = roc_auc_score(y_test_binary, y_score[:, i])
            auc_scores.append(auc_score)

            # Ensure the starting point is (0,0) for each curve
            if fpr[0] > 0:
                fpr = np.insert(fpr, 0, 0)  # Insert 0 at the start of fpr
                tpr = np.insert(tpr, 0, 0)  # Insert 0 at the start of tpr

            # Interpolate each ROC curve at common FPR
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        # Average it to get the macro-average TPR
        mean_tpr /= 3

        # Calculate macro-average AUC
        macro_auc = np.mean(auc_scores)

        AvA_params[name] = [all_fpr, mean_tpr, macro_auc]

    return AvA_params

if __name__ == '__main__':
    #SVM()
    #LogistiRegression()
    #NaiveBayes()
    # 调用绘制函数
    #plot_seperate_auc_curves()
    plot_merged_auc_curves()

