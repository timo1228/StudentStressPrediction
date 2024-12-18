"""
    author: cyt
    description: using chroma db, a vector search database using HNSW in-memory graph for efficient knn search
"""
import uuid
import os
import sys

import chromadb
import numpy as np
from DataSource import StudentStressDataSet

from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt

class KNNModel:
    def __init__(self, similarity_method="l2", num_classes=2):
        self.num_classes = num_classes

        path = "./chromadb"
        os.makedirs(path, exist_ok=True)  # 确保路径存在

        # Initialize PersistentClient, if path does not exist, it will automatically create the database
        chroma_client = chromadb.PersistentClient(path=path)

        if similarity_method == "l2":
            collection_name = "student_stress_dataset_l2"
        elif similarity_method == "cosine":
            collection_name = "student_stress_dataset_cosine"
        else:
            raise RuntimeError("Invalid similarity method")
        #getting collection
        try:
            collection =  chroma_client.get_collection(collection_name)
            self.collection = collection
        except Exception as e:
            #collection doesn't exist(not stored in the disk), create ourselves
            print("Creating collection")
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": similarity_method}
            )# l2 is the default
            self.collection = collection
            self.load_data()
            print("Data has been loaded and collection has been created successfully")

        self.client = chroma_client
        self.similarity_method = similarity_method
        # Verify metadata
        if self.collection.metadata.get("hnsw:space") != self.similarity_method:
            raise RuntimeError("Similarity method does not match with collection")
        print("Initializing KNN model completed")


    def load_data(self):
        dataset = StudentStressDataSet()
        X_train, X_test, y_train, y_test = dataset.train_and_test()

        # Clear the collection
        try:
            collection_count = self.collection.count()
            if collection_count != 0:
                self.collection.delete(self.collection.get()["ids"])
        except Exception as e:
            print("Clear collection failed:", e)
            sys.exit(2)

        print("Loading Data")
        for i in range(X_train.shape[0]):
            rand_id = str(uuid.uuid4())
            self.collection.add(
                embeddings=[X_train[i]],  # 存储的向量
                ids=[rand_id],  # 唯一 ID
                metadatas=[{"label": int(y_train[i])}]  # 存储标签作为元信息
            )

    def knn_search(self, k, query_embedding):
        if k <= 0:
            raise ValueError("k must be positive integer")
        if not isinstance(query_embedding,  (list, np.ndarray)) or len(query_embedding) == 0:
            raise ValueError("query_embedding must be a non-empty list or op array")

        #based on the collection we created, the information on creation to decide which similarity to search e.g. l2 or cosine
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results

    def predict_proba(self, X, k=10):
        """
            calculate the score of every input row in X，used for plotting ROC curve。
            score_j = num_class_j/k
            :param X: input vectors, size (num_vectors, num_features)。
            :param k: search k nearest neighbors
            :return: the score of every input row in X, size is (num_vectors, num_classes)。
            """
        if not isinstance(X, (list, np.ndarray)):
            raise ValueError("X must be a list or numpy array of embeddings.")

        score_matrix = np.zeros((X.shape[0], self.num_classes))
        for i, query_embedding in enumerate(X):
            # knn search
            results = self.knn_search(k, query_embedding)

            # get the neighbors label
            neighbor_labels = [metadata["label"] for metadata in results["metadatas"][0]]

            # calculate the number of every class
            label_counts = Counter(neighbor_labels)

            # calculate the  score of every class
            total_neighbors = sum(label_counts.values())
            label_scores = {label: count / total_neighbors for label, count in label_counts.items()}

            for label, score in label_scores.items():
                score_matrix[i, label] = score

        return score_matrix

def KNNModel_test():
    k = 8  # define k nearest neighbor to search
    model = KNNModel(similarity_method="cosine", num_classes=3) #0.9 k=8
    #model = KNNModel(similarity_method="l2") #0.9 k=12
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()
    #predicting
    y_pred = np.zeros(y_test.shape[0]).reshape(-1, 1)
    for i in range(X_test.shape[0]):
        x = X_test[i]
        results = model.knn_search(k=k, query_embedding=x)
        metadatas = results["metadatas"][0]
        labels = [metadata["label"] for metadata in metadatas]
        label_counts = Counter(labels)
        #get the mode in the knn search
        most_common_label, count = label_counts.most_common(1)[0]
        y_pred[i] = most_common_label

    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN Search Accuracy: {accuracy}")

def KNNModel_test_with_seperate_roc():
    k = 8  # Number of nearest neighbors
    num_classes = 3  # Specify the number of classes
    model = KNNModel(similarity_method="cosine", num_classes=3)  # Initialize with cosine similarity
    model.num_classes = num_classes  # Ensure num_classes is set in the model
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    # Predict probabilities
    probabilities = model.predict_proba(X_test, k=k)  # Directly use the score_matrix output

    # One-hot encode the true labels for multiclass ROC
    y_test_one_hot = np.zeros((len(y_test), num_classes))
    for i, label in enumerate(y_test):
        y_test_one_hot[i, label] = 1

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    auc_scores = []
    for i in range(num_classes):
        #probabilities[:, i] denotes the probability of class i, y_test_one_hot[:, i] denotes whether the row is class i
        fpr, tpr, thresholds = roc_curve(y_test_one_hot[:, i], probabilities[:, i])
        auc = roc_auc_score(y_test_one_hot[:, i], probabilities[:, i])
        auc_scores.append(auc)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc:.4f})")

    # Compute macro-average AUC
    macro_auc = np.mean(auc_scores)

    # Plot settings
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for KNN Model (Macro AUC = {macro_auc:.4f})")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    print(f"Per-class AUC Scores: {dict(zip(range(num_classes), auc_scores))}")
    print(f"Macro-average AUC: {macro_auc:.4f}")

def KNNModel_test_with_merged_roc():
    k = 8  # Number of nearest neighbors
    num_classes = 3  # Specify the number of classes
    model = KNNModel(similarity_method="cosine", num_classes=3)  # Initialize with cosine similarity
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    # Predict probabilities
    probabilities = model.predict_proba(X_test, k=k)  # Directly use the score_matrix output

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
        fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], probabilities[:, i])
        roc_auc[i] = roc_auc_score(y_test_one_hot[:, i], probabilities[:, i])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Average it and compute the macro-average ROC curve
    mean_tpr /= num_classes

    # Compute macro-average AUC
    macro_auc = roc_auc_score(y_test_one_hot, probabilities, average="macro")

    # Plot the macro-average ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(all_fpr, mean_tpr, color="blue", label=f"Macro-Average ROC (AUC = {macro_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess (AUC = 0.5)")

    # Plot settings
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Merged ROC Curve for Multiclass KNN Model")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    print(f"Macro-average AUC: {macro_auc:.4f}")

def get_KNN_ROC_AUC_parameters():
    k = 8  # Number of nearest neighbors
    num_classes = 3  # Specify the number of classes
    model = KNNModel(similarity_method="cosine", num_classes=3)  # Initialize with cosine similarity
    dataset = StudentStressDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    # Predict probabilities
    probabilities = model.predict_proba(X_test, k=k)  # Directly use the score_matrix output

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
        fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], probabilities[:, i])
        roc_auc[i] = roc_auc_score(y_test_one_hot[:, i], probabilities[:, i])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Average it and compute the macro-average ROC curve
    mean_tpr /= num_classes

    # Compute macro-average AUC
    macro_auc = roc_auc_score(y_test_one_hot, probabilities, average="macro")

    return all_fpr, mean_tpr, macro_auc

if __name__ == '__main__':
    #KNNModel_test()
    # Test the function
    #KNNModel_test_with_seperate_roc()
    KNNModel_test_with_merged_roc()
