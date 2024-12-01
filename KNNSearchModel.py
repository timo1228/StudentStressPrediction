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

class KNNModel:
    def __init__(self, similarity_method="l2"):
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

def KNNModel_test():
    k = 10  # define k nearest neighbor to search
    model = KNNModel(similarity_method="cosine")
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




if __name__ == '__main__':
    KNNModel_test()