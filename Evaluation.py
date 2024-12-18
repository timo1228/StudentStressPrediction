import MLP
import Ensemble
import KNN
import ALLvsALL

import matplotlib.pyplot as plt


def plot_all_models_roc_auc():
    models = dict()
    #get roc auc parameters
    models["MLP"] = MLP.get_MLP_merged_roc_curve_parameters()
    models["Random Forest"] = Ensemble.get_RF_ROC_AUC_parameters()
    models["XGBoost"] = Ensemble.get_XGBoost_ROC_AUC_parameters()
    models["KNN"] = KNN.get_KNN_ROC_AUC_parameters()
    ava_params = ALLvsALL.get_AVA_ROC_AUC_parameters()
    for name, params in ava_params.items():
        models[name] = params

    for name, params in models.items():
        all_fpr, mean_tpr, macro_auc = params
        # Plot the ROC curve for the macro-averaged AUC
        plt.plot(all_fpr, mean_tpr, label=f"{name} (Macro AUC={macro_auc:.2f})")

    # Plot settings
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Chance Level (AUC=0.50)")
    plt.title("ROC Curves for All Models (Macro-Averaged)", fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.show()




if __name__ == '__main__':
    plot_all_models_roc_auc()