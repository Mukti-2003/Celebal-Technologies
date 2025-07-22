import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def visualize_results(y_true, y_pred, class_names):
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
from train_model import train_model  # import your training function
from load_dataset import load_data  # import your dataset loading function

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, class_names = load_data()
    model, y_pred = train_model(X_train, X_test, y_train, y_test)

    # visualize the confusion matrix
    visualize_results(y_test, y_pred, class_names)

