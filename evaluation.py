from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import pandas as pd

# Example true labels and predicted labels
test_csv= pd.read_excel("D:\\vit btech final year 2023\\Capstone\\whisper_code\\Sample_Dataset.xlsx")
true_labels = test_csv['tlabel'].tolist()
predicted_labels = test_csv['PL_MS_LCNN'].tolist()

print("Printing evaluation parameters for ASV19 - MESONET")

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)

# Calculate recall
recall = recall_score(true_labels, predicted_labels)
print("Recall:", recall)

# Calculate precision
precision = precision_score(true_labels, predicted_labels)
print("Precision:", precision)

# Calculate F1 score
f1 = f1_score(true_labels, predicted_labels)
print("F1 Score:", f1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)


print(confusion_matrix(true_labels, predicted_labels).ravel())

