from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate(y_test, y_pred):
    print("---  Clasificación ---")
    print(classification_report(y_test, y_pred))
    
    print("--- Matriz de Confusión ---")
    print(confusion_matrix(y_test, y_pred))