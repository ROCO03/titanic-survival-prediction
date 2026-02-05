from preprocessing import preprocessing
from model import model
from evaluate import evaluate
from sklearn.model_selection import train_test_split

def train():
    X, Y = preprocessing()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    sorter = model()
    sorter.fit(X_train, y_train)
    
    y_pred = sorter.predict(X_test)
    
    evaluate(y_test, y_pred)

if __name__ == "__main__":
    train()