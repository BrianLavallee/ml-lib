
from LinearRegression.lms import LinearRegressor, ExactLinearRegressor
from data import load_concrete

def concrete():
    train, train_labels, test, test_labels = load_concrete()
    
    model = LinearRegressor(train, train_labels)
    print(model.cost(train, train_labels))
    print(model.cost(test, test_labels))

    model = ExactLinearRegressor(train, train_labels)
    print(model.cost(train, train_labels))
    print(model.cost(test, test_labels))

def main():
    concrete()

main()
