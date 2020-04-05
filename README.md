This is a machine learning library developed by Brian Lavallee for CS5350/6350 at the University of Utah.

## Decision Trees
To create a Decision Tree, call
```python
tree = DecisionTree(training_data, training_labels)
```

To limit the depth of the tree, use the `max_depth` parameter.
To change the selection criteria from `entropy`, use the `measure` parameter which can also be set to `majority_error` or `gini_index`.
Finally, to limit which attributes the tree will split on, use the `attributes` parameter.

## Adaboost
To create an adaboost classifier, call
```python
model = AdaBoost(train, train_labels)
```

## Bagging and Random Forest
To create a bagged trees classifier, call
```python
model = BaggedForest(train, train_labels)
```

The model can be changed to a random forest by setting the `sample_size` parameter.

## LMS
To create a least mean squares regressor, call
```python
model = LinearRegressor(train, train_labels)
```

To change the batch size, use the parameter `batch_size`.
The default setting is -1 which sets `batch_size` to n_samples.

## Perceptron
To create a perceptron classifier, call
```python
model = Perceptron(x_train, y_train)
```

There are no additional parameters to set.
You may create a ```VotingPerceptron``` or an ```AveragingPerceptron``` in the same way.
