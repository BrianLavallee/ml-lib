This is a machine learning library developed by Brian Lavallee for CS5350/6350 at the University of Utah.

## Decision Trees
To create a Decision Tree, call
```python
tree = DecisionTree(training_data, training_labels)
```

To limit the depth of the tree, use the `max_depth` parameter.
To change the selection criteria from `entropy`, use the `measure` parameter which can also be set to `majority_error` or `gini_index`.
Finally, to limit which attributes the tree will split on, use the `attributes` parameter.
