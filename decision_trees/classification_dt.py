class ClassificationDecisionTree(object):
    """
    This class respresent subtrees in CART Decision Trees Algorithm
    """
    
    def __init__(self, purity_measure='gini', num_classes=2):
        self.purity_measure = purity_measure
        self.num_classes = num_classes
        self.left = None
        self.right = None
    
    def train(self, features, targets, max_depth=2):
        """
        Predict class of the point
        """
        self.purity_value = self.getPurity(targets)
        
        # Break if gini is perfect or max depth is reached
        if self.purity_measure == 0. or max_depth == 0:
            self.label = str(ClassificationDecisionTree.dominatingClass(targets))
            return
    
        labels = features.columns.values
        best_value = 0
        # Go through all available features
        for label in labels:
            for value in features[label]:
                # Find indices of left and right kids
                left_indices = features.loc[features[label] < value].index
                right_indices = features.loc[features[label] >= value].index
                
                # If one of the splits is empty, skip the iteration
                if left_indices.size == 0 or right_indices.size == 0:
                    continue
                
                # Calculate gini's of the kids
                left_purity = self.getPurity(targets.loc[left_indices])
                right_purity = self.getPurity(targets.loc[right_indices])
                
                # Calculate the gini improvement
                improvement_L = (len(left_indices) / len(targets)) * left_purity
                improvement_R = (len(right_indices) / len(targets)) * right_purity
                improvement = self.purity_value - improvement_L - improvement_R
                
                # Save it's better than the one before
                if improvement > best_value:
                    best_value = improvement
                    self.label = label
                    self.value = value
                    
        # Find indices of the winner split
        left_indices = features.loc[features[self.label] < self.value].index
        right_indices = features.loc[features[self.label] >= self.value].index
        
        # Create left child
        self.left = ClassificationDecisionTree(self.purity_measure, self.num_classes)
        self.left.train(features.loc[left_indices], targets.loc[left_indices], max_depth-1)
        
        # Create right child
        self.right = ClassificationDecisionTree(self.purity_measure, self.num_classes)
        self.right.train(features.loc[right_indices],targets.loc[right_indices], max_depth-1)
    
    def predict(self, point):
        """
        Predict class of the point
        """
        # If there are no children, return class
        # End of recursion
        if not self.right:
            return self.label
        
        if point[self.label].values[0] < self.value:
            return self.left.predict(point)
        return self.right.predict(point)
    
    def getPurity(self, labels):
        """
        Calculates purity of the node given defined purity_measure
        """
        class_votes = np.array([sum(labels == i) for i in range(self.num_classes)])
        weighted_votes = class_votes / len(labels)
        
        if self.purity_measure == 'gini':
            return 1 - sum(weighted_votes ** 2)
        elif self.purity_measure == 'entropy':
            return - sum(weighted_votes * np.log(weighted_votes + 1e-8))

    def dominatingClass(targets, num_classes=3):
        """
        Calculates the dominating class in a sample.
        """
        class_votes = [sum(targets == i) for i in range(num_classes)]
        # Compute the winner class
        return np.argmax(class_votes)