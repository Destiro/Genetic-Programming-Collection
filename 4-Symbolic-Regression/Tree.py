from functions import *

# Note: This class is not being used in my final solution
class TreeNode:
    # Initialize the tree node
    def __init__(self, isLeaf, left, right, operation):
        self.isLeaf = isLeaf
        self.left = left
        self.right = right
        self.operation = operation

    def evaluate(self):
        if self.isLeaf:
            return self.left
        else:
            self.performOperation()

    def performOperation(self):
        if self.operation == "add":
            return add(self.left.evaluate(), self.right.evaluate())
        elif self.operation == "sub":
            return sub(self.left.evaluate(), self.right.evaluate())
        elif self.operation == "div":
            return sub(self.left.evaluate(), self.right.evaluate())
        elif self.operation == "mul":
            return sub(self.left.evaluate(), self.right.evaluate())