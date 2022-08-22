# Housing capacity and values of data items
class DataItem:
    def __init__(self, index, data, classif):
        self.index = index
        self.data = data
        self.classif = classif

    def getIndex(self):
        return self.index

    def getData(self):
        return self.data

    def getClassif(self):
        return self.classif