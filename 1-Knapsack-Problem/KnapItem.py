# Housing capacity and values of data items
class KnapItem:
    def __init__(self, index, value, capacity):
        self.index = index
        self.capacity = capacity
        self.value = value

    def getIndex(self):
        return self.index

    def getCapacity(self):
        return self.capacity

    def getValue(self):
        return self.value