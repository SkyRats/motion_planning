class Vertex:
    def __init__(self, node, key, parent):
        self.content = [node]
        self.key = key
        self.parent = parent
        self.right = None
        self.left = None

    def is_empty(self):
        return True if len(self.content) == 0 else False

    def append(self, node):
        self.content.append(node)
        self.content.sort()

    def remove(self, node):
        self.content.remove(node)
