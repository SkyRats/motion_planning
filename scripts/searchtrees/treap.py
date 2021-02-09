from utils import TreapUtils
from bst import BST
from vertex import Vertex

from random import randint

MAX_RANDOM_INT = 10000

class Treap(BST):
    def __init__(self):
        BST.__init__(self)
        self._utils = TreapUtils()

    class TreapVertex(Vertex):
        def __init__(self, node, key, parent):
            Vertex.__init__(self, node, key, parent)
            self.priority = randint(0, MAX_RANDOM_INT)

    def _create_vertex(self, node, key, parent):
        return self.TreapVertex(node, key, parent)

    def insert(self, node, key):
        insertion_vertex = BST.insert(self, node, key)
        self._utils.rebalance_from_bottom(insertion_vertex)
        self.root = self._utils.update_root(self.root)

    def remove(self, node, key):
        assert(self.root != None), 'Trying to remove from empty tree'
        self.root = self._utils.remove_from(node, key, self.root)

    def print_tree(self, root = 'begin'):
        if root != None:
            if root == 'begin':
                root = self.root
            parent = root.parent
            right = root.right
            left = root.left

            print('CURRENT: ' + str(root.key))
            print('Priority: ' + str(root.priority))
            if parent:
                print('Parent: ' + str(root.parent.key))
            if left:
                print('Left: ' + str(root.left.key))
            if right:
                print('Rigth: ' + str(root.right.key))
            print('\n')

            self.print_tree(root.left)
            self.print_tree(root.right)
