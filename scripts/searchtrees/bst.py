# TODO check if locate and _utils.locate_insertion_point can be merged

from utils import Utils
from vertex import Vertex

class BST:
    def __init__(self):
        self._utils = Utils()
        self.root = None

    def _create_vertex(self, node, key, parent):
        return Vertex(node, key, parent)

    def locate(self, key):
        assert(self.root != None), 'Trying to locate from empty tree'
        selection = self.root
        while True:
            if selection.key == key:
                return selection

            new_selection = self._utils.advance_tree(selection, key)
            if new_selection == None:
                return selection
            else:
                selection = new_selection

    def insert(self, node, key):
        if self.root == None:
            self.root = self._create_vertex(node, key, None)
            return None
        else:
            insertion_vertex = self._utils.locate_insertion_point(key, self.root)
            if insertion_vertex.key == key:
                insertion_vertex.append(node)
            elif insertion_vertex.key > key:
                insertion_vertex.left = self._create_vertex(node, key, insertion_vertex)
            else:
                insertion_vertex.right = self._create_vertex(node, key, insertion_vertex)

            return insertion_vertex

    def remove(self, node, key):
        assert(self.root != None), 'Trying to remove from empty tree'
        self.root = self._utils.remove_from(node, key, self.root)

    def print_left_root_right(self, vert = 'begin'):
        if vert == 'begin':
            vert = self.root
        if vert != None:
            self.print_left_root_right(vert.left)
            print(vert.key)
            self.print_left_root_right(vert.right)

    def print_tree(self, root = 'begin'):
        if root != None:
            if root == 'begin':
                root = self.root
            parent = root.parent
            right = root.right
            left = root.left

            print('Current: ' + str(root.key))
            if parent:
                print('Parent: ' + str(root.parent.key))
            if left:
                print('Left: ' + str(root.left.key))
            if right:
                print('Rigth: ' + str(root.right.key))
            print('\n')

            self.print_tree(root.left)
            self.print_tree(root.right)
