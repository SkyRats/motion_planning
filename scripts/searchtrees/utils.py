class Utils:

    def advance_tree(self, vert, key):
        if vert.key > key:
            return vert.left
        elif vert.key < key:
            return vert.right

    def locate_insertion_point(self, key, root):
        parent = child = root
        while True:
            if child == None:
                return parent

            parent = child
            if child.key == key:
                return child
            child = self.advance_tree(child, key)

    def remove_from(self, node, key, root):
        if root != None:
            vert = root
            while vert != None and vert.key != key:
                vert = self.advance_tree(vert, key)

            if vert != None:
                vert.remove(node)
                if vert.is_empty():
                    if vert.parent == None:
                        return self.remove_root(vert)
                    elif vert == vert.parent.right:
                        vert.parent.right = self.remove_root(vert)
                    elif vert == vert.parent.left:
                        vert.parent.left = self.remove_root(vert)
        return root

    def remove_root(self, root):
        if root.left == None:
            if root.right != None:
                root.right.parent = root.parent
            return root.right

        parent = root
        child = root.left
        while child.right != None:
            paremt = child
            child = child.right

        if parent != root:
            parent.right = child.left
            child.left.parent = parent
            child.left = root.left
            root.left.parent = child
        child.right = root.right
        if root.right != None:
            root.right.parent = parent

        child.parent = root.parent

        return child

    def update_root(self, former_root):
        if former_root == None:
            return None
        else:
            root_candidate = former_root
            while root_candidate.parent != None:
                root_candidate = root_candidate.parent
            return root_candidate

class TreapUtils(Utils):

    def remove_root(self, root):
        unbalanced_root = Utils.remove_root(self, root)
        new_root = self.rebalance_from_top(unbalanced_root)
        return new_root

    def rebalance_from_top(self, root):
        first_pass = True
        new_root = None
        while root != None and root.right != None and root.left != None and (
            root.priority < root.left.priority or
            root.priority < root.right.priority
        ):

            if root.priority < root.left.priority:
                self.rotate_right(root)
            elif root.priority < root.right.priority:
                self.rotate_left(root)

            if first_pass:
                new_root = root.parent
                first_pass = False

            return new_root

        return root

    def rebalance_from_bottom(self, vert):
        while vert != None and vert.parent != None and (
            vert.priority > vert.parent.priority
        ):

            if vert == vert.parent.left:
                self.rotate_right(vert.parent)
            else:
                self.rotate_left(vert.parent)

    def rotate_right(self, vert):
        left_child = vert.left

        if vert.parent != None:
            if vert == vert.parent.left:
                vert.parent.left = left_child
            else:
                vert.parent.right = left_child
        left_child.parent = vert.parent

        vert.left = left_child.right
        if left_child.right != None:
            left_child.right.parent = vert

        left_child.right = vert
        vert.parent = left_child

    def rotate_left(self, vert):
        right_child = vert.right

        if vert.parent != None:
            if vert == vert.parent.left:
                vert.parent.left = right_child
            else:
                vert.parent.right = right_child
        right_child.parent = vert.parent

        vert.right = right_child.left
        if right_child.left != None:
            right_child.left.parent = vert

        right_child.left = vert
        vert.parent = right_child
