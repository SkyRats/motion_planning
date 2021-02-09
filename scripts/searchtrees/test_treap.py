from treap import Treap

def locate_nodes_between(init_vert, smaller_key, greater_key):
    selected_nodes = []
    next_verts = []

    if init_vert != None:
        next_verts.append(init_vert)
        while len(next_verts) > 0:
            vert = next_verts.pop(0)
            if vert.key > smaller_key and vert.left != None:
                next_verts.append(vert.left)
            if vert.key < greater_key and vert.right != None:
                next_verts.append(vert.right)
            if smaller_key <= vert.key <= greater_key:                        
                selected_nodes.extend(get_nodes_from_vert(vert))

    return selected_nodes

def get_nodes_from_vert(vert):
    nodes = []
    for node in vert.content:
        nodes.append(node)
    return nodes

nodes = [10, 13, 35, 64, 92, 1231, 1234, 5, 6, 45, 56, 67, 78, 89, 90]
keys = [1, 1, 5, 2, 3, 4, 6, 4, 10, 5, 6, 7, 8, 9, 10]

root_node = 0
root_key = 0

# Constructor test
tree = Treap()
tree.insert(root_node, root_key)

# insert function test
for i in range(len(nodes)):
    tree.insert(nodes[i], keys[i])
tree.print_tree()

# locate test
print(tree.locate(10).content) # Single match
print(tree.locate(1).content) # Multiple matches
print('\n')

# TreapVertex test
print(tree.root)
key_5 = tree.locate(5)
print(key_5)
print(key_5.priority)
print('\n')

# Test key insertion order
tree.print_left_root_right()
print('\n')

# remove test
tree.remove(root_node, root_key) # remove root
# tree.print_tree()
tree.print_left_root_right()
print('\n')

tree.remove(nodes[-1], keys[-1]) # Remove single occurence key
tree.print_tree()
# tree.print_left_root_right()
print('\n')


for i in range(len(nodes)):
    print('{0}:\t{1}'.format(keys[i], nodes[i]))
print('\n')

# Test function from RRT
greater_key = 7
smaller_key = 2
greater_vert = tree.locate(greater_key)
smaller_vert = tree.locate(smaller_key)

print(greater_vert.key)
print(greater_vert.content)
print('\n')

init_vert = None
while (
    greater_vert.parent != None and greater_vert == greater_vert.parent.right
    and smaller_vert.parent != None and smaller_vert == smaller_vert.parent.left
    ):
    greater_vert = greater_vert.parent
    smaller_vert = smaller_vert.parent

if greater_vert.parent == None or greater_vert == greater_vert.parent.left:
    init_vert = greater_vert
else:
    init_vert = smaller_vert

print(init_vert.key)
print(init_vert.content)
print('\n')

for node in locate_nodes_between(init_vert, smaller_key, greater_key):
    print(node)