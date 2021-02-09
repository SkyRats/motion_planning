from bst import BST

nodes = [10, 13, 35, 64, 92, 1231, 1234, 5, 6]
keys = [1, 1, 5, 2, 3, 4, 6, 4, 10]

root_node = 0
root_key = 0

# Constructor test
tree = BST()
tree.insert(root_node, root_key)

# insert function test
for i in range(len(nodes)):
    tree.insert(nodes[i], keys[i])
tree.print_tree()

# locate test
print(tree.locate(10).content) # Single match
print( tree.locate(1).content ) # Multiple matches
print('\n')

# Test key insertion order
tree.print_left_root_right()
print('\n')

# remove test
tree.remove(root_node, root_key) # remove root
tree.print_left_root_right()
# tree.print_tree()
print('\n')

tree.remove(nodes[-1], keys[-1]) # Remove single occurence key
tree.print_left_root_right()
# tree.print_tree()
print('\n')
