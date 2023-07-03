import random
from time import sleep



def symbol_max(symbol_quantity):
    symbol_max = 0
    
    for _ in range(symbol_quantity - 1):
        symbol_max |= 1
        symbol_max <<= 1
    
    return symbol_max

def new_node(symbol_max):
    return random.randint(0, symbol_max)

def node_to_set(node):
    node_bin = [*bin(node)[2:]]
    node_set = {i + 1 for i, v in enumerate(node_bin) if v == "1"}
    return node_set


quantity_range = int(15.4 * 10**6)
symbol_range = 3000

print("Generating nodes...")
print("Quantity range: " + str(quantity_range))
print("Symbol range: " + str(symbol_range))
input("Press enter to continue...")

nodes = set()
sym_max = symbol_max(symbol_range)

for batch in range(1000):
    for i in range(0, quantity_range, 1000):
        node = new_node(sym_max)
        nodes.add(node)
    print(f"Nodes generated: {len(nodes)}")


print()
print()
print(f"Finished. Total unique nodes generated: {len(nodes)}")
print("Writing to file in 10s...")
print()
sleep(10)


# Writing to file


nodes = list(nodes)

file_name = "fake_data.txt"
with open(file_name, "w") as file:
    for batch in range(0, len(nodes), 1000):
        for i in range(batch, batch + 1000):
            file.write(str(nodes[i]) + "\n")
            
        print(f"Nodes written: {batch}")

file.close()

print()
print("Finished.")