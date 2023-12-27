import networkx as nx
import tkinter as tk
from tkinter import ttk
import random
from .edge_weight_dialog import EdgeWeightDialog 

class GraphBuilderApp:
    def __init__(self, root, callback=None):
        self.root = root
        self.root.title("Graph Builder")

        self.graph = nx.Graph()

        self.canvas = tk.Canvas(root, bg="white")
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.canvas.bind("<Button-1>", self.add_node)
        self.canvas.bind("<B1-Motion>", self.drag_edge_start)
        self.canvas.bind("<ButtonRelease-1>", self.drag_edge_end)

        self.root.bind("<Configure>", self.on_window_resize)

        self.current_edge = None
        self.drag_start_node = None

        self.clear_button = ttk.Button(root, text="Clear Graph", command=self.clear_graph)
        self.clear_button.pack(pady=5)

        self.random_weight_var = tk.BooleanVar()
        self.random_weight_var.set(False)  # Default: Manual entry
        self.random_weight_checkbox = ttk.Checkbutton(root, text="Random Weight", variable=self.random_weight_var)
        self.random_weight_checkbox.pack(pady=5)
        
        self.callback = callback

    def on_window_resize(self, event):
        self.draw_graph()

    def add_node(self, event):
        x, y = event.x, event.y
        node = self.find_node(x, y)
        if node is None:
            node_id = len(self.graph.nodes)
            self.graph.add_node(node_id, pos=(x, y))
            self.draw_graph()

    def drag_edge_start(self, event):
        if self.drag_start_node is None:
            # Find the node under the cursor
            node = self.find_node(event.x, event.y)
            if node is not None:
                self.drag_start_node = node
                self.current_edge = self.canvas.create_line(
                    self.graph.nodes[node]['pos'][0], self.graph.nodes[node]['pos'][1],
                    event.x, event.y, fill="black"
                )

    def drag_edge_end(self, event):
        if self.drag_start_node is not None:
            # Find the node under the cursor
            end_node = self.find_node(event.x, event.y)
            if end_node is not None and end_node != self.drag_start_node:
                if self.random_weight_var.get():
                    weight = random.uniform(0, 1)
                else:
                    weight = self.get_valid_edge_weight()
                if weight is not None:
                    # Ensure that the node with the smaller index is always in front
                    start_node, end_node = sorted([self.drag_start_node, end_node])
                    self.graph.add_edge(start_node, end_node, weight=weight)
                    self.draw_graph()

        self.canvas.delete(self.current_edge)
        self.current_edge = None
        self.drag_start_node = None
        self.dragging = False

    def get_valid_edge_weight(self):
        dialog = EdgeWeightDialog(self.root)
        self.root.wait_window(dialog)
        return dialog.result 


    def find_node(self, x, y):
        for node, data in self.graph.nodes(data=True):
            nx, ny = data['pos']
            if x - 10 <= nx <= x + 10 and y - 10 <= ny <= y + 10:
                return node
        return None

    def clear_graph(self):
        self.graph.clear()
        self.canvas.delete("all")

    def draw_graph(self):
        self.canvas.delete("all")

        # Draw nodes
        for node, data in self.graph.nodes(data=True):
            x, y = data['pos']
            self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill="blue", outline="black")
            self.canvas.create_text(x, y, text=str(node), fill="white")

        # Draw edges with weights
        for edge in self.graph.edges(data=True):
            x1, y1 = self.graph.nodes[edge[0]]['pos']
            x2, y2 = self.graph.nodes[edge[1]]['pos']
            weight = edge[2].get('weight', '')
            weight_text = "{:.3f}".format(weight) if weight else ''
            self.canvas.create_line(x1, y1, x2, y2, fill="black")
            self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=weight_text, fill="black")

        if self.callback:
            self.callback(self.graph)

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphBuilderApp(root)
    root.mainloop()
