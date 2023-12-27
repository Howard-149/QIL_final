import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class EdgeWeightDialog(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.geometry("+%d+%d" % (master.winfo_rootx() + (master.winfo_width() - 200) / 2,
                                  master.winfo_rooty() + (master.winfo_height() - 100) / 2))
        self.title("Edge Weight")
        self.result = None
        self.create_widgets()

    def create_widgets(self):
        self.label = ttk.Label(self, text="Enter edge weight (between 0 and 1):")
        self.label.pack(pady=5)
        self.entry = ttk.Entry(self)
        self.entry.pack(pady=5)
        self.ok_button = ttk.Button(self, text="OK", command=self.on_ok)
        self.ok_button.pack(pady=5)
        
        # Bind the <Return> event to the on_ok method
        self.entry.bind("<Return>", lambda event: self.ok_button.invoke())
        self.ok_button.focus()  # Set focus to the OK button

        # Set focus to the input field
        self.entry.focus_set()


    def on_ok(self):
        try:
            weight = float(self.entry.get())
            if 0 <= weight <= 1:
                self.result = weight
                self.destroy()
            else:
                messagebox.showwarning("Invalid Weight", "Please enter a value between 0 and 1.")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid numeric value.")
