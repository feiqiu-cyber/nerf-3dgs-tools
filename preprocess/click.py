import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageClickApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Click Coordinates")
        
        # Initialize dictionary to store coordinates and labels
        self.coordinates_labels = {}
        
        # Load an image
        self.load_image()
        
        self.canvas = tk.Canvas(root, width=self.img.width(), height=self.img.height())
        self.canvas.pack()
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)
        self.canvas.bind("<Button-1>", self.get_coordinates)
        
        # Bind the close window event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def load_image(self):
        file_path = filedialog.askopenfilename()
        pil_image = Image.open(file_path)
        self.img = ImageTk.PhotoImage(pil_image)
        
    def get_coordinates(self, event):
        x, y = event.x, event.y
        print(f"Clicked at: ({x}, {y})")
        
        label = self.ask_for_label(x, y)
        if label is not None:
            self.coordinates_labels[(x, y)] = label
            print(f"Label {label} assigned to coordinates: ({x}, {y})")
            tk.Label(self.root, text=f"Coordinates: ({x}, {y}) with Label: {label}").pack()
        
    def ask_for_label(self, x, y):
        label = tk.simpledialog.askinteger("Input", f"Enter label for coordinates ({x}, {y}):", minvalue=0, maxvalue=1)
        return label
        
    def on_close(self):
        print("Window closed")
        self.root.destroy()
        self.return_results()
        
    def return_results(self):
        # Here you can return the dictionary or save it to a file as needed
        return self.coordinates_labels

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClickApp(root)
    root.mainloop()
    dic = app.return_results()
    print(dic)
