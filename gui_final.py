import tkinter as tk
from PIL import Image, ImageTk
import cv2
hi = 0
class Layout():
    def __init__(self, window):
        self.window = window
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        self.b1 = tk.Button(self.window, activebackground='black', activeforeground='red', bg='red', fg='white', text="Start/Stop", height=int(screen_height/80), width=int(screen_width/20), border=3)
        self.b2 = tk.Button(self.window, activebackground='black', activeforeground='red', bg='red', fg='white', text="Report", height=int(screen_height/80), width=int(screen_width/20),border=3)
        self.b3 = tk.Button(self.window, text="Save Clip", activebackground='black', activeforeground='red', bg='red', fg='white', height=int(screen_height/80), width=int(screen_width/20), border=3)
        self.b1.pack(padx=5,pady=5,side=tk.RIGHT)
        self.b2.pack(padx=5,pady=5,side=tk.RIGHT)
        self.b3.pack(padx=5,pady=5,side=tk.RIGHT)
class MainWindow():
    def __init__(self, window, cap):
        self.window = window
        self.cap = cap  
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        hi = self.height
        self.interval = 5 
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height+200)
        self.canvas.pack(padx=5,pady=5,side=tk.TOP)
        self.update_image()
    def update_image(self):
        self.image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB) # to RGB
        self.image = Image.fromarray(self.image) # to PIL format
        self.image = ImageTk.PhotoImage(self.image) # to ImageTk format
        # Update image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        # Repeat every 'interval' ms
        self.window.after(self.interval, self.update_image)
if __name__ == "__main__":
    root = tk.Tk()
    #p1 = tk.PhotoImage(file = '/Users/mohit/OneDrive/Documents/CAPSTONE/icon.png')
    #root.iconphoto(False, p1)
    root.title("Real Time CCTV Footage Analysis tool")
    MainWindow(root, cv2.VideoCapture(0))
    Layout(root)
    root.mainloop()