import sys, os
import thread
import Tkinter as tk
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.backends.tkagg as tkagg


class GUI():
    def __init__(self):
        self.active = True

        self.window = tk.Tk()
        self.window.title("Real-time data")
        self.canvas = tk.Canvas(self.window, width=900, height=450)
        self.canvas.pack()

        self.quit_button = tk.Button(self.window, text="Exit", command=self.closeWindow, width=15)
        self.quit_button.place(relx=0.82, rely=0.1)

    def draw(self, figure):
        figure_canvas_agg = FigureCanvasAgg(figure)
        figure_canvas_agg.draw()
        figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
        figure_w, figure_h = int(figure_w), int(figure_h)
        self.photo = tk.PhotoImage(master=self.canvas, width=figure_w, height=figure_h)

        self.canvas.create_image(10 + figure_w / 2, 10 + figure_h / 2, image=self.photo)
        tkagg.blit(self.photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

    def closeWindow(self):
        self.active = False
        self.window.destroy()
        sys.exit(0)

    def is_active(self):
        return self.active

    def start_gui(self):
        tk.mainloop()
        self.active = True
