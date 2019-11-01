import os
from tkinter import *
from tkinter import filedialog
import matplotlib
import csv
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np


root = Tk()
root.title("Linear regression")
path = ''

# Windows creation
windowWidth = 1200
windowHeight = 700
root.minsize(600, 400)
positionRight = int(root.winfo_screenwidth()/2 - windowWidth/2)
positionDown = int(root.winfo_screenheight()/2 - windowHeight/2)
root.geometry("1200x700+{}+{}".format(positionRight, positionDown))


# -- PREVIEW ZONE --
previewLabel = LabelFrame(root, labelanchor='nw', text='Training Preview', width=50)
previewLabel.grid(row=0, column=0, rowspan=3, columnspan=4, sticky='news')
previewLabel.grid_propagate(0)
previewCanvas = Canvas(previewLabel)
previewCanvas.pack(side=LEFT, expand=True, fill=BOTH)


# -- CONFIG --
configLabel = LabelFrame(root, labelanchor='n', text='Configuration')
configLabel.grid(row=3, column=0, rowspan=4, columnspan=4, sticky='wes')

# CSV path
pathCSV = StringVar()
pathCSVButton = Button(configLabel, text='Run Training', command=lambda: selectPath(previewLabel))
pathCSVButton.grid(row=1, column=1, rowspan=1, columnspan=1, sticky='w')


def estimatePrice(theta0, theta1, mileage):
    return theta0 + (theta1 * mileage)


def compute_gradients(x, y, m, alpha, iterations):
    theta = np.zeros((1, 2))
    for i in range(0, iterations):
        tmp_theta = np.zeros((1, 2))
        for j in range(0, m):
            tmp_theta[0, 0] += (estimatePrice(tmp_theta[0, 0], tmp_theta[0, 1], x[j]) - y[j])
            tmp_theta[0, 1] += ((estimatePrice(tmp_theta[0, 0], tmp_theta[0, 1], x[j]) - y[j]) * x[j])
        theta -= (tmp_theta * alpha) / m
    print(theta)


# Display the path selection window and chooses a file name
def selectPath(previewLabel):
    global path
    x = []
    y = []
    theta0 = 0
    theta1 = 0
    path = filedialog.askopenfilename(title='Select a file', filetypes=[("CSV", "*.csv")])
    if not os.path.isfile(path):
        return
    fig = Figure(figsize=(13, 7), dpi=96)
    ax = fig.add_subplot()
    with open(path, newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            if row[0].isdigit():
                x.append(float(row[1]))
                y.append(float(row[0]))
            else:
                axes = fig.axes
                axes[0].text(0.5, -0.1, row[1], horizontalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='green')
                axes[0].text(-0.1, 0.5, row[0], verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='green')

    m = len(x)
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    compute_gradients(x, y, m, 0.3, 6)
    ySomme = 0
    xSomme = 0
    squareX = 0
    bSomme = 0
    for i in range(0, m):
        bSomme += (y[i] * x[i])
        ySomme += y[i]
        xSomme += x[i]
        squareX += x[i] * x[i]
        theta0 += estimatePrice(theta0, theta1, y[i]) - x[i]
        theta1 += (estimatePrice(theta0, theta1, y[i]) - x[i]) * y[i]
    b = (m * bSomme) - (ySomme * xSomme) / ((m * squareX) - (xSomme * xSomme))
    a = (ySomme / m) - (1000 * (xSomme / m))
    ax.scatter(x, y)
    l1 = lines.Line2D([0, 1], [0, 1], transform=fig.transFigure, figure=fig)
    l2 = lines.Line2D([0, 1], [1, 0], transform=fig.transFigure, figure=fig)
    fig.lines.extend([l1, l2])
    graph = FigureCanvasTkAgg(fig, master=previewLabel)
    canvas = graph.get_tk_widget()
    canvas.grid(row=0, column=0)


root.grid_columnconfigure(1, weight=4)
root.grid_rowconfigure(1, weight=4)
root.lift()
root.attributes('-topmost', True)
root.after_idle(root.attributes,'-topmost', False)


def main_loop():
    try:
        root.mainloop()
    except UnicodeDecodeError:
        main_loop()

main_loop()
