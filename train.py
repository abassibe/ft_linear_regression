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

# Start button
pathCSV = StringVar()
pathCSVButton = Button(configLabel, text='Run Training', command=lambda: start(previewLabel))
pathCSVButton.grid(row=1, column=1, rowspan=1, columnspan=1, sticky='w')


def estimatePrice(theta0, theta1, mileage):
    return theta0 + theta1 * mileage


def computeGradients(x, y,amountData, theta, alpha, iterations):
    for i in range(0, iterations):
        tmpTheta = np.zeros((1, 2))
        for j in range(0, amountData):
            tmpTheta[0, 0] += (estimatePrice(theta[0, 0], theta[0, 1], x[j]) - y[j])
            tmpTheta[0, 1] += ((estimatePrice(theta[0, 0], theta[0, 1], x[j]) - y[j]) * x[j])
        theta -= (tmpTheta * alpha) /amountData
    return theta

def standardize(value):
    return (value - np.mean(value)) / np.std(value)

def destandardize(x, value):
    return x * np.std(value) + np.mean(value)

# Start trainning
def start(previewLabel):
    theta = np.zeros((1, 2))
    alpha = 0.3
    data = np.loadtxt("data.csv", dtype = np.longdouble, delimiter = ',', skiprows = 1)
    if (len(data) < 2):
        exit()
    fig = Figure(figsize=(13, 7), dpi=96)
    ax = fig.add_subplot()
    axes = fig.axes
    axes[0].text(0.5, -0.1, 'km', horizontalalignment='center',
        transform=ax.transAxes, fontsize=12, color='green')
    axes[0].text(-0.1, 0.5, 'price', verticalalignment='center',
        transform=ax.transAxes, fontsize=12, color='green')

    ax.scatter(data[:, 0], data[:, 1])
    x = standardize(data[:, 0])
    y = standardize(data[:, 1])
    amountData = len(x)
    theta = computeGradients(x, y, amountData, theta, 0.3, 200)
    y = estimatePrice(theta[0, 0], theta[0, 1], x)
    x = destandardize(x, data[:, 0])
    y = destandardize(y, data[:, 1])
    a = (y[0] - y[1]) / (x[0] - x[1])
    b = a * x[0] * -1 + y[0]
    theta = [b, a]

    ax.plot(x, y)

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
