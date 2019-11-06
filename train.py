import os
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import matplotlib
import csv
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
import math


root = Tk()
root.title("Linear regression")

# Windows creation
windowWidth = 1200
windowHeight = 800
root.minsize(600, 400)
positionRight = int(root.winfo_screenwidth()/2 - windowWidth/2)
positionDown = int(root.winfo_screenheight()/2 - windowHeight/2)
root.geometry("1200x800+{}+{}".format(positionRight, positionDown))


# -- PREVIEW ZONE --
previewLabel = LabelFrame(root, labelanchor='nw', text='Training Preview', width=50)
previewLabel.grid(row=0, column=0, rowspan=3, columnspan=4, sticky='news')
previewLabel.grid_propagate(0)
previewCanvas = Canvas(previewLabel)
previewCanvas.pack(side=LEFT, expand=True, fill=BOTH)


# -- CONFIG --
configLabel = LabelFrame(root, labelanchor='n', text='Configuration')
configLabel.grid(row=3, column=0, rowspan=4, columnspan=4, sticky='wes')

# Comput button
pathCSV = StringVar()
pathCSVButton = Button(configLabel, text='Run Training', command=lambda: comput(previewLabel))
pathCSVButton.grid(row=1, column=1, rowspan=1, columnspan=1, sticky='w')

# Separator
configLabel.grid_columnconfigure(2, minsize=10)
ttk.Separator(configLabel, orient=VERTICAL).grid(row=1, column=3, rowspan=1, sticky='ns')
configLabel.grid_columnconfigure(4, minsize=10)

# Learning rate field
learningRateValue = StringVar(configLabel, 100)
Label(configLabel, text='Learning Rate:').grid(row=1, column=5, rowspan=1, columnspan=1, sticky='w')
learningRateEntry = Entry(configLabel, textvariable=learningRateValue)
learningRateEntry.grid(row=1, column=6, rowspan=1, columnspan=1, sticky='w')
learningRateValue.trace('w', lambda *_, var='learningRate': updateIntEvent(learningRateValue, learningRateEntry))

# Correlation coeficient
Label(configLabel, text='Correlation coef.: ').grid(row=1, column=7, rowspan=1, columnspan=1, sticky='w')
Label(configLabel, text='0').grid(row=1, column=8, rowspan=1, columnspan=1, sticky='w')

# Separator
configLabel.grid_columnconfigure(8, minsize=10)
ttk.Separator(configLabel, orient=VERTICAL).grid(row=1, column=9, rowspan=1, sticky='ns')
configLabel.grid_columnconfigure(10, minsize=10)

# Ask for a value
askForValue = StringVar(configLabel, 0)
Label(configLabel, text='Calculate a value:').grid(row=1, column=11, rowspan=1, columnspan=1, sticky='w')
askForValueEntry = Entry(configLabel, textvariable=askForValue)
askForValueEntry.grid(row=1, column=12, rowspan=1, columnspan=1, sticky='w')
askForValue.trace('w', lambda *_, var='askForValue': askValue(askForValue, askForValueEntry))

# Ok button
okString = StringVar()
okButton = Button(configLabel, text='OK', command=lambda: printEstimatePrice(int(askForValueEntry.get())))
okButton.grid(row=1, column=13, rowspan=1, columnspan=1, sticky='w')
okButton['state'] = 'disable'


def printEstimatePrice(x):
    try:
        theta = np.loadtxt("theta.csv", dtype = np.longdouble, delimiter = ',')
    except:
        return
    result = estimatePrice(theta[0], theta[1], x)
    try:
        ax.plot(x, result, color='orange', marker='o')
    except:
        return
    fig.canvas.draw()


def askValue(value, entry):
    if not value.get().isdigit() or not int(value.get()) > 0:
        entry['background'] = '#FFAAAA'
        okButton['state'] = 'disable'
    else:
        entry['background'] = '#FFFFFF'
        okButton['state'] = 'normal'


def updateIntEvent(value, entry):
    if not value.get().isdigit() or not int(value.get()) > 0:
        entry['background'] = '#FFAAAA'
        pathCSVButton['state'] = 'disable'
    else:
        entry['background'] = '#FFFFFF'
        pathCSVButton['state'] = 'normal'


def estimatePrice(theta0, theta1, mileage):
    return theta0 + theta1 * mileage


def computeGradients(x, y,amountData, theta, alpha, learningRate):
    for i in range(0, learningRate):
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

# Calculate correlation coefficient
def calculateCorrelation(data, amountData, x, y):
    tmp1 = 0
    tmp2 = 0
    tmp3 = 0
    mx = 0
    my = 0
    for j in range(0, amountData):
        mx += data[j, 0]
        my += data[j, 1]
    mx /= amountData
    my /= amountData
    for j in range(0, amountData):
        tmp1 += (data[j, 0] - mx) * (data[j, 1] - my)
        tmp2 += math.pow(data[j, 0] - mx, 2)
        tmp3 += math.pow(data[j, 1] - my, 2)
    return tmp1 / math.sqrt(tmp2 * tmp3)

# Comput trainning
def comput(previewLabel):
    theta = np.zeros((1, 2))
    alpha = 0.3
    global ax
    global fig
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
    theta = computeGradients(x, y, amountData, theta, 0.3, int(learningRateValue.get()))
    y = estimatePrice(theta[0, 0], theta[0, 1], x)
    x = destandardize(x, data[:, 0])
    y = destandardize(y, data[:, 1])
    a = (y[0] - y[1]) / (x[0] - x[1])
    b = a * x[0] * -1 + y[0]

    ax.plot(x, y, c='r')
    r = calculateCorrelation(data, amountData, b, a)
    Label(configLabel, text='Correlation coeficient: ').grid(row=1, column=7, rowspan=1, columnspan=1, sticky='w')
    if abs(r) > 0.7:
        Label(configLabel, fg='green', text=str(r)).grid(row=1, column=8, rowspan=1, columnspan=1, sticky='w')
    else:
        Label(configLabel, fg='red', text=str(r)).grid(row=1, column=8, rowspan=1, columnspan=1, sticky='w')

    graph = FigureCanvasTkAgg(fig, master=previewLabel)
    canvas = graph.get_tk_widget()
    canvas.grid(row=0, column=0)
    theta = [b, a]
    np.savetxt("theta.csv", theta, delimiter = ',');


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
