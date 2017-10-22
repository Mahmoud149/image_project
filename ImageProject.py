import ttk
from Tkconstants import RIDGE, X, BOTTOM, N, S, E, W, NE, BOTH, TOP, NW
from Tkinter import Tk, Canvas
import numpy as np
import tkMessageBox
from tkFileDialog import askopenfilename

from matplotlib.pyplot import imshow, show, figure, bar, subplot
import tkSimpleDialog
import cv2
import Tkinter
from PIL import Image, ImageTk


class image:
    root = Tk()
    root.title("Project")
    root.geometry("660x650")
    image_frame = ttk.Frame(root)
    but_frame = ttk.Frame(root)
    button_numbers = list()

    def __init__(self):
        self.add_widgets()
        self.F = 0
        self.ker = np.array([[0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0]])

        self.Km = np.array([[1. / 9, 1. / 9, 1. / 9],
                            [1. / 9, 1. / 9, 1. / 9],
                            [1. / 9, 1. / 9, 1. / 9]])

    def browse(self):
        for i in xrange(1, 4):
            Fname = askopenfilename(filetypes=(("JPG files", "*.jpg;*.jpeg"),
                                               ("PNG files", "*.png"),
                                               ("All files", "*.*")))
            if ('.jpg' or '.jpeg' or '.png') in Fname:
                self.im = cv2.imread(Fname)
                self.im = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
                self.imp = Image.fromarray(self.im)
                imshow(self.imp)
                show()
                im = Image.fromarray(self.im)
                self.imgtk = ImageTk.PhotoImage(image=im)
                self.canvas.create_image(2, 2, image=self.imgtk, anchor=NW)
                show()
                self.F = 1
                self.gim = cv2.cvtColor(self.im, cv2.COLOR_RGB2GRAY)
                imshow(self.gim, cmap='Greys_r')
                show()
                break
            else:
                tkMessageBox.showerror(title="TypeError",
                                       message='Please Input File Image\nRemaining Time  {}'.format(3 - i))

    def sh(self, **kwargs):
        if self.F:
            if 'cmap' not in kwargs:
                kwargs['cmap'] = 'gray'
                figure()
                imshow(self.im, interpolation='none', **kwargs)
                show()
        else:
            tkMessageBox.showerror(title="TypeError",
                                   message='Please Select File Image')

    def his(self):
        if self.F:
            Histogram = np.zeros((256))
            [r, c] = self.gim.shape
            for i in xrange(r):
                for j in xrange(c):
                    if self.gim[i, j] < 1:
                        continue
                    else:
                        t = self.gim[i, j]
                    Histogram[t] += 1
                    # print Histogram
            # Hist = Histogram.tolist()
            range = [x for x in xrange(len(Histogram))]
            # p = collections.Counter(Hist)
            # print p
            bar(range, Histogram, fc='k', ec='k')
            show()
            # hist(self.gim.ravel(), bins=256, range=(0,255), fc='k', ec='k')
            # show()
        else:
            tkMessageBox.showerror(title="TypeError",
                                   message='Please Select File Image')

    def setmask(self):
        self.ker = np.array([[0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0]])
        for i in xrange(3):
            for j in xrange(3):
                self.ker[i, j] = tkSimpleDialog.askinteger('Please Insert Value',
                                                           'Insert Element {},{} : '.format(i + 1, j + 1))
        print self.ker

    def convolve(self, image, row, col, Kernal):
        temp = image[row - 1:row - 1 + 3, col - 1:col - 1 + 3]
        re = temp * Kernal
        conv = sum(sum(re))
        return conv

    def filt(self):
        if self.F:
            self.setmask()
            [r, c] = self.gim.shape
            r %= 3
            c %= 3
            if r == 0:
                r = 1
            if c == 0:
                c = 1
            np.pad(self.gim, [(r, r), (c, c)], mode='constant')
            output = np.zeros(self.gim.shape, dtype=np.uint8)
            (r1, c1) = output.shape
            for i in xrange(1, r1 - r):
                for j in xrange(1, c1 - c):
                    output[i, j] = self.convolve(self.gim, i, j, self.ker)
            imshow(output, cmap='Greys_r')
            show()
            return output
        else:
            tkMessageBox.showerror(title="TypeError",
                                   message='Please Select File Image')

    def filter(self, im):
        if self.F:
            self.Km = np.array([[1. / 9, 1. / 9, 1. / 9],
                                [1. / 9, 1. / 9, 1. / 9],
                                [1. / 9, 1. / 9, 1. / 9]])
            [r, c] = im.shape
            r %= 3
            c %= 3
            if r == 0:
                r = 1
            if c == 0:
                c = 1
            # np.pad(x, [(0, 0), (0, 1)], mode='constant')
            np.pad(im, [(r, r), (c, c)], mode='constant')
            output = np.zeros(im.shape, dtype=np.uint8)
            (r1, c1) = output.shape
            for i in xrange(1, r1 - r):
                for j in xrange(1, c1 - c):
                    output[i, j] = self.convolve(im, i, j, self.Km)
            return output
        else:
            tkMessageBox.showerror(title="TypeError",
                                   message='Please Select File Image')

    def filtm(self, im):
        if self.F:
            [R, G, B] = cv2.split(im)
            R1 = self.filter(R)
            G1 = self.filter(G)
            B1 = self.filter(B)
            out = cv2.merge((R1, G1, B1))
            imshow(out, cmap='Greys_r')
            show()
            return out
        else:
            tkMessageBox.showerror(title="TypeError",
                                   message='Please Select File Image')
            # ndimage.binary_fill_holes(a).astype(int)

    def gradiant(self):
        if self.F:
            x = self.filt()
            y = self.filt()
            imshow(x, cmap='Greys_r')
            show()
            imshow(y, cmap='Greys_r')
            show()
        else:
            tkMessageBox.showerror(title="TypeError",
                                   message='Please Select File Image')

    def edge(self):
        if self.F:
            x = self.filt()
            y = self.filt()
            output = np.zeros(y.shape, dtype=np.uint8)
            [r, c] = x.shape
            for i in xrange(r):
                for j in xrange(c):
                    gx = x[i, j]
                    gy = y[i, j]
                    g = ((gx ** 2) + (gy ** 2)) ** .5
                    if g > 30:
                        output[i, j] = 255
                    else:
                        output[i, j] = 0
            imshow(~output, cmap='Greys_r')
            show()
            return output
        else:
            tkMessageBox.showerror(title="TypeError",
                                   message='Please Select File Image')

    def binmask(self):
        if self.F:
            b = self.edge()
            output = np.zeros(self.im.shape, dtype=np.uint8)
            [r, c] = b.shape
            w = [255, 255, 255]
            mask = b == 255
            for i in xrange(r):
                for j in xrange(c):
                    if mask[i, j]:
                        output[i, j] = self.im[i, j]
                    else:
                        output[i, j] = w
            subplot(121)
            imshow(output)
            subplot(122)
            output = self.filtm(output)
            show()
            return output

    def exit(self):
        exit()

    def add_widgets(self):
        self.canvas = Canvas(width=500, height=530)
        self.canvas.pack(padx=10, pady=10, fill=X)

        self.but_frame.config(height=50)
        self.but_frame.config(relief=RIDGE)
        self.but_frame.config(width=150, height=200)
        self.but_frame.pack(padx=10, pady=10, side=BOTTOM)

        for i in range(8):
            self.button_numbers.append(ttk.Button(self.but_frame))

        count = 0
        for i in range(1):
            for j in range(8):
                self.button_numbers[count].grid(row=i, column=j, sticky=N + S + E + W)
                count += 1

        self.button_numbers[0].config(text="Browse", command=self.browse)
        self.button_numbers[1].config(text="Calculate Gradient", command=self.gradiant)
        self.button_numbers[2].config(text="Get Edges", command=self.edge)
        self.button_numbers[3].config(text="General filter", command=self.filt)
        self.button_numbers[4].config(text="Histogram", command=self.his)
        self.button_numbers[5].config(text="Show Image", command=self.sh)
        self.button_numbers[6].config(text="Binary Mask", command=self.binmask)
        self.button_numbers[7].config(text="Exit", command=self.exit)

        self.root.minsize(400, 400)
        self.root.mainloop()


m = image()
# m.sh(m.im)
# m.his()
# m.filt()
# m.edge()
# m.binmask()
