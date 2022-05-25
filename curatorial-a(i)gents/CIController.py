#!/usr/bin/python
import Tkinter, time
from subprocess import Popen

Freq = 2500
Dur = 150

top = Tkinter.Tk()
top.title('MapAwareness')
top.geometry('200x100') # Size 200, 200

def start():
    import os
#   os.system("python test.py")
    Popen(["python", "DataSensoriumTest.py"])

def stop():
    print ("Stop")
    top.destroy()

startButton = Tkinter.Button(top, height=2, width=20, text ="Start", 
command = start)
stopButton = Tkinter.Button(top, height=2, width=20, text ="Stop", 
command = stop)

startButton.pack()
stopButton.pack()
top.mainloop()