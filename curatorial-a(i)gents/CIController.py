#!/usr/bin/python
import tkinter, time
from subprocess import Popen

Freq = 2500
Dur = 150

top = tkinter.Tk()
top.title('MapAwareness')
top.geometry('200x100') # Size 200, 200

def start():
    import os
#   os.system("python test.py")
    process = Popen(["python", "DataSensoriumTest.py"])
  

def stop():
    print ("Stop")
    Popen.kill(["python", "DataSensoriumTest.py"])
    #top.destroy() # destroy window

startButton = tkinter.Button(top, fg='black',bg='black', height=2, width=20, text ="Start", command = start)
stopButton = tkinter.Button(top, fg='black',bg='black', height=2, width=20, text ="Stop", command = stop)

startButton.pack()
stopButton.pack()
top.mainloop()

