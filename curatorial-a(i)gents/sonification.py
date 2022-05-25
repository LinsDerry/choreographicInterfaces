from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 7777)

def sendOSCMessage(param, value):
    client.send_message(param, value)
    # print("sending OSC message")

def sendXAccelerationOSC(xAcceleration):
    action = 'cursorAcceleration'
    client.send_message(action, xAcceleration)

def sendMouseDown():
    action = 'mouseDown'
    client.send_message(action, True)
def sendMouseUp():
    action = 'mouseDown'
    client.send_message(action, False)

#initialize
sendOSCMessage('refresh', '') #initialize patch
sendOSCMessage('neutral', '') #initialize patch
sendMouseUp()
