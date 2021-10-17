from pydub import AudioSegment
from pydub.playback import play
import threading
import asyncio

# import argparse
# import random
# import time

# from pythonosc import udp_client


# if __name__ == "__main__":
#   parser = argparse.ArgumentParser()
#   parser.add_argument("--ip", default="127.0.0.1",
#       help="The ip of the OSC server")
#   parser.add_argument("--port", type=int, default=5005,
#       help="The port the OSC server is listening on")
#   args = parser.parse_args()

#   client = udp_client.SimpleUDPClient(args.ip, args.port)

#   for x in range(10):
#     client.send_message("/filter", random.random())
#     time.sleep(1)

from pythonosc.dispatcher import Dispatcher
from typing import List, Any

dispatcher = Dispatcher()


def set_filter(address: str, *args: List[Any]) -> None:
    # We expect two float arguments
    if not len(args) == 2 or type(args[0]) is not float or type(args[1]) is not float:
        return

    # Check that address starts with filter
    if not address[:-1] == "/filter":  # Cut off the last character
        return

    value1 = args[0]
    value2 = args[1]
    filterno = address[-1]
    print(f"Setting filter {filterno} values: {value1}, {value2}")


dispatcher.map("/filter*", set_filter)  # Map wildcard address to set_filter function

# Set up server and client for testing
from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 7777)

# Send message and receive exactly one message (blocking)
#client.send_message("/filter1", [1., 2.])

#client.send_message("/filter8", [6., -2.])

def sendOSCMessage(param, value):
    #client.send_message("/filter1", [1., 2.])#og example
    client.send_message(param, value)

    print("sending OSC message")


  

refreshSound = "audioFiles/refresh.wav"
selectSound = "audioFiles/select.wav"
trackSound = "audioFiles/poseSwitchSound.wav"
zoomInSound = "audioFiles/zoomIn.wav"
zoomOutSound = "audioFiles/zoomOut.wav"

dictionary = {
    'refresh': refreshSound,
    'select': selectSound,
    'track': trackSound,
    'zoomIn': zoomInSound,
    'zoomOut': zoomOutSound,
    'neutral': None
}

def playPoseSound(action: str):
    print('attempting to play sound')
    #task = asyncio.create_task(playSound(action))
    if action is not None:
        try:
            sound = AudioSegment.from_wav(dictionary[action])
            if sound is not None:
                play(sound)
                print('playing ' + action + 'sound')
        except Exception as e:
            print(e)
# async def playSound(action: str):
#     if action is not None:
#         try:
#             sound = AudioSegment.from_wav(dictionary[action])
#             if sound is not None:
#                 play(sound)
#                 print('playing ' + action + 'sound')
#         except Exception as e:
#             print(e)
