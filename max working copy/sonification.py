from pythonosc.udp_client import SimpleUDPClient


client = SimpleUDPClient("127.0.0.1", 7777)




def sendOSCMessage(param, value):
    client.send_message(param, value)
    #print("sending OSC message")

def sendXAccelerationOSC(xAcceleration):
    action = 'cursorAcceleration'
    client.send_message(action, xAcceleration)

#BELOW FROM OLD PY ENDOGENOUS APPROACH

# refreshSound = "audioFiles/refresh.wav"
# selectSound = "audioFiles/select.wav"
# trackSound = "audioFiles/poseSwitchSound.wav"
# zoomInSound = "audioFiles/zoomIn.wav"
# zoomOutSound = "audioFiles/zoomOut.wav"

# dictionary = {
#     'refresh': refreshSound,
#     'select': selectSound,
#     'track': trackSound,
#     'zoomIn': zoomInSound,
#     'zoomOut': zoomOutSound,
#     'neutral': None
# }

# def playPoseSound(action: str):
#     print('attempting to play sound')
#     #task = asyncio.create_task(playSound(action))
#     if action is not None:
#         try:
#             sound = AudioSegment.from_wav(dictionary[action])
#             if sound is not None:
#                 play(sound)
#                 print('playing ' + action + 'sound')
#         except Exception as e:
#             print(e)
# async def playSound(action: str):
#     if action is not None:
#         try:
#             sound = AudioSegment.from_wav(dictionary[action])
#             if sound is not None:
#                 play(sound)
#                 print('playing ' + action + 'sound')
#         except Exception as e:
#             print(e)
