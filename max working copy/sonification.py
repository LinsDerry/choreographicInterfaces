from pydub import AudioSegment
from pydub.playback import play
import threading
import asyncio

  

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
