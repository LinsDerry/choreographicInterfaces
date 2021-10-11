from pydub import AudioSegment
from pydub.playback import play
  

refreshSound = "audioFiles/poseSwitchSound.wav"
selectSound = "audioFiles/poseSwitchSound.wav"
trackSound = "audioFiles/poseSwitchSound.wav"
zoomInSound = None
zoomOutSound = None

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
    if action is not None:
        try:
            sound = AudioSegment.from_wav(dictionary[action])
            if sound is not None:
                play(sound)
                print('playing ' + action + 'sound')
        except Exception as e:
            print(e)

