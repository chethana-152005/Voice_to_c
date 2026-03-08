# This program is used to convert live speech from a microphone into text.
import sounddevice as sd
import numpy as np
import queue
import torchgit 
from faster_whisper import WhisperModel
# Voice Activity Detection
class VAD:
    def __init__(self):
        self.sample_rate = 16000
        self.model, _ = torch.hub.load("snakers4/silero-vad","silero_vad",force_reload=False)
        self.model.eval()
        self.threshold = 0.5

    def detect(self, audio):
        if len(audio) < 512:
            return False, 0

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

        with torch.no_grad():
            prob = self.model(audio_tensor, self.sample_rate)

        return prob.item() > self.threshold, prob.item()

# Whisper Speech Recognition
class SpeechRecognizer:
    def __init__(self):
        print("Loading Whisper model...")
        self.model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Model ready")

    def transcribe(self, audio):
        segments, info = self.model.transcribe(audio, beam_size=5)

        text_output = []
        for seg in segments:
            text_output.append(seg.text.strip())

        return text_output, info.language
# Audio Streaming
class MicStream:
    def __init__(self):
        self.rate = 16000
        self.chunk = 512
        self.q = queue.Queue()
        self.stream = sd.InputStream(samplerate=self.rate,channels=1,dtype="float32",blocksize=self.chunk,callback=self.callback)
    def callback(self, indata, frames, time, status):
        self.q.put(indata.copy())
    def start(self):
        self.stream.start()
    def stop(self):
        self.stream.close()
    def read(self):
        try:
            return self.q.get(timeout=0.05)
        except:
            return None
# Main Program
def run():
    mic = MicStream()
    vad = VAD()
    asr = SpeechRecognizer()
    speech_data = []
    silence = 0
    max_silence = 15
    print("\nSpeak something. I'll convert it to text.")
    print("Press Ctrl+C to stop\n")
    mic.start()
    try:
        while True:
            frame = mic.read()
            if frame is None:
                continue
            frame = frame.flatten()
            speech, prob = vad.detect(frame)
            if speech:
                speech_data.append(frame)
                silence = 0
                print("\rListening...", end="")

            else:
                silence += 1
                if len(speech_data) > 0 and silence > max_silence:
                    audio = np.concatenate(speech_data)
                    print("\nProcessing speech...")
                    text, lang = asr.transcribe(audio)
                    print("Language:", lang)
                    for line in text:
                        print("Text:", line)
                    print("-" * 25)
                    speech_data = []

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        mic.stop()
if __name__ == "__main__":
    run()