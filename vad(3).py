#this file is for voice recording and filter voice segmentsusing Voice Activity Dectection(VAD)
import sounddevice as sd
import numpy as np
import queue
import torch
class VoiceActivityDetector:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        print("Loading Silero VAD model...")
        self.model, _ = torch.hub.load('snakers4/silero-vad','silero_vad',force_reload=False,onnx=False)
        self.model.eval()
        self.threshold = 0.5
    def is_speech(self, audio_chunk):
        if len(audio_chunk) != 512:
            if len(audio_chunk) < 512:
                audio_chunk = np.pad(audio_chunk, (0, 512 - len(audio_chunk)))
            else:
                audio_chunk = audio_chunk[:512]
        audio_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
        with torch.no_grad():
            prob = self.model(audio_tensor, self.sample_rate).item()
        return prob > self.threshold, prob
class NoiseSuppressor:
    def process(self, audio_chunk):
        return audio_chunk
class AudioBuffer:
    def __init__(self):
        self.q = queue.Queue()
    def put(self, chunk):
        self.q.put_nowait(chunk)
    def get(self):
        try:
            return self.q.get(timeout=0.05)
        except:
            return None
class AudioStreamingClient:
    def __init__(self, rate=16000):
        self.rate = rate
        self.chunk_size = 512
        self.buffer = AudioBuffer()
        self.stream = None
    def callback(self, indata, frames, time, status):
        self.buffer.put(indata.copy())
    def start(self):
        self.stream = sd.InputStream(samplerate=self.rate,channels=1,dtype="float32",blocksize=self.chunk_size,callback=self.callback)
        self.stream.start()
    def stop(self):
        if self.stream:
            self.stream.close()
    def read_frame(self):
        return self.buffer.get()
def real_time_pipeline():
    client = AudioStreamingClient()
    vad = VoiceActivityDetector()
    ns = NoiseSuppressor()
    speech_buffer = []
    silence_frames = 0
    Max_Silence_Frames = 15
    Min_Speech_Duration = 0.2
    print("Pipeline Started. Speak into the mic (Ctrl+C to stop)")
    client.start()
    try:
        while True:
            raw = client.read_frame()
            if raw is None:
                continue
            frame = raw.flatten()
            is_speech, prob = vad.is_speech(frame)
            if not is_speech:
                silence_frames += 1
                if speech_buffer and silence_frames > Max_Silence_Frames:
                    duration = len(speech_buffer) * 0.032
                    if duration < Min_Speech_Duration:
                        print(f"\rIgnored noise {duration:.2f}s      ")
                    else:
                        print(f"\rUtterance ended ({duration:.2f}s)      ")
                        audio = np.concatenate(speech_buffer)
                        clean = ns.process(audio)
                    speech_buffer = []
                print(f"\rSilence ({prob:.2f})     ", end="")
            else:
                silence_frames = 0
                speech_buffer.append(frame)
                print(f"\rSpeech ({prob:.2f}) Recording...", end="")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        client.stop()#Run program
if __name__ == "__main__":
    real_time_pipeline()

