#this file of the code dectects the audio 
import sounddevice as sd
import numpy as np
import queue
import sys
# 1. Buffer to temporarily store microphone audio
class AudioBuffer:
    def __init__(self, max_chunks=100):
        self.queue = queue.Queue(maxsize=max_chunks)
    def put(self, chunk):
        try:
            self.queue.put_nowait(chunk)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(chunk)
            except:
                pass
    def get(self, timeout=0.1):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
# 2. Microphone Streaming Client
class AudioStreamingClient:
    def __init__(self, sample_rate=16000, frame_duration_ms=30, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = int(sample_rate * frame_duration_ms / 1000)
        self.buffer = AudioBuffer()
        self.stream = None
        self.running = False
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio stream warning:", status, file=sys.stderr)
        self.buffer.put(indata.copy())
    # Start microphone
    def start(self):
        if self.running:
            return
        self.stream = sd.InputStream(samplerate=self.sample_rate,channels=self.channels,blocksize=self.chunk_size,dtype="float32",callback=self.audio_callback)
        self.stream.start()
        self.running = True
    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.running = False
    def read_frame(self):
        return self.buffer.get()
# 3. Calculate Audio Volume in dB
def calculate_dbfs(audio, gain_boost=1.0):
    amplified_audio = audio * gain_boost
    rms = np.sqrt(np.mean(amplified_audio ** 2))
    if rms < 1e-10:
        return -100.0  # silence
    db = 20 * np.log10(rms)
    return db
# 4. Main Program
def process_audio_stream():
    client = AudioStreamingClient()
    GAIN_BOOST = 5.0         
    DETECTION_THRESHOLD = -50 
    try:
        client.start()
        print("Audio monitoring started")
        print("Listening for sounds... Press Ctrl+C to stop\n")
        while True:
            frame = client.read_frame()
            if frame is None:
                continue
            db_level = calculate_dbfs(frame, GAIN_BOOST)
            min_db = -80
            max_db = 0
            normalized = (db_level - min_db) / (max_db - min_db)
            bar_length = int(normalized * 40)
            bar_length = max(0, min(40, bar_length))
            volume_bar = "#" * bar_length
            status = ""
            if db_level > DETECTION_THRESHOLD:
                status = "<< NOISE DETECTED"
            print(f"\rLevel: {db_level:+.1f} dB | [{volume_bar:<40}] {status}",end="")
    except KeyboardInterrupt:
        print("\nStopping program...")
    finally:
        client.stop()
# Run program
if __name__ == "__main__":
    process_audio_stream()