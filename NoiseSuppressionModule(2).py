#this module performs noise suppression on the audio,and remove uneccesary background voices and noise and make the audio clear and clean.
import sounddevice as sd
import numpy as np
import queue
import torch
import sys
# 1. Noise Suppression Module
class NoiseSuppressionModule:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.device = torch.device("cpu")
        self.model = None

        print("Loading Noise Suppression Model...")

        try:
            from denoiser import pretrained
            self.model = pretrained.dns64().to(self.device)
            self.model.eval()
            print("Deep Learning Denoiser Loaded.")

        except Exception as e:
            print("Could not load denoiser. Using simple noise filter.")
            self.model = None
            self.noise_profile = None

    def process(self, audio_chunk):
        if self.model is not None:
            try:
                audio_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    clean_audio = self.model(audio_tensor.to(self.device))[0]
                return clean_audio.squeeze().cpu().numpy()
            except:
                return audio_chunk
        else:
            if self.noise_profile is None:
                self.noise_profile = np.mean(np.abs(audio_chunk))
            energy = np.sqrt(np.mean(audio_chunk ** 2))
            if energy < self.noise_profile * 1.5:
                return audio_chunk * 0.1
            return audio_chunk
# 2. Audio Buffer
class AudioBuffer:
    def __init__(self, max_chunks=100):
        self.queue = queue.Queue(maxsize=max_chunks)
    def put(self, chunk):
        try:
            self.queue.put_nowait(chunk)
        except:
            pass
    def get(self):
        try:
            return self.queue.get(timeout=0.1)
        except:
            return None
# 3. Audio Streaming Client
class AudioStreamingClient:
    def __init__(self, sample_rate=16000, frame_duration_ms=30):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * frame_duration_ms / 1000)
        self.buffer = AudioBuffer()
        self.stream = None
    def audio_callback(self, indata, frames, time_info, status):
        self.buffer.put(indata.copy())
    def start(self):
        self.stream = sd.InputStream(samplerate=self.sample_rate,channels=1,blocksize=self.chunk_size,dtype="float32",callback=self.audio_callback,)
        self.stream.start()
    def stop(self):
        if self.stream:
            self.stream.close()
    def read_frame(self):
        return self.buffer.get()
# 4. Calculate Audio Loudness
def calculate_dbfs(audio, gain_boost=1.0):
    amplified = audio * gain_boost
    rms = np.sqrt(np.mean(amplified ** 2))
    if rms < 1e-10:
        return -100.0
    return 20 * np.log10(rms)
# 5. Main Audio Pipeline
def main_audio_pipeline():
    client = AudioStreamingClient()
    noise_filter = NoiseSuppressionModule()
    GAIN_BOOST = 5.0
    DETECTION_THRESHOLD_DB = -50
    print("\n--- AUDIO PIPELINE STARTED ---")
    print("1. Capture Audio -> 2. Remove Noise -> 3. Detect Voice\n")
    client.start()
    try:
        while True:
            raw_frame = client.read_frame()
            if raw_frame is None:
                continue
            # Step 1 : Remove Noise
            clean_frame = noise_filter.process(raw_frame.flatten())
            # Step 2 : Measure Volume
            db_level = calculate_dbfs(clean_frame, GAIN_BOOST)
            # Step 3 : Create Volume Bar
            min_db = -80
            max_db = 0
            normalized = (db_level - min_db) / (max_db - min_db)
            bar_length = int(normalized * 40)
            bar_length = max(0, min(40, bar_length))
            volume_bar = "#" * bar_length
            # Step 4 : Voice Detection
            status = ""
            if db_level > DETECTION_THRESHOLD_DB:
                status = "<< VOICE DETECTED"
            elif db_level > -90:
                status = "(Noise Removed)"
            # Step 5 : Display Output
            print(f"\rClean Level: {db_level:+.1f} dB | [{volume_bar:<40}] {status}",end="")
    except KeyboardInterrupt:
        print("\nStopping pipeline...")

    finally:
        client.stop()
# Run Program
if __name__ == "__main__":
    main_audio_pipeline()