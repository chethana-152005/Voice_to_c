#TTS modules:converting text into the natural speech
import asyncio
import edge_tts
import pygame
import time  # CHANGED: Using standard time module instead of asyncio loop

class TTSEngine:
    def __init__(self):
        print("--> Initializing TTS Engine (Edge-TTS)...")
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Voice options (Change this to change the speaker)
        # 'en-US-AriaNeural' (Female, natural)
        # 'en-US-GuyNeural' (Male, natural)
        self.voice = "en-US-AriaNeural"

    async def _generate_audio_file(self, text, output_file):
        """Async helper to generate the audio file."""
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_file)

    def generate_audio(self, text, output_filename="response.mp3"):
        """
        Converts text to speech and saves to a file.
        Returns the filename.
        """
        start_time = time.time()  # FIXED: Standard timing start
        
        # Run the async generation
        asyncio.run(self._generate_audio_file(text, output_filename))
        
        # FIXED: Standard timing end
        duration = time.time() - start_time
        print(f"--> TTS Generated in {duration:.2f} seconds.")
        return output_filename

    def speak(self, text):
        """
        Generates audio and plays it immediately.
        """
        output_file = "tts_output.mp3"
        
        # 1. Generate the file
        self.generate_audio(text, output_file)
        
        # 2. Play using pygame
        try:
            pygame.mixer.music.load(output_file)
            pygame.mixer.music.play()
            
            # Wait until playback finishes
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
        except Exception as e:
            print(f"Error playing audio: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    tts = TTSEngine()
    
    sample_text = "The fix is successful. I am now running smoothly on Python 3.12."
    
    print(f"Input: {sample_text}")
    tts.speak(sample_text)