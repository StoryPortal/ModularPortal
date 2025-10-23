"""
Convert MIDI file to WAV audio using pretty_midi
"""

import pretty_midi
import scipy.io.wavfile
import numpy as np

# ============= CONFIGURATION =============
MIDI_FILE = "spikes_sonified.mid"
OUTPUT_WAV = "spikes_sonified.wav"
SAMPLE_RATE = 44100  # CD quality
# ==========================================


def midi_to_wav(midi_file, output_wav, sample_rate=44100):
    """
    Convert MIDI to WAV audio file.

    Parameters:
    -----------
    midi_file : str
        Path to input MIDI file
    output_wav : str
        Path to output WAV file
    sample_rate : int
        Audio sample rate (44100 = CD quality)
    """
    print(f"Loading MIDI file: {midi_file}")
    pm = pretty_midi.PrettyMIDI(midi_file)

    print(f"Synthesizing audio (this may take a moment)...")
    audio_data = pm.fluidsynth(fs=sample_rate)

    # Normalize audio to prevent clipping
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)

    print(f"Saving to {output_wav}...")
    scipy.io.wavfile.write(output_wav, sample_rate, audio_data)

    duration = len(audio_data) / sample_rate
    print(f"\n✓ Audio file created!")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  File: {output_wav}")
    print(f"\nYou can now play this in any audio player!")


if __name__ == "__main__":
    midi_to_wav(MIDI_FILE, OUTPUT_WAV, SAMPLE_RATE)
