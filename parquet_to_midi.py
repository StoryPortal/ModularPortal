"""
Load spike times from Parquet and convert to MIDI with CUSTOMIZABLE PITCHES
"""

import polars as pl
import numpy as np
import pretty_midi


# ============= CONFIGURATION =============
PARQUET_FILE = "spike_times_output.parquet"
OUTPUT_MIDI = "spikes_sonified.mid"

# Sonification parameters
TIME_SCALE = 0.1  # 0.1 = 10x faster, 1.0 = real time
NOTE_DURATION = 0.05  # Duration of each note in seconds
BASE_VELOCITY = 80  # Volume (0-127)

# PITCH MAPPING OPTIONS - Choose one:
PITCH_MODE = "custom"  # Options: 'custom', 'spread', 'octaves', 'firing_rate', 'random'

# Custom pitch mapping: Map specific unit indices to MIDI notes
# MIDI notes: 60=Middle C, 62=D, 64=E, 65=F, 67=G, 69=A, 71=B, 72=High C
CUSTOM_PITCHES = {
    0: 60,  # Unit 0 → Middle C
    1: 64,  # Unit 1 → E
    2: 67,  # Unit 2 → G
    3: 72,  # Unit 3 → High C
    4: 55,  # Unit 4 → Low G
    5: 69,  # Unit 5 → A
    # Add more units as needed, or leave blank to use default
}

# Default pitch for units not in CUSTOM_PITCHES
DEFAULT_PITCH = 60  # Middle C

# For 'spread' mode: pitch range
PITCH_RANGE = (48, 84)  # Low: C3, High: C6 (3 octaves)

# For 'octaves' mode: assign units to different octaves
OCTAVE_SPREAD = 12  # Semitones per octave

# Which units to sonify
UNITS_TO_SONIFY = None  # None = all units, or list like [0, 1, 2, 5]
MAX_UNITS = 16  # Maximum number of units
# ==========================================


def parse_spike_times(spike_times_str):
    """Convert string representation back to numpy array."""
    cleaned = spike_times_str.strip("[]")
    values = cleaned.split()
    return np.array([float(v) for v in values])


def get_pitch_for_unit(unit_idx, unit_id, spike_times, mode="custom"):
    """
    Determine pitch for a unit based on selected mode.

    Parameters:
    -----------
    unit_idx : int
        Index in the dataset (0, 1, 2, ...)
    unit_id : int
        Actual unit ID from kilosort
    spike_times : array
        Spike times for this unit
    mode : str
        Pitch assignment mode

    Returns:
    --------
    int : MIDI pitch (0-127)
    """

    if mode == "custom":
        # Use custom mapping if defined, otherwise default
        return CUSTOM_PITCHES.get(unit_idx, DEFAULT_PITCH)

    elif mode == "spread":
        # Spread evenly across pitch range
        min_pitch, max_pitch = PITCH_RANGE
        normalized = unit_idx / max(MAX_UNITS - 1, 1)
        return int(min_pitch + normalized * (max_pitch - min_pitch))

    elif mode == "octaves":
        # Spread units across octaves (C, C, C, etc.)
        base = 48  # C3
        octave = (unit_idx // 12) * 12
        note_in_octave = unit_idx % 12
        return base + octave + note_in_octave

    elif mode == "firing_rate":
        # Map firing rate to pitch (higher rate = higher pitch)
        if len(spike_times) > 0:
            duration = spike_times[-1] - spike_times[0]
            firing_rate = len(spike_times) / duration if duration > 0 else 0
            # Map 0-50 Hz to pitch range
            min_pitch, max_pitch = PITCH_RANGE
            normalized = min(firing_rate / 50.0, 1.0)
            return int(min_pitch + normalized * (max_pitch - min_pitch))
        return DEFAULT_PITCH

    elif mode == "random":
        # Random but consistent (based on unit_id)
        np.random.seed(unit_id)
        min_pitch, max_pitch = PITCH_RANGE
        return np.random.randint(min_pitch, max_pitch + 1)

    else:
        return DEFAULT_PITCH


def create_midi_from_spikes(
    parquet_file,
    output_midi,
    time_scale=1.0,
    note_duration=0.05,
    base_velocity=80,
    units_to_sonify=None,
    max_units=16,
    pitch_mode="custom",
):
    """
    Load spike times from Parquet and create a MIDI file with custom pitches.
    """

    # Load the Parquet file
    print(f"Loading spike times from {parquet_file}...")
    df = pl.read_parquet(parquet_file)
    print(f"Loaded {len(df)} units\n")

    # Filter units if specified
    if units_to_sonify is not None:
        df = df[units_to_sonify]
        print(f"Filtered to {len(df)} specified units")

    # Limit to max_units
    if len(df) > max_units:
        print(f"Limiting to first {max_units} units")
        df = df[:max_units]

    # Create MIDI object
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)

    # Process each unit
    print(f"\nPitch mode: {pitch_mode}")
    print("Processing units:")
    for idx, row in enumerate(df.iter_rows(named=True)):
        unit_id = row["unit_id"]
        spike_times_str = row["spike_times"]

        # Parse spike times
        spike_times = parse_spike_times(spike_times_str)

        # Scale time
        spike_times_scaled = spike_times * time_scale

        # Determine pitch for this unit
        pitch = get_pitch_for_unit(idx, unit_id, spike_times, mode=pitch_mode)

        # Create instrument for this unit
        program = idx % 128
        instrument = pretty_midi.Instrument(program=program, name=f"Unit_{unit_id}")

        print(
            f"  Unit {idx} (ID={unit_id}): {len(spike_times)} spikes, pitch={pitch} ({pretty_midi.note_number_to_name(pitch)}), program={program}"
        )

        # Add notes for each spike
        for spike_time in spike_times_scaled:
            note = pretty_midi.Note(
                velocity=base_velocity,
                pitch=pitch,
                start=spike_time,
                end=spike_time + note_duration,
            )
            instrument.notes.append(note)

        pm.instruments.append(instrument)

    # Save MIDI file
    print(f"\nSaving MIDI to {output_midi}...")
    pm.write(output_midi)

    # Print summary
    total_duration = max([note.end for inst in pm.instruments for note in inst.notes])
    total_notes = sum([len(inst.notes) for inst in pm.instruments])

    print(f"\n✓ MIDI file created successfully!")
    print(f"  Total units: {len(pm.instruments)}")
    print(f"  Total notes: {total_notes}")
    print(f"  Duration: {total_duration:.2f} seconds")
    print(f"  Time scale: {time_scale}x")
    print(f"  Pitch mode: {pitch_mode}")

    return pm


if __name__ == "__main__":
    print("=" * 60)
    print("Creating MIDI with custom pitches...")
    print("=" * 60)

    pm = create_midi_from_spikes(
        parquet_file=PARQUET_FILE,
        output_midi=OUTPUT_MIDI,
        time_scale=TIME_SCALE,
        note_duration=NOTE_DURATION,
        base_velocity=BASE_VELOCITY,
        units_to_sonify=UNITS_TO_SONIFY,
        max_units=MAX_UNITS,
        pitch_mode=PITCH_MODE,
    )

    print("\n" + "=" * 60)
    print("PITCH MODES AVAILABLE:")
    print("=" * 60)
    print("'custom'       - Use CUSTOM_PITCHES dictionary")
    print("'spread'       - Spread evenly across PITCH_RANGE")
    print("'octaves'      - Each unit gets different octave")
    print("'firing_rate'  - Higher firing rate = higher pitch")
    print("'random'       - Random but consistent pitches")
    print("\nChange PITCH_MODE at the top of the script!")
