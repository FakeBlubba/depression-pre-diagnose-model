import sounddevice as sd
import numpy as np
import soundfile as sf
import assemblyai as aai
import os

def get_preferred_input_device():
    """
    Queries the system for the preferred audio input device and prints its details.

    Returns:
        dict: A dictionary containing the name, maximum input channels, and device ID of the preferred input device.
    """
    device_info = sd.query_devices(kind='input')
    print(f"Recording device name: {device_info['name']}")
    print(f"Input channels: {device_info['max_input_channels']}")
    print(f"Device ID: {device_info['device']}")
    return device_info

def record_voice(threshold=0.01, silence_duration=2, fs=44100, channels=1):
    """
    Records voice from the input device until a specified duration of silence is detected.

    Args:
        threshold (float): The amplitude below which sound is considered silence. Default is 0.01.
        silence_duration (int): The duration (in seconds) of silence after which recording stops. Default is 2 seconds.
        fs (int): Sampling frequency in Hz. Default is 44100 Hz.
        channels (int): Number of audio channels. Default is 1.

    Returns:
        ndarray: An array containing the recorded audio data.
    """
    def callback(indata, frames, time, status):
        nonlocal silent_frames, recording

        is_silent = np.mean(np.abs(indata)) < threshold

        if is_silent:
            silent_frames += frames
        else:
            silent_frames = 0

        recording.extend(indata.copy())

        if silent_frames > fs * silence_duration:
            raise sd.CallbackStop()

    silent_frames = 0
    recording = []
    print("\nSpeak loud and clear to transcript your audio. When you stop talking for {} seconds, the software will convert it.\n".format(silence_duration))
    with sd.InputStream(callback=callback, samplerate=fs, channels=channels):
        sd.sleep(int((silence_duration + 5) * 1000))  

    return np.concatenate(recording)

def save_audio_file(file_name):
    """
    Records voice, saves it to a file, and prints a message upon stopping the recording.

    Args:
        file_name (str): The name of the file where the audio will be saved.
    """
    fs = 44100
    recorded_audio = record_voice(fs=fs)
    print("Thank you, audio stopped recording, now converting in text")
    sf.write(file_name, recorded_audio, fs)

# TODO save key in .env
def transcript_audio(file_name, KEY = "2f1cb1f1178d454bbf8ffe55db3a6551"):
    """
    Transcribes audio from a file using an API and prints the transcription or error message.

    Args:
        file_name (str): The name of the audio file to be transcribed.
        KEY (str): API key used for the transcription service.

    Note:
        After transcription, the audio file is deleted from the storage.
    """

    aai.settings.api_key = KEY
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_name)

    if transcript.status == aai.TranscriptStatus.error:
        print(transcript.error)
    else:
        print(transcript.text)
    os.remove(file_name)

