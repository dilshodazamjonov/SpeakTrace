import torchaudio


def load_audio(path):
    """
    Loads audio file and gets the waveform and it's [channels, samples] shapes.
    """

    waveform, sample_rate = torchaudio.load(path)

    return waveform.shape, sample_rate



    



