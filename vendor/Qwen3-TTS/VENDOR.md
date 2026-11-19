# Vendored from QwenLM/Qwen3-TTS

`finetuning/dataset.py` is the only file this trainer imports from that checkout. `TTSDataset` loads the reference wav and builds the mel. The model classes it imports (`Qwen3TTSConfig`, `mel_spectrogram`) come from the installed `qwen-tts` package, not from this tree.

Pinned commit: `022e286b98fbec7e1e916cb940cdf532cd9f488e` (2026-03-17, `fix finetuning bug`).

Source: https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/dataset.py

License: Apache-2.0. The license text is `LICENSE` in this directory. The file header in `dataset.py` is the upstream copyright notice. Do not drop it.
