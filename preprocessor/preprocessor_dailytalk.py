import os
import random
import re
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.stats import betabinom
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from g2p_en import G2p
import audio as Audio
from text import text_to_sequence, sequence_to_text, grapheme_to_phoneme
# from sentence_transformers import SentenceTransformer


class Preprocessor:
    def __init__(self, preprocess_config, model_config, train_config):
        random.seed(train_config['seed'])
        self.config = preprocess_config
        self.dataset = preprocess_config["dataset"]
        self.speakers = set()
        self.emotions = set()
        self.sub_dir = preprocess_config["path"]["sub_dir_name"]
        # self.speakers = self.load_speaker_dict()
        # self.filelist, self.emotions = self.load_filelist_dict()
        self.data_dir = preprocess_config["path"]["corpus_path"]
        self.in_dir = os.path.join(preprocess_config["path"]["raw_path"], self.sub_dir)
        self.out_dir = preprocess_config["path"]["preprocessed_path"]
        self.val_size = preprocess_config["preprocessing"]["val_size"]
        self.val_dialog_ids = self.get_val_dialog_ids()
        self.metadata = self.load_metadata()
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.beta_binomial_scaling_factor = preprocess_config["preprocessing"]["duration"]["beta_binomial_scaling_factor"]
        # self.text_embbeder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        # self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.g2p = G2p()

        assert preprocess_config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert preprocess_config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = preprocess_config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = preprocess_config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )

    def get_val_dialog_ids(self):
        data_size = len(os.listdir(self.in_dir))
        val_dialog_ids = random.sample(range(data_size), k=self.val_size)
        # print("val_dialog_ids:", val_dialog_ids)
        return val_dialog_ids

    def load_metadata(self):
        with open(os.path.join(self.data_dir, "metadata.json")) as f:
            metadata = json.load(f)
        return metadata

    def load_speaker_dict(self):
        spk_dir = os.path.join(self.config["path"]["raw_path"], 'speaker_info.txt')
        spk_dict = dict()
        with open(spk_dir, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                spk_id = line.split("|")[0]
                spk_dict[spk_id] = i
        return spk_dict
    
    def load_filelist_dict(self):
        filelist_dir = os.path.join(self.config["path"]["raw_path"], 'filelist.txt')
        filelist_dict, emotion_dict, arousal_dict, valence_dict = dict(), dict(), dict(), dict()
        emotions, arousals, valences = set(), set(), set()
        with open(filelist_dir, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                basename, aux_data = line.split("|")[0], line.split("|")[3:]
                filelist_dict[basename] = "|".join(aux_data).strip("\n")
                emotions.add(aux_data[-3])
                arousals.add(aux_data[-2])
                valences.add(aux_data[-1].strip("\n"))
        for i, emotion in enumerate(list(emotions)):
            emotion_dict[emotion] = i
        for i, arousal in enumerate(list(arousals)):
            arousal_dict[arousal] = i 
        for i, valence in enumerate(list(valences)):
            valence_dict[valence] = i 
        emotion_dict = {
            "emotion_dict": emotion_dict,
            "arousal_dict": arousal_dict,
            "valence_dict": valence_dict,
        }
        return filelist_dict, emotion_dict

    def build_from_path(self):
        # os.makedirs((os.path.join(self.out_dir, "text_emb")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch_frame")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_frame")), exist_ok=True)
        # os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "attn_prior")), exist_ok=True)

        print("Processing Data ...")
        train = list()
        val = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        # speakers = self.speakers.copy()
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            # if len(self.speakers) == 0:
            #     speakers[speaker] = i
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                # tg_path = os.path.join(
                #     self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                # )
                # if os.path.exists(tg_path):
                ret = self.process_utterance(speaker, basename)
                if ret is None:
                    continue
                else:
                    info, pitch, energy, n = ret

                if int(speaker) not in self.val_dialog_ids:
                    train.append(info)
                else:
                    val.append(info)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch_frame"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy_frame"), energy_mean, energy_std
        )

        # Save files
        # with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
        #     f.write(json.dumps(speakers))

        if len(self.speakers) != 0:
            speaker_dict = dict()
            for i, speaker in enumerate(list(self.speakers)):
                speaker_dict[speaker] = int(speaker)
            with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
                f.write(json.dumps(speaker_dict))

        if len(self.emotions) != 0:
            emotion_dict = dict()
            for i, emotion in enumerate(list(self.emotions)):
                emotion_dict[emotion] = i
            with open(os.path.join(self.out_dir, "emotions.json"), "w") as f:
                f.write(json.dumps(emotion_dict))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch_frame": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy_frame": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(train)
        train = [r for r in train if r is not None]
        val = [r for r in val if r is not None]
        # Sort validation set by dialog
        val = sorted(val, key=lambda x: (int(x.split("|")[0].split("_")[-1].lstrip("d")), int(x.split("|")[0].split("_")[0])))

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in train:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in val:
                f.write(m + "\n")

        return (train, val)

    def process_utterance(self, speaker, basename):
        # aux_data = ""
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        # tg_path = os.path.join(
        #     self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        # )
        speaker = basename.split("_")[1]
        dialog_id = basename.split("_")[-1].lstrip("d")
        uttr_id = basename.split("_")[0]
        emotion = self.metadata[dialog_id][uttr_id]["emotion"]
        if emotion == "no emotion":
            emotion = "none"
        self.speakers.add(speaker)
        self.emotions.add(emotion)
        # aux_data = self.filelist[basename]

        # # Get alignments
        # textgrid = tgt.io.read_textgrid(tg_path)
        # phone, duration, start, end = self.get_alignment(
        #     textgrid.get_tier_by_name("phones")
        # )
        # text = "{" + " ".join(phone) + "}"
        # if start >= end:
        #     return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path, self.sampling_rate)
        # wav = wav[
        #     int(self.sampling_rate * start) : int(self.sampling_rate * end)
        # ].astype(np.float32)
        wav = wav.astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")
        phone = grapheme_to_phoneme(raw_text, self.g2p)
        phones = "{" + "}{".join(phone) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        text = phones.replace("}{", " ")

        # phone = ["sp" if e == " " else e for e in grapheme_to_phoneme(raw_text, self.g2p)]
        # text = "{" + " ".join(phone) + "}"
        # print(phone)
        # print(text)
        # print(len(phone), len(text_to_sequence(phones, self.cleaners)), len(text_to_sequence(text, self.cleaners)))
        # print(sequence_to_text(text_to_sequence(text, self.cleaners)))
        # exit(0)

        # # Text embedding
        # text_emb = self.text_embbeder.encode([raw_text])[0]

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        # pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        # mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        # energy = energy[: sum(duration)]

        # if self.pitch_phoneme_averaging:
        #     # perform linear interpolation
        #     nonzero_ids = np.where(pitch != 0)[0]
        #     interp_fn = interp1d(
        #         nonzero_ids,
        #         pitch[nonzero_ids],
        #         fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        #         bounds_error=False,
        #     )
        #     pitch = interp_fn(np.arange(0, len(pitch)))

        #     # Phoneme-level average
        #     pos = 0
        #     for i, d in enumerate(duration):
        #         if d > 0:
        #             pitch[i] = np.mean(pitch[pos : pos + d])
        #         else:
        #             pitch[i] = 0
        #         pos += d
        #     pitch = pitch[: len(duration)]

        # if self.energy_phoneme_averaging:
        #     # Phoneme-level average
        #     pos = 0
        #     for i, d in enumerate(duration):
        #         if d > 0:
        #             energy[i] = np.mean(energy[pos : pos + d])
        #         else:
        #             energy[i] = 0
        #         pos += d
        #     energy = energy[: len(duration)]

        # mel_spectrogram = np.load(os.path.join(
        #     self.out_dir,
        #     "mel",
        #     "{}-mel-{}.npy".format(speaker, basename),
        # )).T

        # Compute alignment prior
        attn_prior = self.beta_binomial_prior_distribution(
            mel_spectrogram.shape[1],
            len(phone),
            self.beta_binomial_scaling_factor,
        )

        # Save files
        # text_emb_filename = "{}-text_emb-{}.npy".format(speaker, basename)
        # np.save(os.path.join(self.out_dir, "text_emb", text_emb_filename), text_emb)

        # dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        # np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        attn_prior_filename = "{}-attn_prior-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "attn_prior", attn_prior_filename), attn_prior)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch_frame", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy_frame", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text, emotion]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def beta_binomial_prior_distribution(self, phoneme_count, mel_count, scaling_factor=1.0):
        P, M = phoneme_count, mel_count
        x = np.arange(0, P)
        mel_text_probs = []
        for i in range(1, M+1):
            a, b = scaling_factor*i, scaling_factor*(M+1-i)
            rv = betabinom(P, a, b)
            mel_i_prob = rv.pmf(x)
            mel_text_probs.append(mel_i_prob)
        return np.array(mel_text_probs)

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
