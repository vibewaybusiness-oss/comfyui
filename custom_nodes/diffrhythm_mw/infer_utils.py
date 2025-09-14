# Copyright (c) 2025 ASLP-LAB
#               2025 Huakang Chen  (huakang@mail.nwpu.edu.cn)
#               2025 Guobin Ma     (guobin.ma@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import librosa
import torchaudio
import random
import json
from muq import MuQMuLan, MuQ
from omegaconf import OmegaConf
from safetensors.torch import load_file
from hydra.utils import instantiate
from mutagen.mp3 import MP3
import os
import numpy as np

from sys import path
path.append(os.getcwd())

from diffrhythm.model import DiT, CFM


node_dir = os.path.dirname(os.path.abspath(__file__))

import folder_paths
models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS")

def vae_sample(mean, scale):
    stdev = torch.nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean

    kl = (mean * mean + var - logvar - 1).sum(1).mean()

    return latents, kl

def normalize_audio(y, target_dbfs=0):
    max_amplitude = torch.max(torch.abs(y))

    target_amplitude = 10.0**(target_dbfs / 20.0)
    scale_factor = target_amplitude / max_amplitude

    normalized_audio = y * scale_factor

    return normalized_audio

def set_audio_channels(audio, target_channels):
    if target_channels == 1:
        # Convert to mono
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        # Convert to stereo
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio

class PadCrop(torch.nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output

def prepare_audio(audio, in_sr, target_sr, target_length, target_channels, device):
    
    audio = audio.to(device)

    if in_sr != target_sr:
        resample_tf = torchaudio.transforms.Resample(in_sr, target_sr).to(device)
        audio = resample_tf(audio)
    if target_length is None:
        target_length = audio.shape[-1]
    audio = PadCrop(target_length, randomize=False)(audio)

    # Add batch dimension
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)

    audio = set_audio_channels(audio, target_channels)

    return audio

def decode_audio(latents, vae_model, chunked=False, overlap=32, chunk_size=128):
    downsampling_ratio = 2048
    io_channels = 2
    if not chunked:
        return vae_model.decode_export(latents)
    else:
        # chunked decoding
        hop_size = chunk_size - overlap
        total_size = latents.shape[2]
        batch_size = latents.shape[0]
        chunks = []
        i = 0
        for i in range(0, total_size - chunk_size + 1, hop_size):
            chunk = latents[:, :, i : i + chunk_size]
            chunks.append(chunk)
        if i + chunk_size != total_size:
            # Final chunk
            chunk = latents[:, :, -chunk_size:]
            chunks.append(chunk)
        chunks = torch.stack(chunks)
        num_chunks = chunks.shape[0]
        # samples_per_latent is just the downsampling ratio
        samples_per_latent = downsampling_ratio
        # Create an empty waveform, we will populate it with chunks as decode them
        y_size = total_size * samples_per_latent
        y_final = torch.zeros((batch_size, io_channels, y_size)).to(latents.device)
        for i in range(num_chunks):
            x_chunk = chunks[i, :]
            # decode the chunk
            y_chunk = vae_model.decode_export(x_chunk)
            # figure out where to put the audio along the time domain
            if i == num_chunks - 1:
                # final chunk always goes at the end
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = i * hop_size * samples_per_latent
                t_end = t_start + chunk_size * samples_per_latent
            #  remove the edges of the overlaps
            ol = (overlap // 2) * samples_per_latent
            chunk_start = 0
            chunk_end = y_chunk.shape[2]
            if i > 0:
                # no overlap for the start of the first chunk
                t_start += ol
                chunk_start += ol
            if i < num_chunks - 1:
                # no overlap for the end of the last chunk
                t_end -= ol
                chunk_end -= ol
            # paste the chunked audio into our y_final output audio
            y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]
        return y_final

def encode_audio(audio, vae_model, chunked=False, overlap=32, chunk_size=128):
    downsampling_ratio = 2048
    latent_dim = 128
    if not chunked:
        # default behavior. Encode the entire audio in parallel
        return vae_model.encode_export(audio)
    else:
        # CHUNKED ENCODING
        # samples_per_latent is just the downsampling ratio (which is also the upsampling ratio)
        samples_per_latent = downsampling_ratio
        total_size = audio.shape[2] # in samples
        batch_size = audio.shape[0]
        chunk_size *= samples_per_latent # converting metric in latents to samples
        overlap *= samples_per_latent # converting metric in latents to samples
        hop_size = chunk_size - overlap
        chunks = []
        for i in range(0, total_size - chunk_size + 1, hop_size):
            chunk = audio[:,:,i:i+chunk_size]
            chunks.append(chunk)
        if i+chunk_size != total_size:
            # Final chunk
            chunk = audio[:,:,-chunk_size:]
            chunks.append(chunk)
        chunks = torch.stack(chunks)
        num_chunks = chunks.shape[0]
        # Note: y_size might be a different value from the latent length used in diffusion training
        # because we can encode audio of varying lengths
        # However, the audio should've been padded to a multiple of samples_per_latent by now.
        y_size = total_size // samples_per_latent
        # Create an empty latent, we will populate it with chunks as we encode them
        y_final = torch.zeros((batch_size,latent_dim,y_size)).to(audio.device)
        for i in range(num_chunks):
            x_chunk = chunks[i,:]
            # encode the chunk
            y_chunk = vae_model.encode_export(x_chunk)
            # figure out where to put the audio along the time domain
            if i == num_chunks-1:
                # final chunk always goes at the end
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = i * hop_size // samples_per_latent
                t_end = t_start + chunk_size // samples_per_latent
            #  remove the edges of the overlaps
            ol = overlap//samples_per_latent//2
            chunk_start = 0
            chunk_end = y_chunk.shape[2]
            if i > 0:
                # no overlap for the start of the first chunk
                t_start += ol
                chunk_start += ol
            if i < num_chunks-1:
                # no overlap for the end of the last chunk
                t_end -= ol
                chunk_end -= ol
            # paste the chunked audio into our y_final output audio
            y_final[:,:,t_start:t_end] = y_chunk[:,:,chunk_start:chunk_end]
        return y_final
        

def prepare_model(max_frames, device, model_name):
    # prepare cfm model
    dit_ckpt_path = os.path.join(model_path, "DiffRhythm", model_name)
    dit_config_path = f"{node_dir}/diffrhythm/config/diffrhythm-1b.json"

    with open(dit_config_path) as f:
        model_config = json.load(f)
    dit_model_cls = DiT
    cfm = CFM(
                transformer=dit_model_cls(**model_config["model"], max_frames=max_frames),
                num_channels=model_config["model"]['mel_dim'],
             )
    cfm = cfm.to(device)
    cfm = load_checkpoint(cfm, dit_ckpt_path, device=device, use_ema=False)

    # prepare tokenizer
    tokenizer = CNENTokenizer()

    # prepare muq model
    try:
        from easydict import EasyDict
        main_model_dir = f"{model_path}/DiffRhythm/MuQ-MuLan-large"
        local_audio_model_dir = f"{model_path}/DiffRhythm/MuQ-large-msd-iter"
        local_text_model_dir = f"{model_path}/DiffRhythm/xlm-roberta-base"

        config_path = f"{main_model_dir}/config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        config_dict['audio_model']['name'] = local_audio_model_dir
        config_dict['text_model']['name'] = local_text_model_dir
        config_obj = EasyDict(config_dict) 

        muq = MuQMuLan(config=config_obj, hf_hub_cache_dir=None)
        weights_path = f"{main_model_dir}/pytorch_model.bin"

        try:
            try:
                state_dict = torch.load(weights_path, map_location='cpu')  # defaults weights_only=True on torch>=2.6
            except Exception:
                state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
            # Adjust loading based on how weights are saved (e.g., remove prefixes if needed)
            muq.load_state_dict(state_dict, strict=False) # Use strict=False initially
        except FileNotFoundError:
            raise FileNotFoundError(f"Weights file not found at {weights_path}")

    except Exception as e:
        raise
    
    muq = muq.to(device).eval()

    # prepare vae
    vae_ckpt_path = f"{model_path}/DiffRhythm/vae_model.pt"
    vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to(device)

    # prepare eval model
    train_config = OmegaConf.load(f"{model_path}/DiffRhythm/eval-model/eval.yaml")
    checkpoint_path = f"{model_path}/DiffRhythm/eval-model/eval.safetensors"

    eval_model = instantiate(train_config.generator).to(device).eval()
    state_dict = load_file(checkpoint_path, device="cpu")
    eval_model.load_state_dict(state_dict)

    eval_muq = MuQ.from_pretrained(f"{model_path}/DiffRhythm/MuQ-large-msd-iter")
    eval_muq = eval_muq.to(device).eval()

    return cfm, tokenizer, muq, vae, eval_model, eval_muq


# for song edit, will be added in the future
def get_reference_latent(device, max_frames, edit, pred_segments, ref_song, vae_model):
    sampling_rate = 44100
    downsample_rate = 2048
    io_channels = 2
    if edit:
        input_audio, in_sr = torchaudio.load(ref_song)
        input_audio = prepare_audio(input_audio, in_sr=in_sr, target_sr=sampling_rate, target_length=None, target_channels=io_channels, device=device)
        input_audio = normalize_audio(input_audio, -6)
        
        with torch.no_grad():
            latent = encode_audio(input_audio, vae_model, chunked=True) # [b d t]
            mean, scale = latent.chunk(2, dim=1)
            prompt, _ = vae_sample(mean, scale)
            prompt = prompt.transpose(1, 2) # [b t d]
            prompt = prompt[:,:max_frames,:] if prompt.shape[1] >= max_frames else torch.nn.functional.pad(prompt, (0, 0, 0, max_frames - prompt.shape[1]), mode="constant", value=0)
        
        pred_segments = json.loads(pred_segments)
        # import pdb; pdb.set_trace()
        pred_frames = []
        for st, et in pred_segments:
            sf = 0 if st == -1 else int(st * sampling_rate / downsample_rate)
            # if st == -1:
            #     sf = 0
            # else:
            #     sf = int(st * sampling_rate / downsample_rate )
            
            ef = max_frames if et == -1 else int(et * sampling_rate / downsample_rate)
            # if et == -1:
            #     ef = max_frames
            # else:
            #     ef = int(et * sampling_rate / downsample_rate )
            pred_frames.append((sf, ef))
        # import pdb; pdb.set_trace()
        return prompt, pred_frames
    else:
        prompt = torch.zeros(1, max_frames, 64).to(device)
        pred_frames = [(0, max_frames)]
        return prompt, pred_frames


def get_negative_style_prompt(device):
    file_path = f"{node_dir}/diffrhythm/vocal.npy"
    vocal_stlye = np.load(file_path)

    vocal_stlye = torch.from_numpy(vocal_stlye).to(device)  # [1, 512]
    vocal_stlye = vocal_stlye.half()

    return vocal_stlye


@torch.no_grad()
def eval_song(eval_model, eval_muq, songs):
    
    resampled_songs = [torchaudio.functional.resample(song.mean(dim=0, keepdim=True), 44100, 24000) for song in songs]
    ssl_list = []
    for i in range(len(resampled_songs)):
        output = eval_muq(resampled_songs[i], output_hidden_states=True)
        muq_ssl = output["hidden_states"][6]
        ssl_list.append(muq_ssl.squeeze(0))

    ssl = torch.stack(ssl_list)
    scores_g = eval_model(ssl)
    score = torch.mean(scores_g, dim=1)
    idx = score.argmax(dim=0)
    
    return songs[idx]


@torch.no_grad()
def get_audio_style_prompt(model, wav_path):
    vocal_flag = False
    mulan = model
    audio, _ = librosa.load(wav_path, sr=24000)
    audio_len = librosa.get_duration(y=audio, sr=24000)
    
    if audio_len <= 1:
        vocal_flag = True
    
    if audio_len > 10:
        start_time = int(audio_len // 2 - 5)
        wav = audio[start_time*24000:(start_time+10)*24000]
    
    else:
        wav = audio
    wav = torch.tensor(wav).unsqueeze(0).to(model.device)
    
    with torch.no_grad():
        audio_emb = mulan(wavs = wav) # [1, 512]
        
    audio_emb = audio_emb.half()

    return audio_emb, vocal_flag


@torch.no_grad()
def get_text_style_prompt(model, text_prompt):
    mulan = model
    
    with torch.no_grad():
        text_emb = mulan(texts = text_prompt) # [1, 512]
    text_emb = text_emb.half()

    return text_emb


@torch.no_grad()
def get_style_prompt(model, wav_path=None, prompt=None):
    mulan = model

    if prompt is not None:
        return mulan(texts=prompt).half()

    ext = os.path.splitext(wav_path)[-1].lower()
    if ext == ".mp3":
        meta = MP3(wav_path)
        audio_len = meta.info.length
    elif ext in [".wav", ".flac"]:
        audio_len = librosa.get_duration(path=wav_path)
    else:
        raise ValueError("Unsupported file format: {}".format(ext))

    if audio_len < 10:
        print(
            f"Warning: The audio file {wav_path} is too short ({audio_len:.2f} seconds). Expected at least 10 seconds."
        )

    assert audio_len >= 10

    mid_time = audio_len // 2
    start_time = mid_time - 5
    wav, _ = librosa.load(wav_path, sr=24000, offset=start_time, duration=10)

    wav = torch.tensor(wav).unsqueeze(0).to(model.device)

    with torch.no_grad():
        audio_emb = mulan(wavs=wav)  # [1, 512]

    audio_emb = audio_emb
    audio_emb = audio_emb.half()

    return audio_emb


def parse_lyrics(lyrics: str):
    lyrics_with_time = []
    lyrics = lyrics.strip()
    for line in lyrics.split("\n"):
        try:
            time, lyric = line[1:9], line[10:]
            lyric = lyric.strip()
            mins, secs = time.split(":")
            secs = int(mins) * 60 + float(secs)
            lyrics_with_time.append((secs, lyric))
        except:
            continue
    return lyrics_with_time


class CNENTokenizer:
    def __init__(self):
        with open(f"{node_dir}/diffrhythm/g2p/g2p/vocab.json", "r", encoding='utf-8') as file:
            self.phone2id: dict = json.load(file)["vocab"]
        self.id2phone = {v: k for (k, v) in self.phone2id.items()}
        from diffrhythm.g2p.g2p_generation import chn_eng_g2p

        self.tokenizer = chn_eng_g2p

    def encode(self, text):
        phone, token = self.tokenizer(text)
        token = [x + 1 for x in token]
        return token

    def decode(self, token):
        return "|".join([self.id2phone[x - 1] for x in token])


def get_lrc_token(max_frames, text, tokenizer, device):

    lyrics_shift = 0
    sampling_rate = 44100
    downsample_rate = 2048
    max_secs = max_frames / (sampling_rate / downsample_rate)

    comma_token_id = 1
    period_token_id = 2

    lrc_with_time = parse_lyrics(text)

    modified_lrc_with_time = []
    for i in range(len(lrc_with_time)):
        time, line = lrc_with_time[i]
        line_token = tokenizer.encode(line)
        modified_lrc_with_time.append((time, line_token))
    lrc_with_time = modified_lrc_with_time

    lrc_with_time = [
        (time_start, line)
        for (time_start, line) in lrc_with_time
        if time_start < max_secs
    ]
    if max_frames == 2048:
        lrc_with_time = lrc_with_time[:-1] if len(lrc_with_time) >= 1 else lrc_with_time

    normalized_start_time = 0.0

    lrc = torch.zeros((max_frames,), dtype=torch.long)

    tokens_count = 0
    last_end_pos = 0
    for time_start, line in lrc_with_time:
        tokens = [
            token if token != period_token_id else comma_token_id for token in line
        ] + [period_token_id]
        tokens = torch.tensor(tokens, dtype=torch.long)
        num_tokens = tokens.shape[0]

        gt_frame_start = int(time_start * sampling_rate / downsample_rate)

        frame_shift = random.randint(int(-lyrics_shift), int(lyrics_shift))

        frame_start = max(gt_frame_start - frame_shift, last_end_pos)
        frame_len = min(num_tokens, max_frames - frame_start)

        lrc[frame_start : frame_start + frame_len] = tokens[:frame_len]

        tokens_count += num_tokens
        last_end_pos = frame_start + frame_len

    lrc_emb = lrc.unsqueeze(0).to(device)

    normalized_start_time = torch.tensor(normalized_start_time).unsqueeze(0).to(device)
    normalized_start_time = normalized_start_time.half()

    return lrc_emb, normalized_start_time


def load_checkpoint(model, ckpt_path, device, use_ema=True):
    model = model.half()

    ckpt_type = ckpt_path.split(".")[-1]
    try:
        from safetensors.torch import load_file
    except Exception as e:
        print(e)

    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except Exception:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return model.to(device)