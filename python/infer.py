import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'YuE', 'inference'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'YuE', 'inference', 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'YuE', 'inference', 'xcodec_mini_infer', 'descriptaudiocodec'))
from contextlib import contextmanager

import argparse
import random
import re
import time

import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
from omegaconf import OmegaConf
from einops import rearrange
from optimum.onnxruntime import ORTModelForCausalLM
import soundfile as sf
import onnxruntime as ort

# 注：sys.pathから読み込むため、VSCode等で解決出来ない
from mmtokenizer import _MMSentencePieceTokenizer
from codecmanipulator import CodecManipulator
from models.soundstream_hubert_new import SoundStream
from vocoder import build_codec_model, process_audio
from post_process_audio import replace_low_freq_with_energy_matched

from infer_stage1 import stage1
from infer_stage2 import stage2

print("torch version:", torch.__version__)
print("ONNX Runtime version:", ort.__version__)

parser = argparse.ArgumentParser()
# Model Configuration:
parser.add_argument("--stage1_model", type=str, default="siouni/YuE-s1-7B-anneal-en-cot-onnx-q4", help="The model checkpoint path or identifier for the Stage 1 model.")
parser.add_argument("--stage2_model", type=str, default="siouni/YuE-s2-1B-general-onnx-q4", help="The model checkpoint path or identifier for the Stage 2 model.")
parser.add_argument("--max_new_tokens", type=int, default=3000, help="The maximum number of new tokens to generate in one pass during text generation.")
parser.add_argument("--repetition_penalty", type=float, default=1.1, help="repetition_penalty ranges from 1.0 to 2.0 (or higher in some cases). It controls the diversity and coherence of the audio tokens generated. The higher the value, the greater the discouragement of repetition. Setting value to 1.0 means no penalty.")
parser.add_argument("--run_n_segments", type=int, default=2, help="The number of segments to process during the generation.")
parser.add_argument("--stage2_batch_size", type=int, default=4, help="The batch size used in Stage 2 inference.")
# Prompt
parser.add_argument("--genre_txt", type=str, required=True, help="The file path to a text file containing genre tags that describe the musical style or characteristics (e.g., instrumental, genre, mood, vocal timbre, vocal gender). This is used as part of the generation prompt.")
parser.add_argument("--lyrics_txt", type=str, required=True, help="The file path to a text file containing the lyrics for the music generation. These lyrics will be processed and split into structured segments to guide the generation process.")
parser.add_argument("--use_audio_prompt", action="store_true", help="If set, the model will use an audio file as a prompt during generation. The audio file should be specified using --audio_prompt_path.")
parser.add_argument("--audio_prompt_path", type=str, default="", help="The file path to an audio file to use as a reference prompt when --use_audio_prompt is enabled.")
parser.add_argument("--prompt_start_time", type=float, default=0.0, help="The start time in seconds to extract the audio prompt from the given audio file.")
parser.add_argument("--prompt_end_time", type=float, default=30.0, help="The end time in seconds to extract the audio prompt from the given audio file.")
parser.add_argument("--use_dual_tracks_prompt", action="store_true", help="If set, the model will use dual tracks as a prompt during generation. The vocal and instrumental files should be specified using --vocal_track_prompt_path and --instrumental_track_prompt_path.")
parser.add_argument("--vocal_track_prompt_path", type=str, default="", help="The file path to a vocal track file to use as a reference prompt when --use_dual_tracks_prompt is enabled.")
parser.add_argument("--instrumental_track_prompt_path", type=str, default="", help="The file path to an instrumental track file to use as a reference prompt when --use_dual_tracks_prompt is enabled.")
# Output 
parser.add_argument("--output_dir", type=str, default="../output", help="The directory where generated outputs will be saved.")
parser.add_argument("--keep_intermediate", action="store_true", help="If set, intermediate outputs will be saved during processing.")
parser.add_argument("--disable_offload_model", action="store_true", help="If set, the model will not be offloaded from the GPU to CPU after Stage 1 inference.")
parser.add_argument("--cuda_idx", type=int, default=0)
parser.add_argument("--seed", type=int, default=42, help="An integer value to reproduce generation.")
# Config for xcodec and upsampler
parser.add_argument('--basic_model_config', default='./YuE/inference/xcodec_mini_infer/final_ckpt/config.yaml', help='YAML files for xcodec configurations.')
parser.add_argument('--resume_path', default='./YuE/inference/xcodec_mini_infer/final_ckpt/ckpt_00360000.pth', help='Path to the xcodec checkpoint.')
parser.add_argument('--config_path', type=str, default='./YuE/inference/xcodec_mini_infer/decoders/config.yaml', help='Path to Vocos config file.')
parser.add_argument('--vocal_decoder_path', type=str, default='./YuE/inference/xcodec_mini_infer/decoders/decoder_131000.pth', help='Path to Vocos decoder weights.')
parser.add_argument('--inst_decoder_path', type=str, default='./YuE/inference/xcodec_mini_infer/decoders/decoder_151000.pth', help='Path to Vocos decoder weights.')
parser.add_argument('-r', '--rescale', action='store_true', help='Rescale output to avoid clipping.')


args = parser.parse_args()

if args.use_audio_prompt and not args.audio_prompt_path:
    raise FileNotFoundError("Please offer audio prompt filepath using '--audio_prompt_path', when you enable 'use_audio_prompt'!")
if args.use_dual_tracks_prompt and not args.vocal_track_prompt_path and not args.instrumental_track_prompt_path:
    raise FileNotFoundError("Please offer dual tracks prompt filepath using '--vocal_track_prompt_path' and '--inst_decoder_path', when you enable '--use_dual_tracks_prompt'!")

@contextmanager
def change_dir(new_dir):
    # 現在のディレクトリを記憶
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield  # with ブロック内の処理を実行
    finally:
        # with ブロック終了後、元のディレクトリに戻す
        os.chdir(prev_dir)

def seed_everything(seed=42): 
    print(f"seed: {seed}")
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_lyrics(lyrics):
    pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
    return structured_lyrics

def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio

def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
    if len(audio_prompt.shape) < 3:
        audio_prompt.unsqueeze_(0)
    with torch.no_grad():
        raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
    raw_codes = raw_codes.transpose(0, 1)
    raw_codes = raw_codes.cpu().numpy().astype(np.int16)
    return raw_codes

stage1_model = args.stage1_model
stage2_model = args.stage2_model
cuda_idx = args.cuda_idx
max_new_tokens = args.max_new_tokens
stage1_output_dir = os.path.join(args.output_dir, f"stage1")
stage2_output_dir = stage1_output_dir.replace('stage1', 'stage2')
# reconstruct tracks
recons_output_dir = os.path.join(args.output_dir, "recons")
recons_mix_dir = os.path.join(recons_output_dir, 'mix')

vocoder_output_dir = os.path.join(args.output_dir, 'vocoder')
vocoder_stems_dir = os.path.join(vocoder_output_dir, 'stems')
vocoder_mix_dir = os.path.join(vocoder_output_dir, 'mix')

stage2_batch_size = args.stage2_batch_size

seed = args.seed
genre_txt = args.genre_txt
lyrics_txt = args.lyrics_txt

# Tips:
# genre tags support instrumental，genre，mood，vocal timbr and vocal gender
# all kinds of tags are needed
with open(genre_txt) as f:
    genres = f.read().strip()
with open(lyrics_txt) as f:
    lyrics = split_lyrics(f.read())
    
# intruction
full_lyrics = "\n".join(lyrics)
prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
prompt_texts += lyrics

basic_model_config = args.basic_model_config
resume_path = args.resume_path

run_n_segments = min(args.run_n_segments+1, len(lyrics))

use_dual_tracks_prompt = args.use_dual_tracks_prompt
use_audio_prompt = args.use_audio_prompt
vocal_track_prompt_path = args.vocal_track_prompt_path
instrumental_track_prompt_path = args.instrumental_track_prompt_path
prompt_end_time = args.prompt_end_time
prompt_start_time = args.prompt_start_time
audio_prompt_path = args.audio_prompt_path

repetition_penalty = args.repetition_penalty

disable_offload_model = args.disable_offload_model

config_path = args.config_path
vocal_decoder_path = args.vocal_decoder_path
inst_decoder_path = args.inst_decoder_path

rescale = args.rescale

seed_everything(seed)

device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")

mmtokenizer = _MMSentencePieceTokenizer("./YuE/inference/mm_tokenizer_v0.2_hf/tokenizer.model")

model_config = OmegaConf.load(basic_model_config)
parameter_dict = torch.load(resume_path, map_location='cpu', weights_only=False)

with change_dir("./YuE/inference/"):
    codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)

codec_model.load_state_dict(parameter_dict['codec_model'])
codec_model.to(device)
codec_model.eval()

codectool = CodecManipulator("xcodec", 0, 1)
codectool_stage2 = CodecManipulator("xcodec", 0, 8)

# VRAM 使用量測定のための初期化
torch.cuda.reset_peak_memory_stats(device)
torch.cuda.synchronize()  # 非同期処理があれば同期しておく

t_start = time.time()
print("Initialize audio start")
if use_dual_tracks_prompt or use_audio_prompt:
    if use_dual_tracks_prompt:
        print("--use_dual_tracks_prompt is enabled")
        # audio loading...
        vocals_ids = load_audio_mono(vocal_track_prompt_path)
        instrumental_ids = load_audio_mono(instrumental_track_prompt_path)

        # audio encoding...
        vocals_ids = encode_audio(codec_model, vocals_ids, device, target_bw=0.5)
        instrumental_ids = encode_audio(codec_model, instrumental_ids, device, target_bw=0.5)

        # Format audio prompt
        # npy -> ids
        vocals_ids = codectool.npy2ids(vocals_ids[0])
        instrumental_ids = codectool.npy2ids(instrumental_ids[0])

        # to audio prompt codec
        ids_segment_interleaved = rearrange([np.array(vocals_ids), np.array(instrumental_ids)], 'b n -> (n b)')
        audio_prompt_codec = ids_segment_interleaved[int(prompt_start_time*50*2): int(prompt_end_time*50*2)]
        audio_prompt_codec = audio_prompt_codec.tolist()

    elif use_audio_prompt:
        print("--use_audio_prompt is enabled")
        # audio loading...
        audio_prompt = load_audio_mono(audio_prompt_path)

        # audio encoding...
        raw_codes = encode_audio(codec_model, audio_prompt, device, target_bw=0.5)

        # Format audio prompt
        # npy -> ids
        code_ids = codectool.npy2ids(raw_codes[0])

        # to audio prompt codec
        audio_prompt_codec = code_ids[int(prompt_start_time *50): int(prompt_end_time *50)] # 50 is tps of xcodec

    audio_prompt_codec_ids = [mmtokenizer.soa] + codectool.sep_ids + audio_prompt_codec + [mmtokenizer.eoa]

    sentence_ids = mmtokenizer.tokenize("[start_of_reference]") +  audio_prompt_codec_ids + mmtokenizer.tokenize("[end_of_reference]")
    head_id = mmtokenizer.tokenize(prompt_texts[0]) + sentence_ids
else:
    print("--use_dual_tracks_prompt and --use_audio_prompt is disabled")
    head_id = mmtokenizer.tokenize(prompt_texts[0])

process_time = time.time() - t_start
print(f"Initialize audio done {process_time:.2f}sec")
range_begin = 1 if use_audio_prompt or use_dual_tracks_prompt else 0

t_start = time.time()
print(f"stage 1 model loading... {stage1_model}")
model_stage1 = ORTModelForCausalLM.from_pretrained(
    stage1_model,
    provider="CUDAExecutionProvider",
    provider_options={"device_id": torch.cuda.current_device()},
)
process_time = time.time() - t_start
print(f"stage 1 model loading... done {(process_time / 60):.2f}min")

t_start = time.time()
print("Stage 1 inference...")
stage1_output_set = stage1(
    (
        stage1_output_dir,
        prompt_texts,
        repetition_penalty,
        run_n_segments,
        head_id,
        max_new_tokens,
        range_begin,
        genres,
    ),
    np,
    torch,
    device,
    mmtokenizer,
    codectool,
    model_stage1,
)
process_time = time.time() - t_start
print(f"Stage 1 done {(process_time / 60):.2f}min")
print(stage1_output_set)

# offload model
if not disable_offload_model:
    # model.cpu()
    del model_stage1
    torch.cuda.empty_cache()

# stage1_output_set = ['../output\\stage1\\inspiring-female-uplifting-pop-airy-vocal-electronic-bright-vocal-vocal_tp0@93_T1@0_rp1@1_maxtk3000_7344b3ed-11ac-4cb0-93a4-2df54014681e_vtrack.npy', '../output\\stage1\\inspiring-female-uplifting-pop-airy-vocal-electronic-bright-vocal-vocal_tp0@93_T1@0_rp1@1_maxtk3000_7344b3ed-11ac-4cb0-93a4-2df54014681e_itrack.npy']

t_start = time.time()
print(f"stage 2 model loading... {stage2_model}")
model_stage2 = ORTModelForCausalLM.from_pretrained(
    stage2_model,
    provider="CUDAExecutionProvider",
    provider_options={"device_id": torch.cuda.current_device()},
)
process_time = time.time() - t_start
print(f"stage 2 model loading... done {(process_time / 60):.2f}min")

t_start = time.time()
print("Stage 2 inference...")
stage2_result = stage2((
    stage2_output_dir,
    stage2_batch_size,
), stage1_output_set, np, torch, device, mmtokenizer, codectool, codectool_stage2, model_stage2)
process_time = time.time() - t_start
print(f"Stage 2 done {(process_time / 60):.2f}min")
print(stage2_result)

# offload model
if not disable_offload_model:
    # model.cpu()
    del model_stage2
    torch.cuda.empty_cache()

# stage2_result = ['../output\\stage2\\inspiring-female-uplifting-pop-airy-vocal-electronic-bright-vocal-vocal_tp0@93_T1@0_rp1@1_maxtk3000_7344b3ed-11ac-4cb0-93a4-2df54014681e_vtrack.npy', '../output\\stage2\\inspiring-female-uplifting-pop-airy-vocal-electronic-bright-vocal-vocal_tp0@93_T1@0_rp1@1_maxtk3000_7344b3ed-11ac-4cb0-93a4-2df54014681e_itrack.npy']

# convert audio tokens to audio
def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

os.makedirs(recons_mix_dir, exist_ok=True)
tracks = []
for npy in stage2_result:
    codec_result = np.load(npy)
    decodec_rlt=[]
    with torch.no_grad():
        decoded_waveform = codec_model.decode(torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(device))
    decoded_waveform = decoded_waveform.cpu().squeeze(0)
    decodec_rlt.append(torch.as_tensor(decoded_waveform))
    decodec_rlt = torch.cat(decodec_rlt, dim=-1)
    save_path = os.path.join(recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3")
    tracks.append(save_path)
    save_audio(decodec_rlt, save_path, 16000)
# mix tracks
for inst_path in tracks:
    try:
        if (inst_path.endswith('.wav') or inst_path.endswith('.mp3')) \
            and '_itrack' in inst_path:
            # find pair
            vocal_path = inst_path.replace('_itrack', '_vtrack')
            if not os.path.exists(vocal_path):
                continue
            # mix
            recons_mix = os.path.join(recons_mix_dir, os.path.basename(inst_path).replace('_itrack', '_mixed'))
            vocal_stem, sr = sf.read(inst_path)
            instrumental_stem, _ = sf.read(vocal_path)
            mix_stem = (vocal_stem + instrumental_stem) / 1
            sf.write(recons_mix, mix_stem, sr)
    except Exception as e:
        print(e)

# vocoder to upsample audios
vocal_decoder, inst_decoder = build_codec_model(config_path, vocal_decoder_path, inst_decoder_path)

os.makedirs(vocoder_mix_dir, exist_ok=True)
os.makedirs(vocoder_stems_dir, exist_ok=True)
for npy in stage2_result:
    if '_itrack' in npy:
        # Process instrumental
        instrumental_output = process_audio(
            npy,
            os.path.join(vocoder_stems_dir, 'itrack.mp3'),
            rescale,
            args,
            inst_decoder,
            codec_model
        )
    else:
        # Process vocal
        vocal_output = process_audio(
            npy,
            os.path.join(vocoder_stems_dir, 'vtrack.mp3'),
            rescale,
            args,
            vocal_decoder,
            codec_model
        )
# mix tracks
try:
    mix_output = instrumental_output + vocal_output
    vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix))
    save_audio(mix_output, vocoder_mix, 44100, rescale)
    print(f"Created mix: {vocoder_mix}")
except RuntimeError as e:
    print(e)
    print(f"mix {vocoder_mix} failed! inst: {instrumental_output.shape}, vocal: {vocal_output.shape}")

# Post process
replace_low_freq_with_energy_matched(
    a_file=recons_mix,     # 16kHz
    b_file=vocoder_mix,     # 48kHz
    c_file=os.path.join(args.output_dir, os.path.basename(recons_mix)),
    cutoff_freq=5500.0
)