# YuE-q4onnx-win
YuEを4bit量子化onnxモデルでwindows環境で動作するようにしたものです。  
4bit量子化と、処理速度向上のために品質が低下しています。  
量子化の影響はさほど感じませんでしたが、処理速度向上のための調整は品質に大きく影響しました。  
ひとまずは速度優先にしていますが、今後調整できるようにする予定です。  
（現在でも、infer_stage2.pyのSEC_PER_SEGMENTを6にすることで品質優先に出来ます。）

副産物的なことですが、本家YuEよりも進捗の表示が細かくなっています。  

## インストール方法
必須環境：CUDA対応のGPU  
推奨環境：Windows11、RAM 32GB以上、RTX4070 12GB程度以上  
Python 3.10、git、CUDA 12.4がインストールされている前提となります。  
Python -> [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)  
git -> [https://git-scm.com/](https://git-scm.com/)  
CUDA -> [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)
  
```CLI
# 任意のフォルダ（できれば低階層）で実行
git lfs install
git clone https://github.com/siouni/YuE-q4onnx-win.git

cd YuE-q4onnx-win/python
git clone https://github.com/multimodal-art-projection/YuE.git

cd YuE/inference/
git clone https://huggingface.co/m-a-p/xcodec_mini_infer

cd ../../
python -m venv venv
venv\Scripts\activate
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

cd ./Yue
pip install -r requirements.txt

cd ../
pip install optimum[onnxruntime]
pip install onnxruntime-gpu

cd ./models/
git clone https://huggingface.co/siouni/YuE-s1-7B-anneal-en-cot-onnx-q4
git clone https://huggingface.co/siouni/YuE-s1-7B-anneal-jp-kr-cot-onnx-q4
git clone https://huggingface.co/siouni/YuE-s2-1B-general-onnx-q4
```
## 実行方法
```CLI
cd ./python
python infer.py --cuda_idx 0 --stage1_model ../models/YuE-s1-7B-anneal-en-cot-onnx-q4 --stage2_model ../models/YuE-s2-1B-general-onnx-q4 --genre_txt ./YuE/prompt_egs/genre.txt --lyrics_txt ./YuE/prompt_egs/lyrics.txt --run_n_segments 2 --stage2_batch_size 4 --output_dir ../output --max_new_tokens 3000 --repetition_penalty 1.1
```