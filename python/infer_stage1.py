import os
import uuid
import time

from einops import rearrange
from transformers import LogitsProcessorList

from BlockTokenRangeProcessor import BlockTokenRangeProcessor

def stage1(params, np, torch, device, mmtokenizer, codectool, stage1_model):
    (
        stage1_output_dir,
        prompt_texts,
        repetition_penalty,
        run_n_segments,
        head_id,
        max_new_tokens,
        range_begin,
        genres,
    ) = params
    print(f"stage1_output_dir: {stage1_output_dir}")
    os.makedirs(stage1_output_dir, exist_ok=True)
    # Call the function and print the result
    stage1_output_set = []
    
    random_id = uuid.uuid4()
    output_seq = None
    # Here is suggested decoding config
    top_p = 0.93
    temperature = 1.0
    print(f"top_p: {top_p} / temperature: {temperature} / repetition_penalty: {repetition_penalty}")
    # special tokens
    start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
    end_of_segment = mmtokenizer.tokenize('[end_of_segment]')
    # Format text prompt
    
    print(f"run_n_segments: {run_n_segments}")
    for i, paragraph in enumerate(prompt_texts[:run_n_segments]):
        if i==0:
            print(f"stage1 {i+1}/{run_n_segments} skip...")
            continue
        section_text = paragraph.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
        # print(f"section_text: {section_text}")
        guidance_scale = 1.5 if i <=1 else 1.2
        if i==1:
            prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids
        else:
            prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids

        prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device) 
        input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids
        # Use window slicing in case output sequence exceeds the context of model
        max_context = 16384-max_new_tokens-1
        if input_ids.shape[-1] > max_context:
            print(f'Section {i}: output length {input_ids.shape[-1]} exceeding context length {max_context}, now using the last {max_context} tokens.')
            input_ids = input_ids[:, -(max_context):]
        t_start = time.time()
        print(f"stage1 {i+1}/{run_n_segments} generating...")
        with torch.no_grad():
            output_seq = stage1_model.generate(
                input_ids=input_ids, 
                max_new_tokens=max_new_tokens, 
                min_new_tokens=100, 
                do_sample=True, 
                top_p=top_p,
                temperature=temperature, 
                repetition_penalty=repetition_penalty, 
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=LogitsProcessorList([BlockTokenRangeProcessor(0, 32002), BlockTokenRangeProcessor(32016, 32016)]),
                guidance_scale=guidance_scale,
            )
            if output_seq[0][-1].item() != mmtokenizer.eoa:
                tensor_eoa = torch.as_tensor([[mmtokenizer.eoa]]).to(stage1_model.device)
                output_seq = torch.cat((output_seq, tensor_eoa), dim=1)
        if i > 1:
            raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1]:]], dim=1)
        else:
            raw_output = output_seq
        
        process_time = time.time() - t_start
        print(f"stage1 {i+1}/{run_n_segments} generating... done {process_time:.2f}sec")

    # save raw output and check sanity
    print("save raw output and check sanity")
    ids = raw_output[0].cpu().numpy()
    soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
    eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
    if len(soa_idx)!=len(eoa_idx):
        raise ValueError(f'invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}')

    vocals = []
    instrumentals = []
    
    for i in range(range_begin, len(soa_idx)):
        codec_ids = ids[soa_idx[i]+1:eoa_idx[i]]
        if codec_ids[0] == 32016:
            codec_ids = codec_ids[1:]
        codec_ids = codec_ids[:2 * (codec_ids.shape[0] // 2)]
        vocals_ids = codectool.ids2npy(rearrange(codec_ids,"(n b) -> b n", b=2)[0])
        vocals.append(vocals_ids)
        instrumentals_ids = codectool.ids2npy(rearrange(codec_ids,"(n b) -> b n", b=2)[1])
        instrumentals.append(instrumentals_ids)
    
    vocals = np.concatenate(vocals, axis=1)
    instrumentals = np.concatenate(instrumentals, axis=1)

    vocal_save_path = os.path.join(stage1_output_dir, f"{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}_vtrack".replace('.', '@')+'.npy')

    inst_save_path = os.path.join(stage1_output_dir, f"{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_{random_id}_itrack".replace('.', '@')+'.npy')

    print(f"save vocals npy {vocal_save_path}")
    np.save(vocal_save_path, vocals)
    print(f"save instrumentals npy {inst_save_path}")
    np.save(inst_save_path, instrumentals)

    stage1_output_set.append(vocal_save_path)
    stage1_output_set.append(inst_save_path)

    return stage1_output_set