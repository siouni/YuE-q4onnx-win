import os
import time
from tqdm import tqdm

from transformers import LogitsProcessorList
from collections import Counter
import copy

from BlockTokenRangeProcessor import BlockTokenRangeProcessor

TOKEN_PER_SEC = 50
SEC_PER_SEGMENT = 4 # 6が初期値、下げると品質が下がり、処理速度が上がる
FRAME_SIZE = TOKEN_PER_SEC * SEC_PER_SEGMENT

def stage2(params, stage1_output_set, np, torch, device, mmtokenizer, codectool, codectool_stage2, model_stage2):
    (
        stage2_output_dir,
        stage2_batch_size,
    ) = params
    print(f"stage2_output_dir: {stage2_output_dir}")
    os.makedirs(stage2_output_dir, exist_ok=True)

    def stage2_generate(model, prompt, batch_size=16):
        print("stage2_generate start")
        codec_ids = codectool.unflatten(prompt, n_quantizer=1)
        codec_ids = codectool.offset_tok_ids(
                        codec_ids, 
                        global_offset=codectool.global_offset, 
                        codebook_size=codectool.codebook_size, 
                        num_codebooks=codectool.num_codebooks, 
                    ).astype(np.int32)
        
        # Prepare prompt_ids based on batch size or single input
        print("Prepare prompt_ids based on batch size or single input")
        if batch_size > 1:
            print(f"batch size: {batch_size}")
            codec_list = []
            for i in range(batch_size):
                # t_start = time.time()
                # print(f"batch {i+1}/{batch_size} start")
                idx_begin = i * FRAME_SIZE
                idx_end = (i + 1) * FRAME_SIZE
                codec_list.append(codec_ids[:, idx_begin:idx_end])
                # process_time = time.time() - t_start
                # print(f"batch {i+1}/{batch_size} done {process_time:.2f}sec")

            codec_ids = np.concatenate(codec_list, axis=0)
            prompt_ids = np.concatenate(
                [
                    np.tile([mmtokenizer.soa, mmtokenizer.stage_1], (batch_size, 1)),
                    codec_ids,
                    np.tile([mmtokenizer.stage_2], (batch_size, 1)),
                ],
                axis=1
            )
        else:
            print("single input")
            prompt_ids = np.concatenate([
                np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
                codec_ids.flatten(),  # Flatten the 2D array to 1D
                np.array([mmtokenizer.stage_2])
            ]).astype(np.int32)
            prompt_ids = prompt_ids[np.newaxis, ...]

        codec_ids = torch.as_tensor(codec_ids).to(device)
        prompt_ids = torch.as_tensor(prompt_ids).to(device)
        len_prompt = prompt_ids.shape[-1]
        
        block_list = LogitsProcessorList([BlockTokenRangeProcessor(0, 46358), BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size)])

        # Teacher forcing generate loop
        print("Teacher forcing generate loop")
        pbar = tqdm(total=codec_ids.shape[1], desc="stage 2 generate frames")
        for frames_idx in range(codec_ids.shape[1]):
            t_start = time.time()
            # print(f"stage 2 generate frames {frames_idx+1}/{codec_ids.shape[1]} start")
            cb0 = codec_ids[:, frames_idx:frames_idx+1]
            prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
            input_ids = prompt_ids

            with torch.no_grad():
                stage2_output = model.generate(input_ids=input_ids, 
                    min_new_tokens=7,
                    max_new_tokens=7,
                    eos_token_id=mmtokenizer.eoa,
                    pad_token_id=mmtokenizer.eoa,
                    logits_processor=block_list,
                )
            
            assert stage2_output.shape[1] - prompt_ids.shape[1] == 7, f"output new tokens={stage2_output.shape[1]-prompt_ids.shape[1]}"
            prompt_ids = stage2_output

            process_time = time.time() - t_start
            # print(f"stage 2 generate frames {frames_idx+1}/{codec_ids.shape[1]} done {process_time:.2f}sec")
            pbar.set_postfix({"proc time": process_time})
            pbar.update(1)
        pbar.close()

        # Return output based on batch size
        print("Return output based on batch size")
        if batch_size > 1:
            print("batch size")
            output = prompt_ids.cpu().numpy()[:, len_prompt:]
            output_list = [output[i] for i in range(batch_size)]
            output = np.concatenate(output_list, axis=0)
        else:
            print("single input")
            output = prompt_ids[0].cpu().numpy()[len_prompt:]

        return output
    
    def stage2_inference(model, stage1_output_set, stage2_output_dir, batch_size=4):
        stage2_result = []
        base_segment_num = len(stage1_output_set)
        for i in range(base_segment_num):
            print(f"stage 2 {i+1}/{base_segment_num}")
            output_filename = os.path.join(stage2_output_dir, os.path.basename(stage1_output_set[i]))
            print(f"output_filename: {output_filename}")
            
            # if os.path.exists(output_filename):
            #     print(f'{output_filename} stage2 has done.')
            #     continue
            
            # Load the prompt
            print("Load the prompt")
            prompt = np.load(stage1_output_set[i]).astype(np.int32)
            
            # Only accept 6s segments
            print(f"Only accept {SEC_PER_SEGMENT}s segments")
            output_duration = prompt.shape[-1] // TOKEN_PER_SEC // SEC_PER_SEGMENT * SEC_PER_SEGMENT
            num_batch = output_duration // SEC_PER_SEGMENT

            print(f"num_batch: {num_batch} / batch_size: {batch_size}")
            
            if num_batch <= batch_size:
                # If num_batch is less than or equal to batch_size, we can infer the entire prompt at once
                print("If num_batch is less than or equal to batch_size, we can infer the entire prompt at once")
                output = stage2_generate(model, prompt[:, :output_duration*50], batch_size=num_batch)
            else:
                # If num_batch is greater than batch_size, process in chunks of batch_size
                print("If num_batch is greater than batch_size, process in chunks of batch_size")
                segments = []
                num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)
                print(f"num_segments: {num_segments}")

                for seg in range(num_segments):
                    t_start = time.time()
                    print(f"stage 2 segment: {i+1}/{base_segment_num} - chunks: {seg+1}/{num_segments} generating...")
                    start_idx = seg * batch_size * FRAME_SIZE
                    # Ensure the end_idx does not exceed the available length
                    end_idx = min((seg + 1) * batch_size * FRAME_SIZE, output_duration*50)  # Adjust the last segment
                    current_batch_size = batch_size if seg != num_segments-1 or num_batch % batch_size == 0 else num_batch % batch_size
                    segment = stage2_generate(
                        model,
                        prompt[:, start_idx:end_idx],
                        batch_size=current_batch_size
                    )
                    segments.append(segment)
                    process_time = time.time() - t_start
                    print(f"stage 2 segment: {i+1}/{base_segment_num} - chunks: {seg+1}/{num_segments} done {process_time:.2f}sec")

                # Concatenate all the segments
                print("Concatenate all the segments")
                output = np.concatenate(segments, axis=0)
            
            # Process the ending part of the prompt
            print("Process the ending part of the prompt")
            if output_duration*50 != prompt.shape[-1]:
                ending = stage2_generate(model, prompt[:, output_duration*50:], batch_size=1)
                output = np.concatenate([output, ending], axis=0)
            output = codectool_stage2.ids2npy(output)

            # Fix invalid codes (a dirty solution, which may harm the quality of audio)
            # We are trying to find better one
            print("Fix invalid codes (a dirty solution, which may harm the quality of audio)")
            print("We are trying to find better one")
            fixed_output = copy.deepcopy(output)
            for i, line in enumerate(output):
                for j, element in enumerate(line):
                    if element < 0 or element > 1023:
                        counter = Counter(line)
                        most_frequant = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                        fixed_output[i, j] = most_frequant
            # save output
            print(f"save output: {output_filename}")
            np.save(output_filename, fixed_output)
            stage2_result.append(output_filename)
        return stage2_result
    
    return stage2_inference(model_stage2, stage1_output_set, stage2_output_dir, batch_size=stage2_batch_size)