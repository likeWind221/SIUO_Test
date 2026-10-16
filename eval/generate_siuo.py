#!/usr/bin/env python3
"""Generate SIUO responses with a local Qwen2.5-VL checkpoint."""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-responses", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json_atomic(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def prepare_results(dataset, output_path):
    if not output_path.exists():
        results = copy.deepcopy(dataset)
        for item in results:
            item["responses"] = []
        return results

    results = load_json(output_path)
    expected_ids = [item["question_id"] for item in dataset]
    result_ids = [item.get("question_id") for item in results]
    if result_ids != expected_ids:
        raise ValueError("Existing output does not match the input dataset order.")
    for item in results:
        item.setdefault("responses", [])
    return results


def generate_one(model, processor, image_path, question, args, seed):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.resolve().as_uri()},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda:0")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )

    completion_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation script.")
    if args.num_responses < 1:
        raise ValueError("--num-responses must be at least 1.")

    dataset = load_json(args.dataset)
    end = len(dataset) - 1 if args.end is None else min(args.end, len(dataset) - 1)
    if args.start < 0 or args.start > end:
        raise ValueError(f"Invalid range: start={args.start}, end={end}")

    missing_images = [
        str(args.image_root / item["image"])
        for item in dataset[args.start : end + 1]
        if not (args.image_root / item["image"]).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"Missing {len(missing_images)} SIUO images; first missing: {missing_images[0]}"
        )

    results = prepare_results(dataset, args.output)
    processor = AutoProcessor.from_pretrained(
        args.model,
        max_pixels=args.max_pixels,
        local_files_only=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("cuda:0")
    model.eval()

    print(f"Loaded model: {args.model}", flush=True)
    print(f"Generating items {args.start}..{end} into {args.output}", flush=True)
    for index in range(args.start, end + 1):
        item = results[index]
        responses = item["responses"]
        if len(responses) > args.num_responses:
            item["responses"] = responses[: args.num_responses]
            responses = item["responses"]

        while len(responses) < args.num_responses:
            response_index = len(responses)
            sample_seed = args.seed + index * args.num_responses + response_index
            response = generate_one(
                model=model,
                processor=processor,
                image_path=args.image_root / item["image"],
                question=item["question"],
                args=args,
                seed=sample_seed,
            )
            responses.append(response)
            save_json_atomic(results, args.output)
            print(
                f"[{index + 1}/{len(dataset)}] {item['question_id']} "
                f"response {len(responses)}/{args.num_responses}",
                flush=True,
            )

    save_json_atomic(results, args.output)
    print("SIUO generation complete.", flush=True)


if __name__ == "__main__":
    main()
