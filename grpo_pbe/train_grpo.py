"""GRPO training script using Unsloth + TRL."""
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"  # memory-efficient GRPO with vLLM

import torch
import wandb
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from grpo_pbe.data_generator import load_dataset
from grpo_pbe.reward import compute_reward

# --- Config ---
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 16
OUTPUT_DIR = "checkpoints/grpo-pbe-1.5b"
WANDB_PROJECT = "grpo-pbe"

SYSTEM_PROMPT = (
    "You solve data transformation tasks. Given input/output examples, "
    "write a Python expression using `x` as the input variable. "
    "Respond with reasoning in <think>...</think> tags, "
    "then the expression in <code>...</code> tags."
)


def build_reward_fn(train_data: list[dict]):
    """Build a reward function that looks up test cases by prompt content."""
    # Map the user-message content to test cases
    prompt_to_tests = {}
    for ex in train_data:
        prompt_to_tests[ex["prompt"]] = ex["tests"]

    def reward_fn(prompts, completions, **kwargs) -> list[float]:
        rewards = []
        for prompt_msgs, completion_msgs in zip(prompts, completions):
            # Extract the user message content (our prompt text)
            user_content = prompt_msgs[-1]["content"] if isinstance(prompt_msgs, list) else prompt_msgs
            # Extract the completion text
            response = completion_msgs[0]["content"] if isinstance(completion_msgs, list) else completion_msgs
            tests = prompt_to_tests.get(user_content, [])
            r = compute_reward(response, tests)
            rewards.append(r)
        return rewards

    return reward_fn


def make_conversation(example: dict) -> dict:
    """Convert a raw prompt string to chat message format."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ],
    }


def main():
    # Load data
    train_data = load_dataset("data/train.jsonl")
    print(f"Loaded {len(train_data)} training examples")

    # Build HF dataset with prompts in chat format
    hf_dataset = Dataset.from_dict({"prompt": [ex["prompt"] for ex in train_data]})
    hf_dataset = hf_dataset.map(make_conversation)

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,  # vLLM for faster generation
        gpu_memory_utilization=0.6,  # conservative for 8GB 4060
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )

    # Training config
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_generations=4,  # group size G
        max_prompt_length=512,
        max_completion_length=512,
        learning_rate=5e-6,
        kl_coef=0.04,
        optim="adamw_8bit",
        logging_steps=10,
        save_steps=200,
        report_to="wandb",
        bf16=True,
        seed=42,
        mask_truncated_completions=True,
    )

    # Init W&B
    wandb.init(project=WANDB_PROJECT, name="grpo-run", config={
        "model": MODEL_NAME,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "group_size": training_args.num_generations,
        "kl_coef": training_args.kl_coef,
        "lr": training_args.learning_rate,
        "max_completion_length": training_args.max_completion_length,
    })

    # Build reward function
    reward_fn = build_reward_fn(train_data)

    # Train
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=hf_dataset,
    )
    trainer.train()

    # Save
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    wandb.finish()


if __name__ == "__main__":
    main()
