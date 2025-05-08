import vllm
from vllm import LLM, SamplingParams
import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B")
parser.add_argument("--samples", type=int, default=1, help="Number of samples")
parser.add_argument("--temp", type=float, default=1., help="Sampling temperature")
parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of new tokens to generate")

# Example usage:
# First, you have to get access to https://huggingface.co/meta-llama/Llama-3.2-3B 
"""
python run_td.py --model_name "meta-llama/Llama-3.2-3B" --samples 50 --temp 1.
"""

if __name__ == "__main__":
    args = parser.parse_args()
    samples = args.samples
    model_name = args.model_name
    temp = args.temp
    max_new_tokens = args.max_new_tokens


    lm = LLM(
        model=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        distributed_executor_backend="mp",
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.95,
        enforce_eager=False,
        disable_custom_all_reduce=True,
        trust_remote_code=True,
        enable_prefix_caching=True,
        max_model_len=8192
    )
    tokenizer = lm.get_tokenizer()

    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
    ]

    sampling_params = SamplingParams(
        n=samples,
        top_p=1.0,
        temperature=temp,
        max_tokens=max_new_tokens
    )
    responses = lm.generate(prompts, sampling_params, use_tqdm=False)

    print(responses)
    # responses is a list of RequestOutput objects
    # Each RequestOutput object contains the following attributes:
    # - request_id: ID of the request
    # - prompt: the original prompt
    # - prompt_token_ids: token IDs of the prompt
    # - encoder_prompt: encoder prompt (if any)
    # - encoder_prompt_token_ids: token IDs of the encoder prompt (if any)
    # - prompt_logprobs: log probabilities of the prompt (if any)
    # - outputs: a list of CompletionOutput objects
    # - finished: whether the request is finished
    # - metrics: RequestMetrics object containing various timing metrics
    # - lora_request: LoraRequest object (if any)
    # - num_cached_tokens: number of cached tokens
    # - multi_modal_placeholders: placeholders for multi-modal inputs (if any)
    # Each CompletionOutput object contains the following attributes:
    # - index: index of the output
    # - text: the generated text
    # - token_ids: token IDs of the generated text
    # - cumulative_logprob: cumulative log probability of the generated text (if any)
    # - logprobs: log probabilities of the generated text (if any)

    for response in responses:
        # Each response corresponds to a prompt input
        for output in response.outputs:
            # Each output corresponds to a generated text, number of outputs = number of args.sample
            print(f"Generated text: {output.text}")
            print(f"Token IDs: {output.token_ids}")
            print(f"Cumulative log probability: {output.cumulative_logprob}")
            print(f"Log probabilities: {output.logprobs}")