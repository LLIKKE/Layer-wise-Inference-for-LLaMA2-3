import torch
import data_utils
import transformers
import eval_utils
from transformers import LlamaTokenizerFast
from process_args import process_args
from modeling_llama import LlamaForCausalLM
def main():
    args = process_args()
        
    transformers.set_seed(args.seed)
    model = LlamaForCausalLM.from_pretrained(args.input_model, torch_dtype='auto',low_cpu_mem_usage=True,device_map="cpu")
    model.eval()

    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=args.input_model,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=None,
    )


    testloader = data_utils.get_wikitext2(
        seed=args.seed,
        seqlen=args.model_max_length,
        tokenizer=tokenizer,
        eval_mode=True,
    )

    dataset_ppl = eval_utils.evaluator(model, testloader, args.device, args)
    print(f"{args.model_name} ppl : {dataset_ppl}")


if __name__ == '__main__':
    main()
