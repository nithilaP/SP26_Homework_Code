# question 4b, 4c
import random
import argparse

import dataset
import models
import trainer
import utils

import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer # need to pip install transformers
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

def main():

    # get the dev set
    argp = argparse.ArgumentParser()
    argp.add_argument('--dev_set_path', type=str, default="birth_dev.tsv")
    args = argp.parse_args()

    # transformer set up from lecture
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Hw CHECK IN COULD CHANGE
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()

    print(next(model.parameters()).device)

    # HW CHECK IN   
    # parse the birth dev tsv. 
    with open(args.dev_set_path, 'r', encoding='utf-8') as data_file:
        birth_data = []
        for line in data_file:
            if line.strip(): # consider lines with just \n
                inp, oup = line.split('\t')      
                birth_data.append((inp, oup))
    
    # LOGGING:
    pred_lines_v1 = []

    # Variant 1:
    variant1_correct = 0
    for inp, oup in tqdm(birth_data, desc="Variant 1"):

        # FIX: REMOVE WHITESPACES
        inp = inp.strip()
        oup = oup.strip()

        # need to use chat template for Qwen: https://discuss.huggingface.co/t/fine-tune-a-minimal-llm-model-with-rtx-2050-gpu/171003
        messages = [
            {
                "role": "user",
                "content": inp
            }
        ]
        templ_inp = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input1 = tokenizer(templ_inp, return_tensors="pt").to(model.device)

        # determine number of tokens in input 
        tokens_input = input1["input_ids"] #https://discuss.huggingface.co/t/fine-tune-a-minimal-llm-model-with-rtx-2050-gpu/171003
        num_tokens_input = len(tokens_input[0])
        
        with torch.no_grad():
            output = model.generate(**input1, max_new_tokens=10, do_sample=False, use_cache=True) # do not want random

        output_text = tokenizer.decode(output[0, num_tokens_input:], skip_special_tokens=True)
        output_text = output_text.strip() # FIX: strip the output so we can match with oup

        if (oup == output_text):
            variant1_correct += 1
        
        # LOGGING: 
        pred_lines_v1.append(f"{inp}\t{oup}\t{output_text}")

    variant1_accuracy = (variant1_correct / len(birth_data)) * 100
    
    # Variant 2:
    # LOGGING:
    pred_lines_v2 = []

    variant2_correct = 0  
    for inp, oup in tqdm(birth_data, desc="Variant 2"):

        # FIX: REMOVE WHITESPACES
        inp = inp.strip()
        oup = oup.strip()

        # get name from input
        inp_heading = "Where was "
        inp_ending = " born?"
        name = inp[len(inp_heading):-len(inp_ending)] # slice just name
        reformat_inp = "What is the birthplace of " + name + "?" # from 4c 

        # need to use chat template for Qwen: https://discuss.huggingface.co/t/fine-tune-a-minimal-llm-model-with-rtx-2050-gpu/171003
        messages = [
            {
                "role": "user",
                "content": reformat_inp
            }
        ]
        templ_inp = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input2 = tokenizer(templ_inp, return_tensors="pt").to(model.device)

        # determine number of tokens in input 
        tokens_input = input2["input_ids"] #https://discuss.huggingface.co/t/fine-tune-a-minimal-llm-model-with-rtx-2050-gpu/171003
        num_tokens_input = len(tokens_input[0])

        with torch.no_grad():
            output = model.generate(**input2, max_new_tokens=10, do_sample=False, use_cache=True)

        output_text = tokenizer.decode(output[0, num_tokens_input:], skip_special_tokens=True)
        output_text = output_text.strip() # FIX: strip the output so we can match with oup

        if (oup == output_text):
            variant2_correct += 1

        # LOGGING: 
        pred_lines_v2.append(f"{inp}\t{oup}\t{output_text}")
    
    variant2_accuracy = (variant2_correct / len(birth_data)) * 100

    with open("modern_llm__variant1_v1_debug.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(pred_lines_v1))

    with open("modern_llm__variant1_v2_debug.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(pred_lines_v2))

    ### END YOUR CODE ###
    print("Variant 1 acc:", variant1_accuracy)
    print("Variant 2 acc:", variant2_accuracy)
    return variant1_accuracy, variant2_accuracy

if __name__ == '__main__':
    variant1_accuracy, variant2_accuracy = main()
    with open("modern_llm_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{variant1_accuracy}\n")
        f.write(f"{variant2_accuracy}\n")