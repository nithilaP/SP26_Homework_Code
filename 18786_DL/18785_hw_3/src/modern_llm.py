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

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()

    print(next(model.parameters()).device)

    # HW CHECK IN   
    # parse the birth dev tsv. 
    with open(args.dev_set_path, 'r', encoding='utf-8') as data_file:
        dataset = []
        for line in data_file:
            # if (line.strip()): # if value
                # line = line.strip()
            if line:
                inp, oup = line.split('\t')      
                dataset.append((inp, oup))

    # Variant 1:
    variant1_correct = 0
    for inp, oup in tqdm(dataset, desc="Variant 1"):

        # FIX: REMOVE WHITESPACES
        inp = inp.strip()
        oup = oup.strip()

        input1 = tokenizer(inp, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(**input1, max_new_tokens=10, do_sample=False, use_cache=True) # do not want random

        gen_i = output[0][input1["input_ids"].shape[-1]:]
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_text = generated_text.strip() # strip the output so we can match with oup

        # output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # generated_text = output_text[len(inp):]
        # generated_text = generated_text.strip() # strip the output so we can match with oup

        if (oup == generated_text):
            variant1_correct += 1

    variant1_accuracy = (variant1_correct / len(dataset)) * 100
    
    # Variant 2:
    variant2_correct = 0  
    for inp, oup in tqdm(dataset, desc="Variant 2"):

        # FIX: REMOVE WHITESPACES
        inp = inp.strip()
        oup = oup.strip()

        # get name from input
        inp_heading = "Where was "
        inp_ending = "born?"
        name = inp[len(inp_heading):-len(inp_ending)] # slice just name
        reformat_inp = name + " was born in?"

        input2 = tokenizer(reformat_inp, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(**input2, max_new_tokens=10, do_sample=False, use_cache=True)

        gen_i = output[0][input2["input_ids"].shape[-1]:]
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_text = generated_text.strip() # strip the output so we can match with oup

        # output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # generated_text = output_text[len(reformat_inp):]
        # generated_text = generated_text.strip() # strip the output so we can match with oup

        if (oup == generated_text):
            variant2_correct += 1
    
    variant2_accuracy = (variant2_correct / len(dataset)) * 100

    ### END YOUR CODE ###
    print("Variant 1 acc:", variant1_accuracy)
    print("Variant 2 acc:", variant2_accuracy)
    return variant1_accuracy, variant2_accuracy

if __name__ == '__main__':
    variant1_accuracy, variant2_accuracy = main()
    with open("modern_llm_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{variant1_accuracy}\n")
        f.write(f"{variant2_accuracy}\n")