import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
# import transformers
# import torch
import fire
import math
import numpy as np
    
from tqdm import tqdm
def gao(path, item_path):
    if type(path) != list:
        path = [path]
    CC=0
        
    
    import data_io
    items = data_io.load_info_lines(item_path)
    # item_names = [ _[:-len(_.split('\t')[-1])].strip() for _ in items]
    item_names= [_.split('\t')[0].strip() for _ in items]
    item_ids = [_ for _ in range(len(item_names))]
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:   
            item_dict[item_names[i]].append(item_ids[i])
    
    

    result_dict = dict()
    topk_list = [1, 3, 5, 10, 20, 50]
    n_beam = -1
    for p in path:
        result_dict[p] = {
            "NDCG": [],
            "HR": [],
        }
        f = open(p, 'r')
        import json
        test_data = json.load(f)
        f.close()
        
        text = [ [_.strip("\"\n").strip() for _ in sample["predict"]] for sample in test_data]
        
        for index, sample in tqdm(enumerate(text)):
            if n_beam == -1:
                n_beam = len(sample)
                valid_topk = [k for k in topk_list if k <= n_beam]
                ALLNDCG = np.zeros(len(valid_topk))
                ALLHR = np.zeros(len(valid_topk))
            if type(test_data[index]['output']) == list:
                target_item = test_data[index]['output'][0].strip("\"").strip(" ")
            else:
                target_item = test_data[index]['output'].strip(" \n\"")
            minID = 1000000
            for i in range(len(sample)):
                
                if sample[i] not in item_dict:
                    CC += 1
                    print(sample[i])
                    print(target_item)
                if sample[i] == target_item:
                    minID = i
                    break
            
            for index, topk in enumerate(topk_list):
                if topk > n_beam:
                    continue
                if minID < topk:
                    ALLNDCG[index] = ALLNDCG[index] + (1 / math.log(minID + 2))
                    ALLHR[index] = ALLHR[index] + 1
        print(n_beam)
        valid_topk = [k for k in topk_list if k <= n_beam]
        print(valid_topk)
        print(f"NDCG:\t{ALLNDCG / len(text) / (1.0 / math.log(2))}")
        print(f"HR\t{ALLHR / len(text)}")
        print(CC)

if __name__=='__main__':
    fire.Fire(gao)
