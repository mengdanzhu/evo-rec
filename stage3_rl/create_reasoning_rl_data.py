import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import os
from torch.utils.data import Dataset
import random
import pandas as pd
from tqdm import tqdm
import data_io

# This dataset is used for reasoning activation task.
# The learning objective is to generate reasoning and answer given user history.
class Reasoning_RL_Dataset(Dataset):
    def __init__(
        self,
        data_file,
        item_file,
        index_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        seed=0,
        category="",
        dedup=False,
    ):
        """
        Fusion dataset combining sequence recommendation with item features.
        Uses semantic IDs for user history, outputs item titles or descriptions.
        
        Args:
            train_file: Path to CSV file with sequence data
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)
        
        # Load sequence data
        self.data = data_io.load_df(data_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        
        # Load item features and indices
        self.item_feat = data_io.load_item_feat(item_file)
        self.indices = data_io.load_indices(index_file)
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        # Build sid2title and sid2description mappings
        self.sid2title = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']                                
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
        
        self.get_inputs()
    
    def __len__(self):
        return len(self.data)
    
    def generate_prompt_title(self, history):
        return f"The user has sequentially interacted with items {history}. Can you recommend the next item for him? Let's think step by step before making recommendation. Directly output the item SID after thinking."
    
    def get_history(self, row):
        history_item_sid = eval(row['history_item_sid'])
        history_str = ", ".join(history_item_sid)
        
        target_sid = row['item_sid']
        
        # Use the new sid2title and sid2description mappings
        if target_sid in self.sid2title:
            target_title = self.sid2title[target_sid]
        else:
            target_title = target_sid
        
        # Check for deduplication
        last_history_sid = history_item_sid[-1] if history_item_sid else None
        is_duplicate = target_sid == last_history_sid
        
        return {
            "history_str": history_str,
            "target_title": target_title,
            "target_sid": target_sid,
            "dedup": is_duplicate,
        }
    
    def generate_formatted_prompt(self, prompt, response):
        return f"""{prompt}"""
    
    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
Can you recommend the next item for the user based on their interaction history?
"""  
        # tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history_data = self.get_history(self.data.iloc[idx])
        
        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None
        
        # Randomly choose between title and description tasks
        prompt = self.generate_prompt_title(history_data['history_str'])
        target = history_data['target_sid']
        # print("fusion prompt: ", prompt)

        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        assistant_response = f"{target}"

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": formatted_prompt},
        ]
        return {
            "input": messages,
            "target": assistant_response,
        }
    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            result = self.pre(i)
            if result is not None:  # Skip None results from deduplication
                inputs.append(result)
        self.inputs = inputs
    
    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else []
    
    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.inputs[idx]
        return self.pre(idx)



# Convert torch dataset to parquet manually
def convert_to_verl_format(ds, split, out_path):
    rows = []
    for idx in range(len(ds)):
        example = ds[idx]
        question_raw = example["input"]
        answer_raw = example["target"]

        rows.append({
            "data_source": data_source,
            "prompt": question_raw,      # must be list[dict]
            "ability": "Recommendation",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer_raw
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            }
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Video_Games",
                        help="Selects every ./data/<Category>/... locator below.")
    parser.add_argument("--train_data_dir", default=None,
                        help="Default: ./data/<Category>/reasoning/")
    parser.add_argument("--eval_data_dir", default=None,
                        help="Default: ./data/<Category>/seqrec/validation/")
    parser.add_argument("--local_dir", default=None,
                        help="Where the verl parquets are written. "
                             "Default: ./data/<Category>/rl")
    parser.add_argument("--item_file", default=None,
                        help="Default: ./data/<Category>/catalog/")
    parser.add_argument("--index_file", default=None,
                        help="Default: ./data/<Category>/catalog/")
    args = parser.parse_args()

    category = args.category
    args.train_data_dir = args.train_data_dir or f"./data/{category}/reasoning/"
    args.eval_data_dir = args.eval_data_dir or f"./data/{category}/seqrec/validation/"
    args.local_dir = args.local_dir or f"./data/{category}/rl"
    args.item_file = args.item_file or f"./data/{category}/catalog/"
    args.index_file = args.index_file or f"./data/{category}/catalog/"

    data_source = f"rec/{category}"
    train_dataset = Reasoning_RL_Dataset(
        data_file=args.train_data_dir,
        item_file=args.item_file,
        index_file=args.index_file,
        tokenizer=None,
        max_len=2048,
        sample=-1,
        seed=0,
        category=category,
        dedup=False,
    )

    eval_dataset = Reasoning_RL_Dataset(
        data_file=args.eval_data_dir,
        item_file=args.item_file,
        index_file=args.index_file,
        tokenizer=None,
        max_len=2048,
        sample=-1,
        seed=0,
        category=category,
        dedup=False,
    )
    local_dir = args.local_dir
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    train_save_path = os.path.join(args.local_dir, "train.parquet")
    val_save_path = os.path.join(args.local_dir, "validation.parquet")

    convert_to_verl_format(train_dataset, split="train", out_path=train_save_path)
    convert_to_verl_format(eval_dataset, split="validation", out_path=val_save_path)


    # Debugging
    df_train = pd.read_parquet(train_save_path)
    df_val = pd.read_parquet(val_save_path)

    print("=== First 3 Data in Training Set ===")
    print(df_train.head(3).to_dict(orient="records"))

    print("\n=== First 3 Data in Validation Set ===")
    print(df_val.head(3).to_dict(orient="records"))
