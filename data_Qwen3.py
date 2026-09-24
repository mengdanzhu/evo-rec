import pandas as pd
from torch.utils.data import Dataset
import random
from tqdm import tqdm

import data_io


def mask_assistant_response_only(
    tokenizer,
    messages,
    assistant_response,
    max_len=None,
    mask_eos=True,
):
    """
    Build labels so that only assistant_response tokens contribute to loss.
    Everything before and after is masked.
    """

    # --- 1. raw text ---
    raw_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )

    # --- 2. locate assistant_response start ---
    pos = raw_text.rfind(assistant_response)
    if pos == -1:
        raise ValueError("assistant response not found in raw_text")

    # --- 3. tokenize prefix ---
    prefix_ids = tokenizer.encode(raw_text[:pos], add_special_tokens=False)
    prefix_len = len(prefix_ids)

    # --- 4. tokenize assistant_response itself ---
    response_ids = tokenizer.encode(assistant_response, add_special_tokens=False)
    response_len = len(response_ids)

    # --- 5. full tokenized sequence ---
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
    )
    if "input_ids" in full_ids:
        full_ids = full_ids["input_ids"]

    # --- 6. labels = mask prefix + keep response + mask suffix ---
    total_len = len(full_ids)
    labels = [-100] * total_len

    response_start = prefix_len
    if mask_eos:
        response_end = prefix_len + response_len
        if response_end > total_len:
            raise ValueError("response range exceeds total length")
        labels[response_start:response_end] = full_ids[response_start:response_end]

    else:
        labels[response_start:] = full_ids[response_start:]

    # --- 7. attention mask ---
    attention_mask = [1] * total_len

    # --- 8. truncate if needed ---
    if max_len is not None and total_len > max_len:
        full_ids = full_ids[-max_len:]
        attention_mask = attention_mask[-max_len:]
        labels = labels[-max_len:]

    return full_ids, attention_mask, labels



class TitleHistory2TitleSFTDataset(Dataset):
    """SFT task: the user's *title* history -> the next item's *title* (text space).

    No semantic IDs are involved. Each sample pairs the same task with one of
    several paraphrased instructions (chosen deterministically per sample) for
    instruction diversity; only the assistant response contributes to the loss.
    """

    def __init__(
        self,
        data_category,
        tokenizer,
        split="train",
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        mask_assistant: bool = True,
    ):
        self.data = data_io.load_seqrec(data_category, split)
        random.seed(seed)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)

        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.mask_assistant = mask_assistant

        # Paraphrased instructions for the same task (instruction diversity).
        self.instructs = [
            f"Given a list of {category} the user recently enjoy, please write a new {category} that the user may bought",
            f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
            f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
            f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
            f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
            f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
            f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
            f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
            f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
            f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
            f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history.",
        ]
        # Fix one instruction per sample up front (seed-fixed) so lazy tokenization
        # reproduces the same sample<->instruction mapping every epoch.
        self._instr_idx = [random.randint(0, len(self.instructs) - 1)
                           for _ in range(len(self.data))]

    def __len__(self):
        return len(self.data)

    def get_history(self, row):
        titles = eval(row["history_item_title"])
        history = ",\t".join(f'"{t}"' for t in titles)
        return {
            "input": f"The user has played the following {self.category}s before: {history}",
            "output": f'"{row["item_title"]}"\n',
        }

    def pre(self, idx):
        instruction = (
            "Below is an instruction that describes a task, paired with an input that "
            "provides further context. Write a response that appropriately completes the request.\n"
            f"{self.instructs[self._instr_idx[idx]]}\n \n"
        )
        history = self.get_history(self.data.iloc[idx])
        assistant_response = history["output"] if not self.test else ""

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": history["input"]},
            {"role": "assistant", "content": assistant_response},
        ]

        if self.mask_assistant:
            input_ids, attention_mask, labels = mask_assistant_response_only(
                tokenizer=self.tokenizer,
                messages=messages,
                assistant_response=assistant_response,
                max_len=self.max_len,
            )
        else:
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=self.test, tokenize=True, return_tensors=None)
            labels = list(input_ids)
            if len(input_ids) >= self.max_len:
                input_ids = input_ids[-self.max_len:]
                labels = labels[-self.max_len:]
            attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __getitem__(self, idx):
        return self.pre(idx)
 

class SidHistory2SidSFTDataset(Dataset):
    def __init__(
        self,
        data_category,
        tokenizer,
        split="train",
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        K=4,
        dedup=False,
        mask_assistant: bool = True,
    ):
        self.data = data_io.load_seqrec(data_category, split)
        random.seed(seed)
        
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.mask_assistant = mask_assistant
        # Lazy tokenization: each sample is tokenized on-the-fly in __getitem__.
    
    def __len__(self):
        return len(self.data)

    def generate_prompt(self, data_point):
        return f"""{data_point["input"]}"""

    def get_history(self, row):
        row['history_item_sid'] = eval(row['history_item_sid'])
        L = len(row['history_item_sid']) 
        history = ""
        history_str = ", ".join(row["history_item_sid"])
        for i in range(L):
            if i == 0:
                history += row['history_item_sid'][i]
            else:
                history += ", " + row['history_item_sid'][i]      
        target_item = str(row['item_sid'])
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        return {"input": f"The user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?",
                "output": target_item.strip(),
                "history_str": history_str,
                "dedup": target_item_sid == last_history_item_sid}
    
    def pre(self, idx):
        instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Can you predict the next possible item that the user may expect?"
        history = self.get_history(self.data.iloc[idx])
        target_item = history["output"]

        history_for_prompt = history.copy()
        history_for_prompt["output"] = ""
        prompt = self.generate_prompt(history_for_prompt)

        assistant_response = target_item.strip() if not self.test else ""
        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": prompt},
        ]
        messages.append({"role": "assistant", "content": assistant_response})

        input_ids, attention_mask, labels = mask_assistant_response_only(
            tokenizer=self.tokenizer,
            messages=messages,
            assistant_response=assistant_response,
            max_len=self.max_len,
        )

        if len(input_ids) >= self.max_len:
            print(len(input_ids))
            input_ids = input_ids[-self.max_len:]
            attention_mask = attention_mask[-self.max_len:]
            labels = labels[-self.max_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


    def __getitem__(self, idx):
        return self.pre(idx)


class SidNextItemEvalDataset(Dataset):

    def __init__(
        self,
        train_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        K=4,
        dedup=False,
        mask_assistant: bool = True,
    ):
        self.data = data_io.load_df(train_file)
        random.seed(seed)
        
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.mask_assistant = mask_assistant
        self.get_inputs()  


    def __len__(self):
        return len(self.data)

    
    def generate_prompt(self, data_point):
        return f"""{data_point["input"]}"""

    def get_history(self, row):
        row['history_item_sid'] = eval(row['history_item_sid'])
        L = len(row['history_item_sid']) 
        history = ""
        for i in range(L):
            if i == 0:
                history += row['history_item_sid'][i]
            else:
                history += ", " + row['history_item_sid'][i]      
        target_item = str(row['item_sid'])
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        return {"input": # f"The user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?",
                f"Can you predict the next possible item the user may expect, given the following chronological interaction history: {history}",
                "output": target_item + '\n',
                "dedup": target_item_sid == last_history_item_sid}
    
    
    def pre(self, idx):
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 
Can you predict the next possible item that the user may expect?
"""
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history_for_prompt = history.copy()
        history_for_prompt['output'] = ''
        prompt = self.generate_prompt(history_for_prompt)

        assistant_response = target_item if not self.test else ""
        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": prompt},
        ]

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True if self.test else False,
            tokenize=True,
            return_tensors=None,
        )
        attention_mask = [1] * len(tokenized)

        if self.test:
            prefix_prompt = "<think>\n</think>\n\n"
            prefix_prompt_ids = self.tokenizer.encode(prefix_prompt)
            tokenized = tokenized + prefix_prompt_ids
            attention_mask = attention_mask + [1] * len(prefix_prompt_ids)

            len_prompt = self.max_len + len(prefix_prompt_ids)

            if len(tokenized) >= len_prompt:
                print(len(tokenized))
                tokenized = tokenized[-len_prompt:]
                attention_mask = attention_mask[-len_prompt:]
            return {
                "input_ids": tokenized,
                "attention_mask": attention_mask,
            }

        messages.append({"role": "assistant", "content": assistant_response})
        input_ids, attention_mask, labels = mask_assistant_response_only(
            tokenizer=self.tokenizer,
            messages=messages,
            assistant_response=assistant_response,
            max_len=self.max_len,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    

    
    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
            
        self.inputs = inputs
    
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp

    def __getitem__(self, idx):
        return self.inputs[idx]



class TitleSidTranslationDataset(Dataset):
    def __init__(
        self,
        data_category,
        tokenizer=None,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        task_type=None, # select from ['sid2title', 'title2sid'] to indicate the task type, set to None for both
        mask_assistant: bool = True,
    ):
        """
        Dataset for sid2title and title2sid tasks.
        
        Args:
            data_category: dataset category directory under ./data/
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
        """
        random.seed(seed)
        
        # Load item features and indices
        self.item_feat = data_io.load_item_features(data_category)
        self.indices = data_io.load_sid_indices(data_category)
        
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.system_prompt = "You are a helpful assistant for mapping between item titles and semantic IDs."
        self.instruction_text = "Answer the question about item identification."
        self.mask_assistant = mask_assistant
        
        # Build sid2title and title2sid mappings
        self.sid2title = {}
        self.title2sid = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
                    self.title2sid[title] = combined_sid
        
        # Create data samples
        self.data = []
        
        # Create sid2title samples
        if task_type is None or task_type == 'sid2title':
            for sid, title in self.sid2title.items():
                self.data.append({
                    'task': 'sid2title',
                    'input': sid,
                    'output': title
                })
        
        # Create title2sid samples
        if task_type is None or task_type == 'title2sid':
            for title, sid in self.title2sid.items():
                self.data.append({
                    'task': 'title2sid',
                    'input': title,
                    'output': sid
                })
        
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        
        # Lazy tokenization: __getitem__ tokenizes each sample on-the-fly.
    
    def __len__(self):
        return len(self.data)
    
    def generate_prompt(self, data_point):
        if data_point['task'] == 'title2sid':
            prompt = f"Which item has the title: {data_point['input']}?"
        else:  # sid2title
            prompt = f'What is the title of item "{data_point["input"]}"?'
        
        return f"""{prompt}"""
    
    def pre(self, idx):
        if self.tokenizer is None:
            return self.data[idx]

        data_point = self.data[idx]
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 
Answer the question about item identification.
"""
        user_prompt = self.generate_prompt(data_point)
        assistant_response = data_point["output"]

        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": user_prompt},
        ]
        if self.test:
            messages.append({"role": "assistant", "content": ""})
        else:
            messages.append({"role": "assistant", "content": assistant_response})

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True if self.test else False,
            tokenize=True,
            return_tensors=None,
        )
        attention_mask = [1] * len(tokenized)

        if self.mask_assistant:
            input_ids, attention_mask, labels = mask_assistant_response_only(
                tokenizer=self.tokenizer,
                messages=messages,
                assistant_response=assistant_response,
                max_len=self.max_len,
            )
        else:
            input_ids = tokenized
            labels = list(tokenized)
            if len(input_ids) > self.max_len:
                print(f"Sequence length {len(input_ids)} exceeds max_len {self.max_len}")
                input_ids = input_ids[-self.max_len:]
                attention_mask = attention_mask[-self.max_len:]
                labels = labels[-self.max_len:]
            else:
                attention_mask = [1] * len(input_ids)

        if len(input_ids) > self.max_len:
            print(f"Sequence length {len(input_ids)} exceeds max_len {self.max_len}")
            input_ids = input_ids[-self.max_len:]
            attention_mask = attention_mask[-self.max_len:]
            labels = labels[-self.max_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def __getitem__(self, idx):
        return self.pre(idx)

class SidHistory2TitleSFTDataset(Dataset):
    def __init__(
        self,
        data_category,
        tokenizer,
        split="train",
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        dedup=False,
        mask_assistant: bool = True,
    ):
        """
        Fusion dataset combining sequence recommendation with item features.
        Uses semantic IDs for user history, outputs item titles or descriptions.
        
        Args:
            data_category: dataset category directory under ./data/
            split: Sequential-recommendation split to load
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)
        
        # Load sequence data
        self.data = data_io.load_seqrec(data_category, split)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        
        # Load item features and indices
        self.item_feat = data_io.load_item_features(data_category)
        self.indices = data_io.load_sid_indices(data_category)
        
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.mask_assistant = mask_assistant
        # Build sid2title and sid2description mappings
        self.sid2title = {}
        self.sid2description = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                description = self.item_feat[item_id]['description']
                
                # Process description according to requirements:
                # 1. If description is empty, use title
                # 2. If description is a list, select the longest one
                # 3. If the longest in list is also empty, use title
                processed_description = self._process_description(description, title)
                
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
                    self.sid2description[combined_sid] = processed_description
        # print("self.sid2title: ", self.sid2title)
        # print("self.sid2description: ", self.sid2description)
        # Lazy tokenization: __getitem__ tokenizes each sample on-the-fly.
    
    def _process_description(self, description, title):
        """
        Process description according to the requirements:
        1. If description is empty, use title
        2. If description is a list, select the longest one
        3. If the longest in list is also empty, use title
        
        Args:
            description: The description field from item_feat
            title: The title field from item_feat
        
        Returns:
            str: Processed description
        """
        # Check if description is empty or None
        if not description or description == '':
            return title
        
        # Check if description is a list (either actual list or string representation)
        if isinstance(description, list):
            # It's already a list
            desc_list = description
        elif isinstance(description, str) and description.startswith('[') and description.endswith(']'):
            try:
                # Try to parse string representation of list
                desc_list = eval(description)
            except:
                # If parsing fails, treat as regular string
                return description if description.strip() else title
        else:
            # Regular string description
            return description if description.strip() else title
        
        # If we have a list, find the longest non-empty item
        if desc_list:
            # Filter out empty strings and find the longest
            non_empty_descriptions = [desc for desc in desc_list if desc and desc.strip()]
            if non_empty_descriptions:
                # Return the longest description
                longest_desc = max(non_empty_descriptions, key=len)
                return longest_desc
            else:
                # All descriptions in list are empty, use title
                return title
        else:
            # Empty list, use title
            return title
    
    def __len__(self):
        return len(self.data)
    
    def generate_prompt_title(self, history):
        return f"The user has sequentially interacted with items {history}. Can you recommend the next item for him? Tell me the title of the item?"
    
    def get_history(self, row):
        history_item_sid = eval(row['history_item_sid'])
        history_str = ", ".join(history_item_sid)
        
        target_sid = row['item_sid']
        
        # Use the new sid2title and sid2description mappings
        if target_sid in self.sid2title:
            target_title = self.sid2title[target_sid]
        else:
            target_title = target_sid
            
        if target_sid in self.sid2description:
            target_description = self.sid2description[target_sid]
            # Clean description if it's a string representation of a list
            if isinstance(target_description, str) and target_description.startswith("['") and target_description.endswith("']"):
                try:
                    desc_list = eval(target_description)
                    target_description = desc_list[0] if desc_list else target_description
                except:
                    pass  # Keep original if eval fails
        else:
            target_description = f"An item with semantic ID {target_sid}"
        
        # Check for deduplication
        last_history_sid = history_item_sid[-1] if history_item_sid else None
        is_duplicate = target_sid == last_history_sid
        
        return {
            "history_str": history_str,
            "target_title": target_title,
            "target_description": target_description,
            "target_sid": target_sid,
            "dedup": is_duplicate
        }
    
    def generate_formatted_prompt(self, prompt, response):
        return f"""{prompt}"""
    
    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
Can you recommend the next item for the user based on their interaction history?
"""
        history_data = self.get_history(self.data.iloc[idx])

        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None

        prompt = self.generate_prompt_title(history_data['history_str'])
        target = history_data['target_title']

        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        assistant_response = target if not self.test else ""

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": formatted_prompt},
        ]
        messages.append({"role": "assistant", "content": assistant_response})

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True if self.test else False,
            tokenize=True,
            return_tensors=None,
        )
        attention_mask = [1] * len(tokenized)

        if self.mask_assistant:
            input_ids, attention_mask, labels = mask_assistant_response_only(
                tokenizer=self.tokenizer,
                messages=messages,
                assistant_response=assistant_response,
                max_len=self.max_len,
            )
        else:
            input_ids = tokenized
            labels = list(tokenized)
            if len(input_ids) >= self.max_len:
                print(f"Sequence length {len(input_ids)} exceeds max_len {self.max_len}")
                input_ids = input_ids[-self.max_len:]
                attention_mask = attention_mask[-self.max_len:]
                labels = labels[-self.max_len:]
            else:
                attention_mask = [1] * len(input_ids)

        if len(input_ids) > self.max_len:
            print(f"Sequence length {len(input_ids)} exceeds max_len {self.max_len}")
            input_ids = input_ids[-self.max_len:]
            attention_mask = attention_mask[-self.max_len:]
            labels = labels[-self.max_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def __getitem__(self, idx):
        return self.pre(idx)


class TitleHistory2SidSFTDataset(Dataset):
    def __init__(
        self,
        data_category,
        tokenizer,
        split="train",
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        dedup=False,
        mask_assistant: bool = True,
    ):
        """
        SFT dataset that uses item titles in user history to predict next item's semantic ID.
        
        Args:
            data_category: dataset category directory under ./data/
            split: Sequential-recommendation split to load
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)
        
        # Load sequence data
        self.data = data_io.load_seqrec(data_category, split)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        
        # Load item features and indices
        self.item_feat = data_io.load_item_features(data_category)
        self.indices = data_io.load_sid_indices(data_category)
        
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.mask_assistant = mask_assistant
        
        # Build item_id to semantic ID mapping
        self.id2sid = {}
        for item_id, sids in self.indices.items():
            if len(sids) >= 3:
                combined_sid = sids[0] + sids[1] + sids[2]
                self.id2sid[item_id] = combined_sid
        
        # Lazy tokenization: __getitem__ tokenizes each sample on-the-fly.
    
    def __len__(self):
        return len(self.data)
    
    def generate_prompt(self, data_point):
        return f"""{data_point["input"]}"""
    
    def get_history(self, row):
        """Extract user history from title sequence and target semantic ID"""
        # Parse history_item_title field
        history_item_title = eval(row['history_item_title'])
        
        # Format title sequence for prompt
        history_titles = ", ".join([f'"{title}"' for title in history_item_title])
        
        # Get target item's semantic ID from item_id
        target_item_id = str(row['item_id'])
        if target_item_id in self.id2sid:
            target_sid = self.id2sid[target_item_id]
        else:
            target_sid = target_item_id  # Fallback to item_id if no semantic ID found
        
        # Check for deduplication if needed
        is_duplicate = False
        if self.dedup and 'history_item_id' in row:
            try:
                history_item_id = eval(row['history_item_id'])
                last_history_item_id = str(history_item_id[-1]) if history_item_id else None
                is_duplicate = target_item_id == last_history_item_id
            except:
                is_duplicate = False
        
        return {
            "input": f"The user has interacted with the following {self.category} items in chronological order: {history_titles}. Can you predict the next item the user may expect?",
            "output": target_sid,
            "history_titles": history_titles,
            "target_sid": target_sid,
            "dedup": is_duplicate
        }
    
    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 
Based on the user's historical interaction with item titles, predict the semantic ID of the next item they may expect.
"""
        history_data = self.get_history(self.data.iloc[idx])
        
        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None
        
        target_output = history_data['output']
        history_for_prompt = history_data.copy()
        history_for_prompt['output'] = ''
        prompt = self.generate_prompt(history_for_prompt)

        assistant_response = target_output if not self.test else ""
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ]
        messages.append({"role": "assistant", "content": assistant_response})

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True if self.test else False,
            tokenize=True,
            return_tensors=None,
        )
        attention_mask = [1] * len(tokenized)

        if self.mask_assistant:
            input_ids, attention_mask, labels = mask_assistant_response_only(
                tokenizer=self.tokenizer,
                messages=messages,
                assistant_response=assistant_response,
                max_len=self.max_len,
            )
        else:
            input_ids = tokenized
            labels = list(tokenized)
            if len(input_ids) >= self.max_len:
                print(f"Sequence length {len(input_ids)} exceeds max_len {self.max_len}")
                input_ids = input_ids[-self.max_len:]
                attention_mask = attention_mask[-self.max_len:]
                labels = labels[-self.max_len:]
            else:
                attention_mask = [1] * len(input_ids)

        if len(input_ids) > self.max_len:
            print(f"Sequence length {len(input_ids)} exceeds max_len {self.max_len}")
            input_ids = input_ids[-self.max_len:]
            attention_mask = attention_mask[-self.max_len:]
            labels = labels[-self.max_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def __getitem__(self, idx):
        result = self.pre(idx)
        return result if result is not None else {"input_ids": [], "attention_mask": [], "labels": []}


# This class reads LLM-generated coherent data mixing sid and natural language.
class SidTextInterleaveItemDataset(Dataset):
    def __init__(
        self,
        data_category,
        tokenizer,
        max_len=2048,
        sample=-1,
        seed=0,
    ):
        self.json_data = data_io.load_item_narratives(data_category)
        random.seed(seed)

        if sample > 0:
            # ``load_enhanced`` returns a dict {id: {...}}, which has no
            # ``.sample()`` (that is a pandas method). Sample by key instead.
            keys = list(self.json_data.keys())
            keys = random.sample(keys, min(sample, len(keys)))
            self.json_data = {k: self.json_data[k] for k in keys}

        # ``sid_interleaved_narrative`` is the item-level SID/text narrative (a
        # parquet column of the same name). Its legacy field name was
        # ``llm_stage2``, where "stage2" meant the offline data-generation step
        # -- NOT training Stage 2.
        self.data = {}
        real_id = 0
        for item_meta in self.json_data.values():
            if "sid_interleaved_narrative" in item_meta:
                self.data[real_id] = item_meta["sid_interleaved_narrative"]
                real_id += 1
    
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Lazy tokenization: __getitem__ tokenizes each sample on-the-fly.


    def __len__(self):
        return len(self.data)

    def pre(self, idx):
        item_desc = self.data[idx]
        # Truncate <think> ... <\think> parts if exists
        if "</think>" in item_desc:
            item_desc = item_desc.split("</think>")[-1].strip()

        # Tokenize as plain text
        tokenized = self.tokenizer.encode(
            item_desc,
            add_special_tokens=False
        )
        # Truncate
        if len(tokenized) > self.max_len:
            tokenized = tokenized[-self.max_len:]
            print(f"Truncated sequence at idx {idx} to max_len {self.max_len}. Original length was {len(tokenized)}.")

        # LM target = shift left by 1
        labels = tokenized.copy()
        attention_mask = [1] * len(tokenized)
        return {
            "input_ids": tokenized,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __getitem__(self, idx):
        return self.pre(idx)





# This class reads LLM-generated coherent data mixing sid and natural language.
class SidTextInterleaveSequenceDataset(Dataset):
    def __init__(
        self,
        data_category,
        tokenizer,
        max_len=2048,
        sample=-1,
        seed=0,
    ):
        self.csv_data = data_io.load_sequence_narratives(data_category)
        random.seed(seed)

        if sample > 0:
            self.csv_data = self.csv_data.sample(sample, random_state=seed)

        # Drop empty narratives up front (cheap, no tokenization) so the lazy
        # __getitem__ never has to skip an index.
        self.data = [x for x in self.csv_data['integrated_narrative'].tolist() if x is not None]
        self.tokenizer = tokenizer
        self.max_len = max_len


    def __len__(self):
        return len(self.data)

    def pre(self, idx):
        item_desc = self.data[idx]
        if item_desc is None:
            return None
        
        if "</think>" in item_desc:
            item_desc = item_desc.split("</think>")[-1].strip()

        # Tokenize as plain text
        tokenized = self.tokenizer.encode(
            item_desc,
            add_special_tokens=False
        )
        # Truncate
        if len(tokenized) > self.max_len:
            tokenized = tokenized[-self.max_len:]
            print(f"Truncated sequence at idx {idx} to max_len {self.max_len}. Original length was {len(tokenized)}.")

        # LM target = shift left by 1
        labels = tokenized.copy()
        attention_mask = [1] * len(tokenized)
        return {
            "input_ids": tokenized,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __getitem__(self, idx):
        return self.pre(idx)




# This dataset is used for reasoning activation task.
# The learning objective is to generate reasoning and answer given user history.
class ReasoningActivationDataset(Dataset):
    def __init__(
        self,
        reasoning_train_file,
        item_file,
        index_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
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
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)
        
        # Load sequence data
        self.data = data_io.load_df(reasoning_train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        
        # Load item features and indices
        self.item_feat = data_io.load_item_feat(item_file)
        self.indices = data_io.load_indices(index_file)
        
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        # Build sid2title and sid2description mappings
        self.sid2title = {}
        self.sid2description = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                description = self.item_feat[item_id]['description']
                
                processed_description = self._process_description(description, title)
                
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
                    self.sid2description[combined_sid] = processed_description
        
        # Lazy tokenization: each sample is tokenized on-the-fly in __getitem__.
        # Drop rows whose reasoning is missing / unclosed up front (cheap, no
        # tokenization) so lazy __getitem__ never has to skip an index. This
        # reproduces exactly the rows the original eager get_inputs() filtered out.
        def _valid_reasoning(r):
            if pd.isna(r):
                return False
            r = str(r).strip()
            if r == "":
                return False
            if r.startswith("<think>") and "</think>" not in r:
                return False
            return True
        self.data = self.data[self.data['reasoning_path'].apply(_valid_reasoning)].reset_index(drop=True)
    
    
    def _process_description(self, description, title):
        """
        Process description according to the requirements:
        1. If description is empty, use title
        2. If description is a list, select the longest one
        3. If the longest in list is also empty, use title
        
        Args:
            description: The description field from item_feat
            title: The title field from item_feat
        
        Returns:
            str: Processed description
        """
        # Check if description is empty or None
        if not description or description == '':
            return title
        
        # Check if description is a list (either actual list or string representation)
        if isinstance(description, list):
            # It's already a list
            desc_list = description
        elif isinstance(description, str) and description.startswith('[') and description.endswith(']'):
            try:
                # Try to parse string representation of list
                desc_list = eval(description)
            except:
                # If parsing fails, treat as regular string
                return description if description.strip() else title
        else:
            # Regular string description
            return description if description.strip() else title
        
        # If we have a list, find the longest non-empty item
        if desc_list:
            # Filter out empty strings and find the longest
            non_empty_descriptions = [desc for desc in desc_list if desc and desc.strip()]
            if non_empty_descriptions:
                # Return the longest description
                longest_desc = max(non_empty_descriptions, key=len)
                return longest_desc
            else:
                # All descriptions in list are empty, use title
                return title
        else:
            # Empty list, use title
            return title
    
    def __len__(self):
        return len(self.data)
    
    def generate_prompt_title(self, history):
        return f"The user has sequentially interacted with items {history}. Can you recommend the next item for him? Let's think step by step before making recommendation. Directly output the item SID after thinking."
    
    def get_history(self, row):
        history_item_sid = eval(row['history_item_sid'])
        history_str = ", ".join(history_item_sid)
        
        target_sid = row['item_sid']
        reasoning = row['reasoning_path']
        # return None if reasoning is empty or nan
        if pd.isna(reasoning) or reasoning.strip() == "":
            return None


        # if reasoning.strip() start with <think>, then we need to remove content between <think> and </think>
        if reasoning.strip().startswith("<think>"):
            if "</think>" in reasoning:
                reasoning = reasoning.split("</think>")[-1].strip()
            else:
                return None
        
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
            "reasoning": reasoning,
        }
    
    def generate_formatted_prompt(self, prompt, response):
        return f"""{prompt}"""
    
    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
Can you recommend the next item for the user based on their interaction history?
"""  
        # tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history_data = self.get_history(self.data.iloc[idx])
        if history_data is None:
            return None
        
        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None
        
        # Randomly choose between title and description tasks
        prompt = self.generate_prompt_title(history_data['history_str'])
        target = history_data['target_sid']
        # print("fusion prompt: ", prompt)

        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        assistant_response = f"<think>\n{history_data['reasoning'].strip()}\n</think>\n\n{target}"

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": formatted_prompt},
        ]
        messages.append({"role": "assistant", "content": assistant_response})

        input_ids, attention_mask, labels = mask_assistant_response_only(
            tokenizer=self.tokenizer,
            messages=messages,
            assistant_response=assistant_response,
            max_len=self.max_len,
            mask_eos=False,
        )


        if len(input_ids) > self.max_len:
            print(f"Sequence length {len(input_ids)} exceeds max_len {self.max_len}")
            input_ids = input_ids[-self.max_len:]
            attention_mask = attention_mask[-self.max_len:]
            labels = labels[-self.max_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.inputs[idx]
        return self.pre(idx)




# This dataset is used for reasoning recommendation task.
# The learning objective is to generate reasoning then answer given user history.
class ReasoningEvalDataset(Dataset):
    def __init__(
        self,
        data_file,
        item_file,
        index_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
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
            test: Whether this is test mode
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
        self.test = test
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
            "output": target_sid,
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
        target = history_data['output']
        # print("fusion prompt: ", prompt)

        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        assistant_response = f"{target}"

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": formatted_prompt},
        ]

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True if self.test else False,
            tokenize=True,
            return_tensors=None,
        )
        attention_mask = [1] * len(tokenized)

        if len(tokenized) >= self.max_len:
            print(len(tokenized))
            tokenized = tokenized[self.max_len:]
            attention_mask = attention_mask[self.max_len:]
        return {
            "target": target,
            "input_ids": tokenized,
            "attention_mask": attention_mask,
        }

    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            result = self.pre(i)
            if result is not None:  # Skip None results from deduplication
                inputs.append(result)
        self.inputs = inputs
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.inputs[idx]
        return self.pre(idx)



class GeneralReasoningSFTDataset(Dataset):
    def __init__(self, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        self.data = data_io.load_general_reasoning()
        random.seed(seed)
        if sample > 0:
            self.data = random.sample(self.data, sample)
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.cnt = 0
        # Lazy tokenization: samples are tokenized on-the-fly in __getitem__.
        # Drop rows with no assistant response up front (cheap, no tokenization)
        # so the dataset length / composition match the original eager version.
        def _has_assistant(messages):
            golden = ""
            for elm in messages:
                if elm["role"] == "assistant":
                    golden = elm["content"][0]["text"] if type(elm["content"]) == list else elm["content"]
            return golden != ""
        self.data = [m for m in self.data if _has_assistant(m)]
        
    
    def __len__(self):
        return len(self.data)

    
    def pre(self, idx):
        prompt_messages = []
        for message in self.data[idx]:
            # print(f"message: {message['content']}")
            # message["content"] = eval(message["content"])
            if message["role"] == "user":
                if type(message["content"]) == list:
                    prompt_messages.append({"role": "user", "content": message["content"][0]["text"]})
                else:
                    prompt_messages.append({"role": "user", "content": message["content"]})
            elif message["role"] == "system":
                if type(message["content"]) == list:
                    prompt_messages.append({"role": "user", "content": message["content"][0]["text"]})
                else:
                    prompt_messages.append({"role": "system", "content": message["content"]})
        try:
            processed_template = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"Error processing messages: {self.data[idx]}")
            print(f"Error processing idx {idx}: {e}")
            raise e
        try:
            input_ids = self.tokenizer.encode(processed_template)
        except Exception:
            return None
        
        if idx == 0:
            print(f"General data example: {[processed_template]}")
        
        attention_mask = [1] * len(input_ids)

        if self.test:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        golden_output = ""
        for elm in self.data[idx]:
            if elm["role"] == "assistant":
                if type(elm["content"]) == list:
                    golden_output = elm["content"][0]["text"]
                else:
                    golden_output = elm["content"]

        if golden_output == "":
            print(f"No assistant response found in idx {idx}")
            return None
        
        if idx == 0:
            print(f"General data golden output example: {[golden_output]}")
        # golden_output = sample["messages"]
        try:
            golden_tokens = self.tokenizer.encode(golden_output)
        except Exception:
            return None
        
        golden_tokens = golden_tokens + [self.tokenizer.eos_token_id]
        
        input_prompt_len = len(input_ids)
        input_ids = input_ids + golden_tokens
        attention_mask = [1] * len(input_ids)
        labels = [-100] * input_prompt_len + input_ids[input_prompt_len:]
        
        # if len(input_ids) >= self.max_len:
        #     print(len(input_ids))

        return {
            "input_ids": input_ids[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }
    
    def __getitem__(self, idx):
        out = self.pre(idx)
        if out is None:
            # A few raw rows have no assistant turn / fail to encode; fall back to
            # the next usable sample instead of dropping the index.
            for j in range(1, len(self.data)):
                out = self.pre((idx + j) % len(self.data))
                if out is not None:
                    break
        return out
