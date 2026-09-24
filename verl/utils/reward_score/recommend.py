# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str):
    # Find all <answer>...</answer> occurrences (non-greedy)
    matches = re.findall(r'<answer>(.+?)</answer>', solution_str, flags=re.DOTALL)
    if not matches:
        return None
    # Return the last occurrence, trimmed
    return matches[-1].strip()


def compute_score(data_source, solution_str, ground_truth, extra_info=None, format_score=0.05, score=1.0, tool_penalty_lambda=0.6):
    if data_source == "recommendation":
        answer = extract_solution(solution_str=solution_str)
        if answer is None:
            return 0
        else:
            if ground_truth in answer:
                num_tool_calls = extra_info.get("num_turns", 0) or 0
                # every tool call generates new reward score by multiplying the score by a penalty factor
                final_score = score * (tool_penalty_lambda ** num_tool_calls)
                return final_score
            else:
                return format_score
    else:
        raise ValueError(f"Unsupported data source: {data_source}")
