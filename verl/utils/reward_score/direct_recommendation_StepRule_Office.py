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

import math
import re
from typing import Optional
_SOLUTION_CLIP_CHARS = 50



def extract_sid_tokens(s: str) -> list[str]:
    # regex: \( <  任意非 > 字符  +  > \)
    pattern = r'<[^>]+>'
    tokens = re.findall(pattern, s)
    return tokens


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 50 characters, which is a safe approximation for 50 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    match = re.search(r"</think>\s*(.*)", solution_str, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        answer_sids = extract_sid_tokens(final_answer)[:3]
        if len(answer_sids) == 3:
            return answer_sids
    return None
    

def calculate_ndcg_at_10(beam_predictions: list[str], ground_truth_sids: list[str]) -> tuple[float, int]:
    for rank, prediction in enumerate(beam_predictions[:10], start=1):
        if extract_sid_tokens(prediction)[:3] == ground_truth_sids:
            return 1.0 / math.log2(rank + 1), rank
    return 0.0, 0



class MyRewardComputer:
    def compute(
        self,
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: Optional[dict] = None,
    ) -> dict[str, float]:
        answer = extract_solution(solution_str=solution_str)
        ground_truth = extract_sid_tokens(ground_truth)[:3]
        beam_predictions = (extra_info or {}).get("sid_beam_predictions")
        if not beam_predictions:
            raise ValueError("sid_beam_predictions are required for NDCG@10 reward")

        ndcg_at_10, beam_rank = calculate_ndcg_at_10(beam_predictions, ground_truth)
        return {
            "score": ndcg_at_10,
            "sid_match_reward": ndcg_at_10,
            "ndcg_at_10": ndcg_at_10,
            "beam_rank": float(beam_rank),
            "hit_at_1": float(beam_rank == 1),
            "hit_at_3": float(0 < beam_rank <= 3),
            "hit_at_5": float(0 < beam_rank <= 5),
            "hit_at_10": float(beam_rank > 0),
            "prefix_1_match": float(answer is not None and answer[0] == ground_truth[0]),
            "prefix_2_match": float(answer is not None and answer[:2] == ground_truth[:2]),
            "exact_match": float(beam_rank == 1),
        }



# ---- 模块级单例（懒加载） ----
_reward_computer: Optional[MyRewardComputer] = None

def _get_reward_computer() -> MyRewardComputer:
    global _reward_computer
    if _reward_computer is None:
        # 只在第一次被调用时初始化一次
        _reward_computer = MyRewardComputer()
    return _reward_computer


# ---- 暴露给 VERL 的函数接口 ----
def rule_base_reward(data_source, solution_str, ground_truth, extra_info=None):
    rc = _get_reward_computer()
    return rc.compute(data_source, solution_str, ground_truth, extra_info)