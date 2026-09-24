# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import logging
import os
from typing import Any, Optional
from uuid import uuid4
import re
import asyncio
import numpy as np

from openai import OpenAI
from verl.utils.reward_score import gsm8k
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RecTool(BaseTool):
    """A demo tool for calculating the reward of gsm8k.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.

    NOTE (verl 0.6.0 port): `create()` now returns ``(instance_id, ToolResponse)``
    and `execute()` returns ``(ToolResponse, reward, metrics)`` to match the
    updated BaseTool interface. The original (verl 0.4.1) implementation returned
    a bare ``str`` / ``(str, reward, metrics)``.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "chat_with_user",
                "description": "A tool for chatting with the user to clarify the user intention.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to chat with user.",
                        },
                    },
                    "required": ["query"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        assert "embedding_path" in config, "embedding_path is required in rec_tool config"
        # load item embeddings
        self.item_embedding = np.load(config["embedding_path"])

        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, intention: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "query": "",
            "intention": intention,
            "user_response": "",
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query", "")
        if not isinstance(query, str):
            query = str(query)

        self._instance_dict[instance_id]["query"] = query
        user_response = await self.get_chat_response(instance_id)

        # a constant penalty for launching the chat. Change this later.
        tool_reward = -0.01
        # update the reward
        self._instance_dict[instance_id]["user_response"] = user_response

        return ToolResponse(text=f"{user_response}"), tool_reward, {}

    async def get_chat_response(self, instance_id: str, **kwargs) -> float:
        # response_str = f"You asked: {self._instance_dict[instance_id]['query']}. I would say that {self._instance_dict[instance_id]['intention']}."

        query = self._instance_dict[instance_id]["query"]
        intention = self._instance_dict[instance_id]["intention"]

        # 构造用于 OpenAI 的 prompt
        full_prompt = f'''You are acting as a simulated user in an online shopping recommender system.

GOAL:
- You have a predefined intent profile (e.g., likes/dislikes certain genres).
- Your responses must align with this intent and remain consistent.
- When asked by the recommender (assistant), reply truthfully based on your intent.

RULES:
1. You must always reply with exactly one very short sentence in a distant and neutral tone, like a real user casually using a recommender system.
2. Directly refuse the recommender assistant if any specific item names mentioned. Your replies should stay at the level of general tastes, attributes, or abstract intentions, not concrete items.
3. Do NOT reveal your internal intent profile; just answer naturally and concisely.

INPUT VARIABLES:
- {intention}

QUERY:
- {query}
'''

        # openai_api_key = os.getenv('OPENAI_API_KEY')
        # client = OpenAI(api_key=openai_api_key)

        # response = await client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[{"role": "user", "content": full_prompt}],
        # )
        # response_str = response.choices[0].message.content

        response_str = await asyncio.to_thread(
            lambda: OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                    .chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"user","content":full_prompt}]
                    ).choices[0].message.content
        )
        return response_str

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
