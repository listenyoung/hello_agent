import re
from typing import Optional, List, Dict, Any

from hello_agents import ReflectionAgent, HelloAgentsLLM, Config, Message, ToolRegistry


INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""


REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在算法效率上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种算法上更优的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议。
如果代码在算法层面已经达到最优，才能回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
"""


REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}

# 评审员的反馈:
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""


class Memory:
    """一个简单的短期记忆模块，用于存储执行与反思轨迹。"""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        self.records.append({"type": record_type, "content": content})
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        trajectory = []
        for record in self.records:
            if record["type"] == "execution":
                trajectory.append(f"--- 上一轮尝试 (代码) ---\n{record['content']}")
            elif record["type"] == "reflection":
                trajectory.append(f"--- 评审员反馈 ---\n{record['content']}")
        return "\n\n".join(trajectory)

    def get_last_execution(self) -> str:
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]
        return ""


class MyReflectionAgent(ReflectionAgent):
    """
    在第四章 Reflection 基础上，增加工具调用能力的版本。
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        max_tool_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, llm, system_prompt, config, max_iterations, custom_prompts)
        self.tool_registry = tool_registry
        self.max_tool_iterations = max_tool_iterations
        self.memory = Memory()
        self.prompts = custom_prompts or {
            "initial": INITIAL_PROMPT_TEMPLATE,
            "reflect": REFLECT_PROMPT_TEMPLATE,
            "refine": REFINE_PROMPT_TEMPLATE,
        }
        print(
            f"✅ {name} 初始化完成，最大反思轮数: {max_iterations}，"
            f"最大工具迭代次数: {max_tool_iterations}"
        )

    def run(self, input_text: str, **kwargs) -> str:
        """运行支持工具调用的 Reflection Agent。"""
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")
        self.memory = Memory()

        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = self.prompts["initial"].format(task=input_text)
        initial_result = self._get_llm_response(initial_prompt, **kwargs)
        self.memory.add_record("execution", initial_result)

        for i in range(self.max_iterations):
            print(f"\n--- 第 {i + 1}/{self.max_iterations} 轮迭代 ---")

            print("\n-> 正在进行反思...")
            last_result = self.memory.get_last_execution()
            reflect_prompt = self.prompts["reflect"].format(task=input_text, code=last_result)
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            self.memory.add_record("reflection", feedback)

            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break

            print("\n-> 正在进行优化...")
            refine_prompt = self.prompts["refine"].format(
                task=input_text,
                last_code_attempt=last_result,
                feedback=feedback,
            )
            refined_result = self._get_llm_response(refine_prompt, **kwargs)
            self.memory.add_record("execution", refined_result)

        final_result = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终结果:\n{final_result}")

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))
        return final_result

    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """增加工具调用能力。"""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._enhance_prompt_with_tools(prompt)})
        current_iteration = 0
        while current_iteration < self.max_iterations:
            response_text = self.llm.invoke(messages,**kwargs) or ""
            tool_calls = self._parse_tool_calls(response_text)
            if not tool_calls:
                return response_text
            print(f"🔧 检测到 {len(tool_calls)} 个工具调用")
            clean_response = response_text
            tool_results = []
            for call in tool_calls:
                result = self._execute_tool_call(call["tool_name"], call["parameters"])
                tool_results.append(f"{call['tool_name']}: {result}")
                clean_response = clean_response.replace(call["original"], "").strip()
            if clean_response:
                messages.append({"role": "assistant", "content": clean_response})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "工具执行结果如下：\n"
                        + "\n".join(tool_results)
                        + "\n\n请基于这些结果继续完成刚才的任务。"
                    ),
                }
            )
            current_iteration += 1

        return self.llm.invoke(messages, **kwargs) or ""


    def _enhance_prompt_with_tools(self, prompt: str) -> str:
        """把工具说明附加原始提示词后面。"""
        if not self.tool_registry:
            return prompt
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return prompt
        return(
            f"{prompt}\n\n"
            "你可以在需要时调用工具辅助完成任务。\n\n"
            "## 可用工具\n"
            f"{tools_description}\n\n"
            "## 工具调用格式\n"
            "当你需要使用工具时，请在回复中输出：\n"
            "`[TOOL_CALL:tool_name:parameters]`\n"
            "例如：`[TOOL_CALL:search:素数筛法]`\n"
            "或者：`[TOOL_CALL:calculator:100*25]`\n\n"
            "如果不需要工具，就直接正常完成任务。"
        )

    def _parse_tool_calls(self, text: str) -> List[Dict[str, str]]:
        pattern = r"\[TOOL_CALL:([^:]+):([^\]]+)\]"
        matches = re.findall(pattern, text)
        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append(
                {
                    "tool_name": tool_name.strip(),
                    "parameters": parameters.strip(),
                    "original": f"[TOOL_CALL:{tool_name}:{parameters}]",
                }
            )
        return tool_calls

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        try:
            return self.tool_registry.execute_tool(tool_name, parameters)
        except Exception as e:
            return f"❌ 工具调用失败：{str(e)}"
