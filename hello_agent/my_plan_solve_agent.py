import ast
from typing import Optional, List, Dict

from hello_agents import PlanAndSolveAgent, HelloAgentsLLM, Config, Message


PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划，```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""


EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""


class Planner:
    """规划器，负责把复杂问题拆成步骤。"""

    def __init__(self, llm: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm = llm
        self.prompt_template = prompt_template or PLANNER_PROMPT_TEMPLATE

    def plan(self, question: str, **kwargs) -> List[str]:
        prompt = self.prompt_template.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        response_text = self.llm.invoke(messages, **kwargs) or ""
        print(f"✅ 计划已生成:\n{response_text}")

        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []


class Executor:
    """执行器，负责按计划逐步求解。"""

    def __init__(self, llm: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm = llm
        self.prompt_template = prompt_template or EXECUTOR_PROMPT_TEMPLATE

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        history = ""
        final_answer = ""

        print("\n--- 正在执行计划 ---")
        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {i}/{len(plan)}: {step}")
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step,
            )
            messages = [{"role": "user", "content": prompt}]

            response_text = self.llm.invoke(messages, **kwargs) or ""
            history += f"步骤 {i}: {step}\n结果: {response_text}\n\n"
            final_answer = response_text
            print(f"✅ 步骤 {i} 已完成，结果: {final_answer}")

        return final_answer


class MyPlanAndSolveAgent(PlanAndSolveAgent):
    """
    基于第四章 Plan-and-Solve 思路实现的自定义 Agent。
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, llm, system_prompt, config)

        planner_prompt = custom_prompts.get("planner") if custom_prompts else None
        executor_prompt = custom_prompts.get("executor") if custom_prompts else None

        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(self.llm, executor_prompt)
        print(f"✅ {name} 初始化完成")

    def run(self, input_text: str, **kwargs) -> str:
        """运行 Plan-and-Solve Agent。"""
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止。"
            print(f"\n--- 任务终止 ---\n{final_answer}")
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))
            return final_answer

        final_answer = self.executor.execute(input_text, plan, **kwargs)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer
