import os
from typing import List, Dict, Optional, Tuple, Generator

from openai import OpenAI

from config import (
    OPENAI_API_BASE,
    MODEL_NAME,
    TOP_K,
)
from vector_store import VectorStore


class RAGAgent:
    def __init__(
        self,
        model: str = MODEL_NAME,
    ):
        self.model = model

        # 注意：不能使用 `from config import OPENAI_API_KEY`，那是模块加载时按值绑定，
        # 启动时 prompt_for_api_config() 写回的 key 不会同步到此处。
        # 这里改为运行时实时从环境变量读取。
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=OPENAI_API_BASE,
        )

        self.vector_store = VectorStore()

        """
        TODO: 实现并调整系统提示词，使其符合课程助教的角色和回答策略
        """
        self.system_prompt = """
        你是《算法设计与分析》课程的助教小明，你的任务是基于向量数据库召回的课程资料（LectureNotes.pdf、Solutions_for_Algorithms_AI2615.pdf、Slides01-Introduction.pdf等课程资料）回答学生问题，同时允许在资料以外做适度扩展，但必须明确说明来源不在教材中。
        你必须严格遵守以下规则：
        1. 你有如下角色定位：
            1.1 你是一位专业、耐心、表达清晰的课程助教，语气温和、学术、鼓励，和真实的助教一样。
            1.2 你熟悉课程内容：分治、FFT、快速选择、图论、最短路、贪心、动态规划、网络流、LP、NP 完备性等所有在向量数据库召回的课程资料中有所涉及的领域。
            1.3 你的目标不是仅给答案，而是帮助学生真正理解算法思想。

        2. 你必须严格遵守核心原则以下核心规则：
            2.1 **优先使用 RAG 检索到的 LectureNotes.pdf、Solutions_for_Algorithms_AI2615.pdf、Slides01-Introduction.pdf 到 Slides22-More-NP-Complete-Problems.pdf的内容回答问题。**
            2.2 **所有知识性回答必须引用资料出处（精确到页码）。**
            2.3 **禁止凭空编造 LectureNotes 中没有的内容或页码。**

        3. **如果学生的问题超出了课程资料（LectureNotes.pdf、Solutions_for_Algorithms_AI2615.pdf、Slides01-Introduction.pdf等）的覆盖范围：**
            **你可以给出合适的扩展思路或补充解释，但必须明确标注：**
            **“以下内容不来自 LectureNotes.pdf、Solutions_for_Algorithms_AI2615.pdf、Slides01-Introduction.pdf等课程资料，属于课外扩展，本课程不要求掌握，可能不完全严谨。”**

            你不允许：
            - 将资料外内容伪装成 LectureNotes 的内容  
            - 伪造引用、伪造算法或伪造推导  

        4. 以下是你必须遵守的回答逻辑，需要注意的是，回答逻辑需要根据问题类型区分：
            4.1 知识点问题（概念、定义、时间复杂度、算法思想）
                回答必须包含：
                4.1.1 **引用资料出处（引用页码）**
                    例如：“根据 LectureNotes.pdf P10 的 Master Theorem…“
                4.1.2 **结构化讲解**
                    需要在回答中分以下几点讲解：
                    - 定义  
                    - 核心思想  
                    - 时间复杂度  
                    - 正确性证明关键点  
                4.1.3 **互动式引导**
                    例如：“你觉得这一类分治属于 Master Theorem 的哪一种情况？”

            4.2 向你提问题目（习题 / 作业题 / 推导题）
                回答必须包含：
                4.2.1 **引导回顾相关知识点（引用页码）**
                    例如：“这道题涉及到 LectureNotes.pdf P26 动态规划的相关内容，我们可以先考虑…“
                4.2.2 **给出题目方案或算法思路（基于资料）**
                    例如：“根据动态规划的模型，这道题可以用…“
                4.2.3 **互动式提问**
                    例如：“你觉得这里为什么能减少一次乘法？要不要我再举个例子？”

            4.3 请你出题（习题 / 作业题 / 推导题）
                回答必须包含：
                4.3.1 **引用和题目有关的资料出处（引用页码）**
                    例如：“我根据 LectureNotes.pdf P26 的动态规划模型，设计出一道关于动态规划的习题...”
                    或者：“我将仿照Solutions_for_Algorithms_AI2615.pdf P10 的第一道题，出一道同类的关于动态规划的习题...”
                4.3.2 **引导学生独立思考**
                    例如：“这道题是关于动态规划的，你可以先考虑一下…“
                4.3.2 **最后再详细讲解题目相关知识点（引用页码）**
                    例如：“这道题需要用到 LectureNotes.pdf P26 动态规划的相关内容，我们可以先考虑…“

            4.4 资料外问题（重要！）
                如果问题无法用 LectureNotes.pdf、Solutions_for_Algorithms_AI2615.pdf、Slides01-Introduction.pdf等课程资料的内容回答，你必须在回答问题之前做出如下说明：
                【说明】  
                我在 LectureNotes.pdf、Solutions_for_Algorithms_AI2615.pdf、Slides01-Introduction.pdf等课程资料中没有找到与你的问题直接对应的内容，因此下面的解释属于**课外扩展**，**本课程不要求掌握**，并且 **可能不如课程资料那样严谨**。
                （接着提供合理的扩展解释）
                【助教引导】  
                如果你感兴趣，我可以告诉你一些进一步阅读的方向，也欢迎你查查网络或教材。

        5. 风格要求：必须保持
            5.1 使用 LectureNotes.pdf、Solutions_for_Algorithms_AI2615.pdf、Slides01-Introduction.pdf等课程资料中的术语和风格（例如 DP, BFS, Dijkstra等术语）
            5.2 回答结构清晰，有层次，有标题
            5.3 每次回答结尾都要带有助教式互动问题
            5.4 所有引用必须来自 LectureNotes.pdf、Solutions_for_Algorithms_AI2615.pdf、Slides01-Introduction.pdf等课程资料（除非明确标注为“资料外内容”）

        6. 最终目标
            你是一名真正的《算法设计与分析》课程助教。  
            你通过引用课程资料、结合算法思想、辅以引导式提问，帮助学生真正理解课程内容，同时在资料之外允许适度扩展，但必须明确标注其来源和课程要求的界限。
        """

    def retrieve_context(
        self, query: str, top_k: int = TOP_K
    ) -> Tuple[str, List[Dict]]:
        """检索相关上下文
        TODO: 实现检索相关上下文
        要求：
        1. 使用向量数据库检索相关文档
        2. 格式化检索结果，构建上下文字符串
        3. 每个检索结果需要包含来源信息（文件名和页码）
        4. 返回格式化的上下文字符串和原始检索结果列表
        """
        retrieved_docs = self.vector_store.search_hybrid(query, top_k=top_k)
        #格式化检索结果，构建上下文字符串
        context_parts = []
        for idx, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            filename = metadata.get("filename", "unknown")
            page_number = metadata.get("page_number", "N/A")
            
            #格式化每个检索结果，包含来源信息
            doc_text = f"【资料片段 {idx}】\n来源：{filename} 第 {page_number} 页\n内容：\n{content}\n"
            context_parts.append(doc_text)
        
        #构建完整的上下文字符串
        context = "\n".join(context_parts) if context_parts else ""
        
        return context, retrieved_docs

    def generate_response(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict]] = None,
    ) -> str:
        """生成回答
        
        参数:
            query: 用户问题
            context: 检索到的上下文
            chat_history: 对话历史
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if chat_history:
            messages.extend(chat_history)

        """
        TODO: 实现用户提示词
        要求：
        1. 包含相关的课程内容
        2. 包含学生问题
        3. 包含来源信息（文件名和页码）
        4. 返回用户提示词
        """
        user_text = f"""请基于以下课程资料回答学生的问题。

【课程资料】
{context}

【学生问题】
{query}

请根据上述课程资料给出清晰、结构化的回答，并始终引用资料出处。如果资料中没有相关内容，请明确说明并给出适度的课外扩展解释。"""

        messages.append({"role": "user", "content": user_text})
        
        # 多模态接口示意（如需添加图片支持，可参考以下格式）：
        # content_parts = [{"type": "text", "text": user_text}]
        # content_parts.append({
        #     "type": "image_url",
        #     "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        # })
        # messages.append({"role": "user", "content": content_parts})

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.7, max_tokens=1500
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"生成回答时出错: {str(e)}"

    def generate_response_stream(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict]] = None,
    ) -> Generator[str, None, None]:
        messages = [{"role": "system", "content": self.system_prompt}]
        if chat_history:
            messages.extend(chat_history)
        user_text = f"""请基于以下课程资料回答学生的问题。

【课程资料】
{context}

【学生问题】
{query}

请根据上述课程资料给出清晰、结构化的回答，并始终引用资料出处。如果资料中没有相关内容，请明确说明并给出适度的课外扩展解释。"""
        messages.append({"role": "user", "content": user_text})
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
                stream=True,
            )
            yielded = False
            for chunk in stream:
                content_piece = None
                try:
                    choice = chunk.choices[0]
                    if hasattr(choice, "delta") and choice.delta is not None:
                        content_piece = getattr(choice.delta, "content", None)
                    if not content_piece and hasattr(choice, "message") and choice.message is not None:
                        content_piece = getattr(choice.message, "content", None)
                except Exception:
                    content_piece = None
                if content_piece:
                    yielded = True
                    yield content_piece
            if not yielded:
                text = self.generate_response(query, context, chat_history)
                yield text
        except Exception:
            text = self.generate_response(query, context, chat_history)
            yield text

    def answer_question(
        self, query: str, chat_history: Optional[List[Dict]] = None, top_k: int = TOP_K
    ) -> Dict[str, any]:
        """回答问题
        
        参数:
            query: 用户问题
            chat_history: 对话历史
            top_k: 检索文档数量
            
        返回:
            生成的回答
        """
        context, retrieved_docs = self.retrieve_context(query, top_k=top_k)

        if not context:
            context = "（未检索到特别相关的课程材料）"

        answer = self.generate_response(query, context, chat_history)

        return answer

    def answer_question_stream(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None,
        top_k: int = TOP_K,
    ) -> Generator[str, None, None]:
        context, retrieved_docs = self.retrieve_context(query, top_k=top_k)
        if not context:
            context = "（未检索到特别相关的课程材料）"
        for delta in self.generate_response_stream(query, context, chat_history):
            yield delta

    def chat(self) -> None:
        """交互式对话"""
        print("=" * 60)
        print("欢迎使用智能课程助教系统！")
        print("=" * 60)

        chat_history = []

        while True:
            try:
                query = input("\n学生: ").strip()

                if not query:
                    continue

                answer = self.answer_question(query, chat_history=chat_history)

                print(f"\n助教: {answer}")

                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": answer})

            except Exception as e:
                print(f"\n错误: {str(e)}")
