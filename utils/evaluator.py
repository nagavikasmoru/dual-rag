"""
RAG Evaluation Engine
Metrics: Faithfulness, Answer Relevance, Context Relevance
Uses Groq LLM as judge (no OpenAI required)
"""

import time
import json
import re
from typing import Dict
from groq import Groq


class RAGEvaluator:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.model = "llama-3.3-70b-versatile"

    def _ask_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    def _parse(self, text: str) -> Dict:
        """Extract score and reason from LLM JSON response."""
        try:
            clean = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
            return {
                "score": round(float(data.get("score", 0.5)), 2),
                "reason": data.get("reason", "")
            }
        except Exception:
            nums = re.findall(r'0\.\d+|1\.0|[01]', text)
            return {
                "score": round(float(nums[0]), 2) if nums else 0.5,
                "reason": text[:120]
            }

    def faithfulness(self, answer: str, context: str) -> Dict:
        """Is every claim in the answer supported by the context?"""
        prompt = f"""You are a RAG evaluator. Check if the answer is fully supported by the context.

Context: {context[:1200]}

Answer: {answer}

Score 0.0 to 1.0:
1.0 = every claim supported by context
0.5 = some claims supported
0.0 = answer contradicts or ignores context

Respond ONLY with JSON: {{"score": <float>, "reason": "<one sentence>"}}"""
        return self._parse(self._ask_llm(prompt))

    def answer_relevance(self, question: str, answer: str) -> Dict:
        """Does the answer directly address the question?"""
        prompt = f"""You are a RAG evaluator. Check if the answer addresses the question.

Question: {question}
Answer: {answer}

Score 0.0 to 1.0:
1.0 = directly and completely answers the question
0.5 = partially answers
0.0 = irrelevant or says only "I don't know"

Respond ONLY with JSON: {{"score": <float>, "reason": "<one sentence>"}}"""
        return self._parse(self._ask_llm(prompt))

    def context_relevance(self, question: str, context: str) -> Dict:
        """Did the retriever fetch chunks relevant to the question?"""
        prompt = f"""You are a RAG evaluator. Check if the retrieved context is useful for answering the question.

Question: {question}
Context: {context[:1200]}

Score 0.0 to 1.0:
1.0 = context contains exactly what is needed
0.5 = context partially relevant
0.0 = context has nothing useful

Respond ONLY with JSON: {{"score": <float>, "reason": "<one sentence>"}}"""
        return self._parse(self._ask_llm(prompt))

    def evaluate_all(self, question: str, answer: str, context: str) -> Dict:
        """Run all 3 metrics and return combined results."""
        t0 = time.time()
        faith = self.faithfulness(answer, context)
        rel   = self.answer_relevance(question, answer)
        ctx   = self.context_relevance(question, context)
        overall = round((faith["score"] + rel["score"] + ctx["score"]) / 3, 2)
        return {
            "faithfulness":      faith,
            "answer_relevance":  rel,
            "context_relevance": ctx,
            "overall":           overall,
            "eval_time":         round(time.time() - t0, 2),
        }
