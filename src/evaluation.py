# evaluation.py
# Uses LLM-as-judge to score agent responses, and tracks tool success rates.

import datetime
import re
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import (
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
)


class AgentEvaluator:
    """Scores responses and keeps track of tool usage stats."""

    def __init__(self):
        self.evaluation_log: list[dict] = []
        self.tool_success_counts: dict[str, dict] = {}

    def evaluate_response(self, query: str, response: str,
                          context: str = "") -> dict:
        """Ask the LLM to rate a response on relevance, accuracy, etc. (1-5)."""
        eval_prompt = f"""You are evaluating a healthcare assistant's response.
Rate the response on these criteria (1-5 scale):

1. **Relevance**: Does the response address the user's question?
2. **Accuracy**: Is the medical information correct and appropriate?
3. **Helpfulness**: Is the response actionable and clear?
4. **Completeness**: Does it cover all aspects of the query?

User Query: {query}
{f"Context: {context}" if context else ""}
Assistant Response: {response}

Respond ONLY in this exact format (no extra text):
Relevance: <score>
Accuracy: <score>
Helpfulness: <score>
Completeness: <score>
Overall: <score>
Feedback: <one sentence>
"""
        try:
            llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                azure_deployment=AZURE_OPENAI_DEPLOYMENT,
                api_version=AZURE_OPENAI_API_VERSION,
                temperature=1,
            )
            result = llm.invoke([
                SystemMessage(content="You are a response quality evaluator."),
                HumanMessage(content=eval_prompt),
            ])

            scores = self._parse_eval_scores(result.content)
            eval_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "query": query[:100],
                "response_preview": response[:100],
                "scores": scores,
            }
            self.evaluation_log.append(eval_entry)
            return eval_entry

        except Exception as e:
            fallback = {
                "timestamp": datetime.datetime.now().isoformat(),
                "query": query[:100],
                "response_preview": response[:100],
                "scores": {
                    "relevance": 0, "accuracy": 0, "helpfulness": 0,
                    "completeness": 0, "overall": 0,
                    "feedback": f"Evaluation error: {str(e)}"
                },
            }
            self.evaluation_log.append(fallback)
            return fallback

    def _parse_eval_scores(self, text: str) -> dict:
        """Extract numeric scores from the LLM's evaluation output."""
        scores = {
            "relevance": 0, "accuracy": 0, "helpfulness": 0,
            "completeness": 0, "overall": 0, "feedback": "",
        }
        for line in text.strip().split("\n"):
            line_lower = line.lower().strip()
            for key in ["relevance", "accuracy", "helpfulness",
                        "completeness", "overall"]:
                if line_lower.startswith(key):
                    match = re.search(r"(\d)", line)
                    if match:
                        scores[key] = int(match.group(1))
            if line_lower.startswith("feedback"):
                scores["feedback"] = line.split(":", 1)[-1].strip()
        return scores

    def log_tool_usage(self, tool_name: str, success: bool):
        if tool_name not in self.tool_success_counts:
            self.tool_success_counts[tool_name] = {
                "success": 0, "failure": 0, "total": 0
            }
        self.tool_success_counts[tool_name]["total"] += 1
        if success:
            self.tool_success_counts[tool_name]["success"] += 1
        else:
            self.tool_success_counts[tool_name]["failure"] += 1

    def get_tool_metrics(self) -> dict:
        """Success rate per tool."""
        metrics = {}
        for tool_name, counts in self.tool_success_counts.items():
            rate = (counts["success"] / counts["total"] * 100
                    if counts["total"] > 0 else 0)
            metrics[tool_name] = {
                **counts,
                "success_rate": round(rate, 1),
            }
        return metrics

    def get_evaluation_summary(self) -> dict:
        """Aggregate averages across all evaluations."""
        if not self.evaluation_log:
            return {"total_evaluations": 0, "avg_scores": {}}

        score_keys = ["relevance", "accuracy", "helpfulness",
                      "completeness", "overall"]
        totals = {k: 0 for k in score_keys}
        count = 0

        for entry in self.evaluation_log:
            scores = entry.get("scores", {})
            if scores.get("overall", 0) > 0:
                for k in score_keys:
                    totals[k] += scores.get(k, 0)
                count += 1

        avg = {k: round(v / count, 2) if count > 0 else 0
               for k, v in totals.items()}

        return {
            "total_evaluations": len(self.evaluation_log),
            "valid_evaluations": count,
            "avg_scores": avg,
        }

    def get_recent_evaluations(self, n: int = 10) -> list[dict]:
        return self.evaluation_log[-n:]
