# agent.py
# LangGraph-based agentic workflow: planner node decides what to do,
# tool node executes, then loops back to planner until done.

import datetime
from typing import Annotated, TypedDict
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from src.config import (
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
    LLM_TEMPERATURE,
)
from src.tools import ALL_TOOLS
from src.prompts import SYSTEM_PROMPT


# state that flows through the graph

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_context: str
    plan: str
    tool_log: list[dict]


# conversation memory - keeps track of recent exchanges and patient context

class ConversationMemory:
    """Stores recent chat history + any patient info we've seen so far."""

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.conversations: list[dict] = []
        self.patient_context: dict = {}

    def add_interaction(self, role: str, content: str) -> None:
        self.conversations.append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat(),
        })
        if len(self.conversations) > self.max_history:
            self.conversations = self.conversations[-self.max_history:]

    def update_patient_context(self, key: str, value: str) -> None:
        self.patient_context[key] = value

    def get_context_string(self) -> str:
        if not self.patient_context:
            return "No prior patient context."
        return "\n".join(
            f"  {k}: {v}" for k, v in self.patient_context.items()
        )

    def get_recent_history(self, n: int = 5) -> list[dict]:
        return self.conversations[-n:]

    def clear(self) -> None:
        self.conversations.clear()
        self.patient_context.clear()


# llm setup

def get_llm():
    """Create an Azure OpenAI LLM instance with tools bound."""
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=1,
    )
    return llm.bind_tools(ALL_TOOLS)


# graph nodes

def planner_node(state: AgentState) -> dict:
    """Main planner - interprets the query and picks tools to call."""
    llm = get_llm()
    system_msg = SystemMessage(content=SYSTEM_PROMPT)

    # add patient context to system message if we have any
    context = state.get("patient_context", "")
    if context:
        system_msg = SystemMessage(
            content=SYSTEM_PROMPT + f"\n\nCurrent Patient Context:\n{context}"
        )

    messages = [system_msg] + state["messages"]
    response = llm.invoke(messages)

    return {"messages": [response]}


def should_use_tools(state: AgentState) -> str:
    """Route to tool node if the LLM wants to call tools, otherwise end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# build the compiled graph

def build_agent_graph():
    """Assemble and compile the planner -> tools -> planner loop."""
    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", should_use_tools)
    graph.add_edge("tools", "planner")  # loop back after tool execution

    return graph.compile()


# high-level agent wrapper

class HealthcareAgent:
    """Wraps the graph with memory and a simple chat() interface."""

    def __init__(self):
        self.graph = build_agent_graph()
        self.memory = ConversationMemory()
        self.tool_log: list[dict] = []

    def chat(self, user_input: str) -> str:
        """Send a message and get the agent's reply."""
        self.memory.add_interaction("user", user_input)

        # Build full conversation history so the LLM sees prior turns
        history_messages = []
        for turn in self.memory.get_recent_history(n=20):
            if turn["role"] == "user":
                history_messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                history_messages.append(AIMessage(content=turn["content"]))

        # If history includes the current message as last, use as-is;
        # otherwise append it (shouldn't happen, but safeguard)
        if not history_messages or not isinstance(history_messages[-1], HumanMessage):
            history_messages.append(HumanMessage(content=user_input))

        initial_state: AgentState = {
            "messages": history_messages,
            "patient_context": self.memory.get_context_string(),
            "plan": "",
            "tool_log": [],
        }

        # Run the graph
        result = self.graph.invoke(initial_state)

        # grab the last AI message that has actual text content
        ai_messages = [
            m for m in result["messages"]
            if isinstance(m, AIMessage) and m.content
            and not getattr(m, "tool_calls", None)
        ]

        response = ai_messages[-1].content if ai_messages else "Sorry, I couldn't process that. Could you try rephrasing?"

        self.memory.add_interaction("assistant", response)

        # log which tools the LLM called
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    self.tool_log.append({
                        "tool": tc["name"],
                        "args": tc["args"],
                        "timestamp": datetime.datetime.now().isoformat(),
                    })

        # remember patient names if they came up in tool results
        for msg in result["messages"]:
            if hasattr(msg, "content") and msg.content:
                content_lower = msg.content.lower()
                if "patient found" in content_lower or "patient:" in content_lower:
                    self.memory.update_patient_context(
                        "last_referenced_patient",
                        msg.content[:200]
                    )

        return response

    def get_tool_log(self):
        return self.tool_log

    def get_memory_trace(self):
        """Return current memory state (useful for the UI)."""
        return {
            "patient_context": self.memory.patient_context,
            "conversation_count": len(self.memory.conversations),
            "recent_history": self.memory.get_recent_history(),
        }

    def reset(self):
        self.memory.clear()
        self.tool_log.clear()
