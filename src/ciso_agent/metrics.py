# Copyright contributors to the ITBench project. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Metrics module for ITBench CISO Agent.

Provides fine-grained timing metrics for:
- Time to First Token (TTFT): Latency from request sent to first token generated
- Token Generation Speed: Pure inference speed in tokens/second
- Tool Round-Trip Time: Latency for tool execution vs LLM thinking time
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from contextlib import contextmanager
from functools import wraps


@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM call."""
    call_id: str
    model: str
    start_time: float
    end_time: float = 0.0
    time_to_first_token: float = 0.0  # TTFT in seconds
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0
    total_duration: float = 0.0  # Total wall-clock time
    generation_duration: float = 0.0  # Time spent generating (after first token)
    
    def finalize(self):
        """Calculate derived metrics after call completion."""
        if self.end_time > 0 and self.start_time > 0:
            self.total_duration = self.end_time - self.start_time
            # Generation duration excludes TTFT
            self.generation_duration = self.total_duration - self.time_to_first_token
            # Calculate tokens per second (based on generation time, not total time)
            if self.generation_duration > 0 and self.completion_tokens > 0:
                self.tokens_per_second = self.completion_tokens / self.generation_duration
            elif self.total_duration > 0 and self.completion_tokens > 0:
                # Fallback if we don't have TTFT
                self.tokens_per_second = self.completion_tokens / self.total_duration


@dataclass
class ToolCallMetrics:
    """Metrics for a single tool call."""
    call_id: str
    tool_name: str
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    success: bool = True
    error_message: str = ""
    
    def finalize(self):
        """Calculate derived metrics after call completion."""
        if self.end_time > 0 and self.start_time > 0:
            self.duration = self.end_time - self.start_time


@dataclass
class AgentTurnMetrics:
    """Metrics for a complete agent turn (LLM thinking + tool execution)."""
    turn_id: str
    start_time: float
    end_time: float = 0.0
    llm_thinking_time: float = 0.0  # Time spent in LLM calls
    tool_execution_time: float = 0.0  # Time spent executing tools
    total_duration: float = 0.0
    llm_calls: List[str] = field(default_factory=list)  # List of LLM call IDs
    tool_calls: List[str] = field(default_factory=list)  # List of tool call IDs
    
    def finalize(self):
        """Calculate derived metrics after turn completion."""
        if self.end_time > 0 and self.start_time > 0:
            self.total_duration = self.end_time - self.start_time


class MetricsCollector:
    """
    Thread-safe singleton metrics collector for ITBench CISO Agent.
    
    Collects and aggregates metrics for:
    - LLM calls (TTFT, tokens/sec)
    - Tool executions (round-trip time)
    - Agent turns (decomposed timing)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._llm_calls: Dict[str, LLMCallMetrics] = {}
        self._tool_calls: Dict[str, ToolCallMetrics] = {}
        self._agent_turns: Dict[str, AgentTurnMetrics] = {}
        self._current_turn_id: Optional[str] = None
        self._call_counter = 0
        self._data_lock = threading.Lock()
        self._session_start = time.time()
        self._initialized = True
    
    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID for a metric entry."""
        with self._data_lock:
            self._call_counter += 1
            return f"{prefix}_{self._call_counter}_{int(time.time() * 1000)}"
    
    # -------------------------------------------------------------------------
    # LLM Call Metrics
    # -------------------------------------------------------------------------
    
    def start_llm_call(self, model: str) -> str:
        """Start tracking an LLM call. Returns call_id."""
        call_id = self._generate_id("llm")
        metrics = LLMCallMetrics(
            call_id=call_id,
            model=model,
            start_time=time.time()
        )
        with self._data_lock:
            self._llm_calls[call_id] = metrics
            if self._current_turn_id and self._current_turn_id in self._agent_turns:
                self._agent_turns[self._current_turn_id].llm_calls.append(call_id)
        return call_id
    
    def record_first_token(self, call_id: str):
        """Record when the first token was received."""
        first_token_time = time.time()
        with self._data_lock:
            if call_id in self._llm_calls:
                metrics = self._llm_calls[call_id]
                metrics.time_to_first_token = first_token_time - metrics.start_time
    
    def end_llm_call(
        self,
        call_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0
    ):
        """End tracking an LLM call with token counts."""
        end_time = time.time()
        with self._data_lock:
            if call_id in self._llm_calls:
                metrics = self._llm_calls[call_id]
                metrics.end_time = end_time
                metrics.prompt_tokens = prompt_tokens
                metrics.completion_tokens = completion_tokens
                metrics.total_tokens = total_tokens or (prompt_tokens + completion_tokens)
                metrics.finalize()
                
                # Update turn metrics
                if self._current_turn_id and self._current_turn_id in self._agent_turns:
                    turn = self._agent_turns[self._current_turn_id]
                    turn.llm_thinking_time += metrics.total_duration
    
    # -------------------------------------------------------------------------
    # Tool Call Metrics
    # -------------------------------------------------------------------------
    
    def start_tool_call(self, tool_name: str) -> str:
        """Start tracking a tool call. Returns call_id."""
        call_id = self._generate_id("tool")
        metrics = ToolCallMetrics(
            call_id=call_id,
            tool_name=tool_name,
            start_time=time.time()
        )
        with self._data_lock:
            self._tool_calls[call_id] = metrics
            if self._current_turn_id and self._current_turn_id in self._agent_turns:
                self._agent_turns[self._current_turn_id].tool_calls.append(call_id)
        return call_id
    
    def end_tool_call(self, call_id: str, success: bool = True, error_message: str = ""):
        """End tracking a tool call."""
        end_time = time.time()
        with self._data_lock:
            if call_id in self._tool_calls:
                metrics = self._tool_calls[call_id]
                metrics.end_time = end_time
                metrics.success = success
                metrics.error_message = error_message
                metrics.finalize()
                
                # Update turn metrics
                if self._current_turn_id and self._current_turn_id in self._agent_turns:
                    turn = self._agent_turns[self._current_turn_id]
                    turn.tool_execution_time += metrics.duration
    
    # -------------------------------------------------------------------------
    # Agent Turn Metrics
    # -------------------------------------------------------------------------
    
    def start_agent_turn(self) -> str:
        """Start tracking an agent turn. Returns turn_id."""
        turn_id = self._generate_id("turn")
        metrics = AgentTurnMetrics(
            turn_id=turn_id,
            start_time=time.time()
        )
        with self._data_lock:
            self._agent_turns[turn_id] = metrics
            self._current_turn_id = turn_id
        return turn_id
    
    def end_agent_turn(self, turn_id: str):
        """End tracking an agent turn."""
        end_time = time.time()
        with self._data_lock:
            if turn_id in self._agent_turns:
                metrics = self._agent_turns[turn_id]
                metrics.end_time = end_time
                metrics.finalize()
            if self._current_turn_id == turn_id:
                self._current_turn_id = None
    
    # -------------------------------------------------------------------------
    # Context Managers for Convenience
    # -------------------------------------------------------------------------
    
    @contextmanager
    def track_llm_call(self, model: str):
        """Context manager for tracking LLM calls."""
        call_id = self.start_llm_call(model)
        try:
            yield call_id
        finally:
            # Note: Caller should call end_llm_call with token counts
            pass
    
    @contextmanager
    def track_tool_call(self, tool_name: str):
        """Context manager for tracking tool calls."""
        call_id = self.start_tool_call(tool_name)
        try:
            yield call_id
            self.end_tool_call(call_id, success=True)
        except Exception as e:
            self.end_tool_call(call_id, success=False, error_message=str(e))
            raise
    
    @contextmanager
    def track_agent_turn(self):
        """Context manager for tracking agent turns."""
        turn_id = self.start_agent_turn()
        try:
            yield turn_id
        finally:
            self.end_agent_turn(turn_id)
    
    # -------------------------------------------------------------------------
    # Aggregation & Reporting
    # -------------------------------------------------------------------------
    
    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated metrics summary."""
        with self._data_lock:
            llm_metrics = list(self._llm_calls.values())
            tool_metrics = list(self._tool_calls.values())
            turn_metrics = list(self._agent_turns.values())
        
        # Aggregate LLM metrics
        total_llm_calls = len(llm_metrics)
        total_tokens = sum(m.total_tokens for m in llm_metrics)
        total_prompt_tokens = sum(m.prompt_tokens for m in llm_metrics)
        total_completion_tokens = sum(m.completion_tokens for m in llm_metrics)
        
        ttft_values = [m.time_to_first_token for m in llm_metrics if m.time_to_first_token > 0]
        tps_values = [m.tokens_per_second for m in llm_metrics if m.tokens_per_second > 0]
        llm_durations = [m.total_duration for m in llm_metrics if m.total_duration > 0]
        
        # Aggregate tool metrics
        total_tool_calls = len(tool_metrics)
        successful_tool_calls = sum(1 for m in tool_metrics if m.success)
        tool_durations = [m.duration for m in tool_metrics if m.duration > 0]
        
        # Tool breakdown by name
        tool_breakdown = {}
        for m in tool_metrics:
            if m.tool_name not in tool_breakdown:
                tool_breakdown[m.tool_name] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "avg_duration": 0.0,
                    "success_rate": 0.0,
                    "successes": 0
                }
            tool_breakdown[m.tool_name]["count"] += 1
            tool_breakdown[m.tool_name]["total_duration"] += m.duration
            if m.success:
                tool_breakdown[m.tool_name]["successes"] += 1
        
        for name, stats in tool_breakdown.items():
            if stats["count"] > 0:
                stats["avg_duration"] = stats["total_duration"] / stats["count"]
                stats["success_rate"] = stats["successes"] / stats["count"]
        
        # Calculate total time decomposition
        total_llm_time = sum(llm_durations) if llm_durations else 0
        total_tool_time = sum(tool_durations) if tool_durations else 0
        session_duration = time.time() - self._session_start
        
        return {
            "session": {
                "duration_seconds": session_duration,
                "start_time": datetime.fromtimestamp(self._session_start).isoformat(),
            },
            "llm": {
                "total_calls": total_llm_calls,
                "total_tokens": total_tokens,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_time_seconds": total_llm_time,
                "avg_ttft_seconds": sum(ttft_values) / len(ttft_values) if ttft_values else 0,
                "min_ttft_seconds": min(ttft_values) if ttft_values else 0,
                "max_ttft_seconds": max(ttft_values) if ttft_values else 0,
                "avg_tokens_per_second": sum(tps_values) / len(tps_values) if tps_values else 0,
                "min_tokens_per_second": min(tps_values) if tps_values else 0,
                "max_tokens_per_second": max(tps_values) if tps_values else 0,
                "avg_call_duration_seconds": sum(llm_durations) / len(llm_durations) if llm_durations else 0,
            },
            "tools": {
                "total_calls": total_tool_calls,
                "successful_calls": successful_tool_calls,
                "success_rate": successful_tool_calls / total_tool_calls if total_tool_calls > 0 else 0,
                "total_time_seconds": total_tool_time,
                "avg_duration_seconds": sum(tool_durations) / len(tool_durations) if tool_durations else 0,
                "min_duration_seconds": min(tool_durations) if tool_durations else 0,
                "max_duration_seconds": max(tool_durations) if tool_durations else 0,
                "breakdown_by_tool": tool_breakdown,
            },
            "time_decomposition": {
                "llm_thinking_seconds": total_llm_time,
                "tool_execution_seconds": total_tool_time,
                "other_overhead_seconds": session_duration - total_llm_time - total_tool_time,
                "llm_percentage": (total_llm_time / session_duration * 100) if session_duration > 0 else 0,
                "tool_percentage": (total_tool_time / session_duration * 100) if session_duration > 0 else 0,
            },
            "usage": {
                # Key metric requested by user
                "tokens_per_second": sum(tps_values) / len(tps_values) if tps_values else 0,
            }
        }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed per-call metrics."""
        with self._data_lock:
            return {
                "llm_calls": [asdict(m) for m in self._llm_calls.values()],
                "tool_calls": [asdict(m) for m in self._tool_calls.values()],
                "agent_turns": [asdict(m) for m in self._agent_turns.values()],
            }
    
    def export_metrics(self, filepath: str):
        """Export all metrics to a JSON file."""
        metrics = {
            "summary": self.get_summary(),
            "detailed": self.get_detailed_metrics(),
            "exported_at": datetime.now().isoformat(),
        }
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Metrics] Exported to {filepath}")
    
    def print_summary(self):
        """Print a formatted summary of metrics."""
        summary = self.get_summary()
        
        print("\n" + "=" * 80)
        print(" ITBench CISO Agent - Performance Metrics Summary")
        print("=" * 80)
        
        print(f"\n📊 Session Duration: {summary['session']['duration_seconds']:.2f}s")
        
        print("\n🤖 LLM Performance:")
        llm = summary["llm"]
        print(f"   Total Calls: {llm['total_calls']}")
        print(f"   Total Tokens: {llm['total_tokens']} (prompt: {llm['prompt_tokens']}, completion: {llm['completion_tokens']})")
        print(f"   ⏱️  Time to First Token (TTFT):")
        print(f"       Avg: {llm['avg_ttft_seconds']*1000:.2f}ms | Min: {llm['min_ttft_seconds']*1000:.2f}ms | Max: {llm['max_ttft_seconds']*1000:.2f}ms")
        print(f"   🚀 Token Generation Speed:")
        print(f"       Avg: {llm['avg_tokens_per_second']:.2f} tokens/s | Min: {llm['min_tokens_per_second']:.2f} | Max: {llm['max_tokens_per_second']:.2f}")
        print(f"   Total LLM Time: {llm['total_time_seconds']:.2f}s")
        
        print("\n🔧 Tool Performance:")
        tools = summary["tools"]
        print(f"   Total Calls: {tools['total_calls']} (success rate: {tools['success_rate']*100:.1f}%)")
        print(f"   ⏱️  Round-Trip Time:")
        print(f"       Avg: {tools['avg_duration_seconds']*1000:.2f}ms | Min: {tools['min_duration_seconds']*1000:.2f}ms | Max: {tools['max_duration_seconds']*1000:.2f}ms")
        print(f"   Total Tool Time: {tools['total_time_seconds']:.2f}s")
        
        if tools["breakdown_by_tool"]:
            print("\n   Tool Breakdown:")
            for name, stats in tools["breakdown_by_tool"].items():
                print(f"     - {name}: {stats['count']} calls, avg {stats['avg_duration']*1000:.2f}ms")
        
        print("\n⏱️  Time Decomposition:")
        decomp = summary["time_decomposition"]
        print(f"   LLM Thinking: {decomp['llm_thinking_seconds']:.2f}s ({decomp['llm_percentage']:.1f}%)")
        print(f"   Tool Execution: {decomp['tool_execution_seconds']:.2f}s ({decomp['tool_percentage']:.1f}%)")
        print(f"   Other Overhead: {decomp['other_overhead_seconds']:.2f}s")
        
        print("\n📈 Key Metric: llm.usage.tokens_per_second =", f"{summary['usage']['tokens_per_second']:.2f}")
        print("=" * 80 + "\n")
    
    def reset(self):
        """Reset all collected metrics."""
        with self._data_lock:
            self._llm_calls.clear()
            self._tool_calls.clear()
            self._agent_turns.clear()
            self._current_turn_id = None
            self._call_counter = 0
            self._session_start = time.time()


# Global singleton instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def instrument_tool(tool_name: str):
    """Decorator to instrument a tool function with metrics tracking."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            call_id = collector.start_tool_call(tool_name)
            try:
                result = func(*args, **kwargs)
                collector.end_tool_call(call_id, success=True)
                return result
            except Exception as e:
                collector.end_tool_call(call_id, success=False, error_message=str(e))
                raise
        return wrapper
    return decorator

