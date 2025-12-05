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

from typing import Any
from crewai.tools import BaseTool

from ciso_agent.metrics import get_metrics_collector


def trim_quote(val: str) -> str:
    if not isinstance(val, str):
        return val
    return val.strip().strip('"').strip("'")


class InstrumentedBaseTool(BaseTool):
    """
    Base tool class with automatic metrics instrumentation.
    
    Tracks tool round-trip time for each call.
    Subclasses should override _run_instrumented() instead of _run().
    """
    
    def _run(self, *args, **kwargs) -> Any:
        """
        Wrapper that instruments the tool execution with metrics.
        """
        collector = get_metrics_collector()
        call_id = collector.start_tool_call(self.name)
        
        try:
            result = self._run_instrumented(*args, **kwargs)
            collector.end_tool_call(call_id, success=True)
            return result
        except Exception as e:
            collector.end_tool_call(call_id, success=False, error_message=str(e))
            raise
    
    def _run_instrumented(self, *args, **kwargs) -> Any:
        """
        Override this method in subclasses instead of _run().
        This method contains the actual tool logic.
        """
        raise NotImplementedError("Subclasses must implement _run_instrumented()")
