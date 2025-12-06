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

import argparse
import json

from ciso_agent.manager import CISOManager, CISOState
from ciso_agent.metrics import get_metrics_collector


def run(inputs: dict, export_metrics: bool = True):
    manager = CISOManager(
        eval_policy=False,
    )
    state = inputs
    output = manager.invoke(state, export_metrics=export_metrics)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CISO Agent - Compliance automation with LLM")
    parser.add_argument("-g", "--goal", default="", help="The compliance goal for the agent to achieve")
    parser.add_argument("-o", "--output", default="", help="The path to the output JSON file")
    parser.add_argument("-a", "--auto-approve", action="store_true", help="do nothing for now")
    parser.add_argument("-m", "--metrics", default="", help="The path to export detailed metrics JSON file")
    parser.add_argument("--no-metrics", action="store_true", help="Disable metrics collection and export")
    args = parser.parse_args()

    inputs = CISOState(goal=args.goal)
    export_metrics = not args.no_metrics
    result = run(inputs=inputs, export_metrics=export_metrics)
    result_json_str = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result_json_str)
    
    # Export detailed metrics to custom path if specified
    if args.metrics and export_metrics:
        collector = get_metrics_collector()
        collector.export_metrics(args.metrics)
