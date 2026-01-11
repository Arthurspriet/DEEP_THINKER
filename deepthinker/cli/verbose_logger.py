"""
Verbose Logger for DeepThinker.

Provides comprehensive, structured logging for council execution,
context flow tracing, iteration progress, and state manager snapshots.

Uses Rich for beautiful terminal output with plain text fallback.
"""

import dataclasses
import inspect
import json
import time
from dataclasses import is_dataclass, fields as dataclass_fields
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, get_type_hints

# Try to import Rich for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    from rich.syntax import Syntax
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# SSE event publishing integration
try:
    from api.sse import sse_manager
    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False
    sse_manager = None


def _publish_sse_event(coro):
    """
    Helper to publish SSE events from sync code.
    Safely schedules the coroutine if an event loop is running.
    """
    if not SSE_AVAILABLE or sse_manager is None:
        coro.close()  # Clean up the coroutine
        return
    try:
        import asyncio
        asyncio.get_running_loop()
        asyncio.create_task(coro)
    except RuntimeError:
        # No event loop running - close coroutine to avoid warning
        coro.close()


class VerboseLogger:
    """
    Comprehensive verbose logger for DeepThinker workflow execution.
    
    Provides real-time visibility into:
    - Council requirements and execution
    - Context flow between components
    - Phase artifacts and iteration progress
    - State manager snapshots
    """
    
    # Truncation limits for different modes
    TRUNCATE_LIMITS = {
        "system_prompt": 500,
        "council_output": 1500,
        "code_preview": 300,
        "context_value": 500,
        "artifact_value": 1500,
    }
    
    def __init__(self, enabled: bool = False, full_mode: bool = False):
        """
        Initialize the verbose logger.
        
        Args:
            enabled: Whether verbose logging is enabled
            full_mode: If True, disable truncation (--verbose-full)
        """
        self.enabled = enabled
        self.full_mode = full_mode
        self._console = Console() if RICH_AVAILABLE else None
        self._context_stack: List[Dict[str, Any]] = []
        self._flow_history: List[Dict[str, Any]] = []
        self._start_times: Dict[str, float] = {}
    
    def configure(self, enabled: bool = False, full_mode: bool = False) -> None:
        """Configure the logger settings."""
        self.enabled = enabled
        self.full_mode = full_mode
    
    def _truncate(self, text: str, limit_key: str) -> str:
        """Truncate text based on mode and limit key."""
        if self.full_mode:
            return text
        
        limit = self.TRUNCATE_LIMITS.get(limit_key, 500)
        if len(text) <= limit:
            return text
        return text[:limit] + f"... [truncated, {len(text) - limit} more chars]"
    
    def _print(self, content: Any) -> None:
        """Print content using Rich if available, else plain print."""
        if not self.enabled:
            return
        
        if self._console and RICH_AVAILABLE:
            self._console.print(content)
        else:
            print(str(content))
    
    def _print_divider(self, char: str = "─", width: int = 60) -> None:
        """Print a divider line."""
        if not self.enabled:
            return
        print(char * width)
    
    # =========================================================================
    # COUNCIL INTROSPECTION
    # =========================================================================
    
    def log_council_requirements(self, council_class: Type) -> None:
        """
        Log council requirements by introspecting the class.
        
        Extracts:
        - System prompt
        - Context dataclass schema
        - Expected inputs/outputs
        
        Args:
            council_class: The council class to introspect
        """
        if not self.enabled:
            return
        
        council_name = council_class.__name__
        
        # Extract system prompt
        system_prompt = None
        if hasattr(council_class, '_system_prompt'):
            system_prompt = getattr(council_class, '_system_prompt', None)
        
        # Try to find context dataclass from build_prompt signature
        context_schema = self._extract_context_schema(council_class)
        
        # Get expected outputs from postprocess return type
        output_info = self._extract_output_info(council_class)
        
        if RICH_AVAILABLE and self._console:
            # Build Rich panel
            content = Text()
            content.append(f"Council: ", style="bold cyan")
            content.append(f"{council_name}\n\n")
            
            if context_schema:
                content.append("Context Schema:\n", style="bold yellow")
                for field_name, field_info in context_schema.items():
                    content.append(f"  • {field_name}: ", style="green")
                    content.append(f"{field_info}\n")
                content.append("\n")
            
            if output_info:
                content.append("Expected Output: ", style="bold yellow")
                content.append(f"{output_info}\n\n")
            
            if system_prompt:
                content.append("System Prompt:\n", style="bold yellow")
                truncated = self._truncate(system_prompt, "system_prompt")
                content.append(f"{truncated}\n", style="dim")
            
            panel = Panel(
                content,
                title="🔍 Council Requirements",
                border_style="blue",
                box=box.ROUNDED
            )
            self._console.print(panel)
        else:
            # Plain text output
            self._print_divider()
            print(f"🔍 Council Requirements: {council_name}")
            self._print_divider()
            
            if context_schema:
                print("Context Schema:")
                for field_name, field_info in context_schema.items():
                    print(f"  • {field_name}: {field_info}")
                print()
            
            if output_info:
                print(f"Expected Output: {output_info}")
                print()
            
            if system_prompt:
                print("System Prompt:")
                print(self._truncate(system_prompt, "system_prompt"))
            
            self._print_divider()
    
    def _extract_context_schema(self, council_class: Type) -> Dict[str, str]:
        """Extract context schema from council's build_prompt method."""
        schema = {}
        
        try:
            # Get build_prompt signature
            if hasattr(council_class, 'build_prompt'):
                sig = inspect.signature(council_class.build_prompt)
                hints = get_type_hints(council_class.build_prompt) if hasattr(council_class, 'build_prompt') else {}
                
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    
                    # Get type hint
                    type_hint = hints.get(param_name, param.annotation)
                    type_str = self._format_type_hint(type_hint)
                    
                    # Check if it's a dataclass and extract fields
                    if type_hint != inspect.Parameter.empty and is_dataclass(type_hint):
                        schema[param_name] = f"{type_str} (dataclass)"
                        # Add dataclass fields
                        for field in dataclass_fields(type_hint):
                            field_type = self._format_type_hint(field.type)
                            default_info = ""
                            if field.default is not dataclasses.MISSING:
                                default_info = f" = {field.default}"
                            elif field.default_factory is not dataclasses.MISSING:
                                default_info = " = <factory>"
                            schema[f"  └─ {field.name}"] = f"{field_type}{default_info}"
                    else:
                        # Add default value info
                        default_info = ""
                        if param.default is not inspect.Parameter.empty:
                            default_info = f" = {param.default}"
                        schema[param_name] = f"{type_str}{default_info}"
        except Exception:
            pass
        
        return schema
    
    def _extract_output_info(self, council_class: Type) -> Optional[str]:
        """Extract output type from postprocess method."""
        try:
            if hasattr(council_class, 'postprocess'):
                hints = get_type_hints(council_class.postprocess)
                if 'return' in hints:
                    return self._format_type_hint(hints['return'])
        except Exception:
            pass
        return None
    
    def _format_type_hint(self, hint: Any) -> str:
        """Format a type hint for display."""
        if hint is inspect.Parameter.empty or hint is None:
            return "Any"
        
        if hasattr(hint, '__name__'):
            return hint.__name__
        
        # Handle Optional, List, Dict, etc.
        origin = getattr(hint, '__origin__', None)
        args = getattr(hint, '__args__', ())
        
        if origin is not None:
            origin_name = getattr(origin, '__name__', str(origin))
            if args:
                args_str = ", ".join(self._format_type_hint(a) for a in args)
                return f"{origin_name}[{args_str}]"
            return origin_name
        
        return str(hint)
    
    def log_council_activated(
        self,
        council_name: str,
        context: Any,
        models: Optional[List[str]] = None
    ) -> None:
        """
        Log when a council is activated with its input context.
        
        Args:
            council_name: Name of the council
            context: The context object passed to the council
            models: List of models used by the council
        """
        if not self.enabled:
            return
        
        self._start_times[council_name] = time.time()
        
        # Extract context data
        context_data = self._extract_context_data(context)
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append(f"🔍 Council Activated: ", style="bold")
            content.append(f"{council_name}\n\n", style="bold cyan")
            
            if models:
                content.append("Models: ", style="yellow")
                content.append(f"{', '.join(models)}\n\n")
            
            content.append("📥 Received Context:\n", style="bold green")
            
            panel = Panel(
                content,
                border_style="blue",
                box=box.ROUNDED
            )
            self._console.print(panel)
            
            # Print context as JSON
            if context_data:
                try:
                    json_str = json.dumps(context_data, indent=2, default=str)
                    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
                    self._console.print(syntax)
                except Exception:
                    self._console.print(str(context_data))
        else:
            self._print_divider()
            print(f"🔍 Council Activated: {council_name}")
            if models:
                print(f"   Models: {', '.join(models)}")
            print()
            print("📥 Received Context:")
            if context_data:
                try:
                    print(json.dumps(context_data, indent=2, default=str))
                except Exception:
                    print(str(context_data))
            self._print_divider()
        
        # Track for flow analysis
        self._context_stack.append({
            "council": council_name,
            "input": context_data,
            "timestamp": datetime.now().isoformat()
        })
    
    def _extract_context_data(self, context: Any) -> Dict[str, Any]:
        """Extract context data for display."""
        if context is None:
            return {}
        
        if is_dataclass(context):
            data = {}
            for field in dataclass_fields(context):
                value = getattr(context, field.name)
                if isinstance(value, str) and len(value) > self.TRUNCATE_LIMITS["context_value"]:
                    value = self._truncate(value, "context_value")
                data[field.name] = value
            return data
        
        if isinstance(context, dict):
            return {
                k: self._truncate(str(v), "context_value") if isinstance(v, str) and len(str(v)) > self.TRUNCATE_LIMITS["context_value"]
                else v
                for k, v in context.items()
            }
        
        return {"value": str(context)}
    
    # =========================================================================
    # CONTEXT FLOW TRACING
    # =========================================================================
    
    def log_context_received(self, label: str, context_obj: Any) -> None:
        """
        Log context received at a specific point.
        
        Args:
            label: Label for this context point
            context_obj: The context object
        """
        if not self.enabled:
            return
        
        context_data = self._extract_context_data(context_obj)
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append(f"📥 {label}:\n", style="bold green")
            
            self._console.print(content)
            
            if context_data:
                try:
                    json_str = json.dumps(context_data, indent=2, default=str)
                    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
                    self._console.print(syntax)
                except Exception:
                    self._console.print(str(context_data))
        else:
            print(f"📥 {label}:")
            if context_data:
                try:
                    print(json.dumps(context_data, indent=2, default=str))
                except Exception:
                    print(str(context_data))
    
    def log_context_output(
        self,
        label: str,
        data: Any,
        truncate: bool = True
    ) -> None:
        """
        Log context/output from a component.
        
        Args:
            label: Label for this output
            data: The output data
            truncate: Whether to truncate long values
        """
        if not self.enabled:
            return
        
        if RICH_AVAILABLE and self._console:
            suffix = " (truncated)" if truncate and not self.full_mode else ""
            content = Text()
            content.append(f"📤 {label}{suffix}:\n", style="bold magenta")
            
            self._console.print(content)
            
            # Format output based on type
            output_str = self._format_output(data, truncate)
            
            if isinstance(data, dict) or is_dataclass(data):
                try:
                    if is_dataclass(data):
                        data = dataclasses.asdict(data)
                    json_str = json.dumps(data, indent=2, default=str)
                    if truncate and not self.full_mode:
                        json_str = self._truncate(json_str, "council_output")
                    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
                    self._console.print(syntax)
                except Exception:
                    self._console.print(output_str)
            else:
                self._console.print(output_str)
        else:
            suffix = " (truncated)" if truncate and not self.full_mode else ""
            print(f"📤 {label}{suffix}:")
            output_str = self._format_output(data, truncate)
            print(output_str)
    
    def _format_output(self, data: Any, truncate: bool = True) -> str:
        """Format output data for display."""
        if data is None:
            return "None"
        
        if is_dataclass(data):
            try:
                data = dataclasses.asdict(data)
            except Exception:
                pass
        
        if isinstance(data, dict):
            try:
                json_str = json.dumps(data, indent=2, default=str)
                if truncate and not self.full_mode:
                    return self._truncate(json_str, "council_output")
                return json_str
            except Exception:
                pass
        
        result = str(data)
        if truncate and not self.full_mode:
            return self._truncate(result, "council_output")
        return result
    
    def log_context_flow(
        self,
        from_step: str,
        to_step: str,
        payload: Dict[str, Any]
    ) -> None:
        """
        Log context flow between two steps.
        
        Args:
            from_step: Source step name
            to_step: Destination step name
            payload: Data being passed
        """
        if not self.enabled:
            return
        
        # Track flow
        self._flow_history.append({
            "from": from_step,
            "to": to_step,
            "payload_keys": list(payload.keys()) if isinstance(payload, dict) else [],
            "timestamp": datetime.now().isoformat()
        })
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("➡️  Context Flow: ", style="bold")
            content.append(f"{from_step}", style="cyan")
            content.append(" → ", style="white")
            content.append(f"{to_step}\n", style="green")
            
            if isinstance(payload, dict):
                content.append("   Fields: ", style="dim")
                content.append(", ".join(payload.keys()), style="yellow")
            
            self._console.print(content)
        else:
            print(f"➡️  Context Flow: {from_step} → {to_step}")
            if isinstance(payload, dict):
                print(f"   Fields: {', '.join(payload.keys())}")
    
    def log_flow_summary(self) -> None:
        """Log a summary of all context flows in the session."""
        if not self.enabled or not self._flow_history:
            return
        
        if RICH_AVAILABLE and self._console:
            tree = Tree("📊 Context Flow Summary", style="bold")
            
            for flow in self._flow_history:
                branch = tree.add(f"{flow['from']} → {flow['to']}")
                if flow['payload_keys']:
                    branch.add(f"Fields: {', '.join(flow['payload_keys'])}")
            
            self._console.print(tree)
        else:
            print("\n📊 Context Flow Summary")
            self._print_divider("=")
            for flow in self._flow_history:
                print(f"  {flow['from']} → {flow['to']}")
                if flow['payload_keys']:
                    print(f"    Fields: {', '.join(flow['payload_keys'])}")
    
    # =========================================================================
    # PHASE AND ARTIFACT LOGGING
    # =========================================================================
    
    def log_phase_start(self, phase: Any) -> None:
        """
        Log the start of a mission phase.
        
        Args:
            phase: The MissionPhase object
        """
        if not self.enabled:
            return
        
        phase_name = getattr(phase, 'name', str(phase))
        description = getattr(phase, 'description', '')
        
        self._start_times[f"phase_{phase_name}"] = time.time()
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("▶️  Phase Started: ", style="bold")
            content.append(f"{phase_name}\n", style="bold cyan")
            if description:
                content.append(f"   {description}", style="dim")
            
            panel = Panel(content, border_style="green", box=box.ROUNDED)
            self._console.print(panel)
        else:
            self._print_divider("═", 60)
            print(f"▶️  Phase Started: {phase_name}")
            if description:
                print(f"   {description}")
            self._print_divider("═", 60)
    
    def log_phase_artifacts(self, phase: Any) -> None:
        """
        Log artifacts from a completed phase.
        
        Args:
            phase: The MissionPhase object with artifacts
        """
        if not self.enabled:
            return
        
        phase_name = getattr(phase, 'name', str(phase))
        artifacts = getattr(phase, 'artifacts', {})
        
        # Calculate duration if available
        duration = None
        start_key = f"phase_{phase_name}"
        if start_key in self._start_times:
            duration = time.time() - self._start_times[start_key]
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("✅ Phase Completed: ", style="bold")
            content.append(f"{phase_name}", style="bold green")
            if duration:
                content.append(f" ({duration:.1f}s)", style="dim")
            content.append("\n\n")
            
            if artifacts:
                content.append("📦 Artifacts:\n", style="bold yellow")
                for key, value in artifacts.items():
                    content.append(f"  • {key}: ", style="cyan")
                    truncated = self._truncate(str(value), "artifact_value")
                    content.append(f"{truncated}\n", style="dim")
            else:
                content.append("   No artifacts", style="dim")
            
            panel = Panel(content, border_style="green", box=box.ROUNDED)
            self._console.print(panel)
        else:
            print(f"✅ Phase Completed: {phase_name}", end="")
            if duration:
                print(f" ({duration:.1f}s)")
            else:
                print()
            
            if artifacts:
                print("📦 Artifacts:")
                for key, value in artifacts.items():
                    truncated = self._truncate(str(value), "artifact_value")
                    print(f"  • {key}: {truncated}")
    
    def log_mission_artifact_summary(self, phases: List[Any]) -> None:
        """
        Log a summary of all mission phase artifacts.
        
        Args:
            phases: List of MissionPhase objects
        """
        if not self.enabled:
            return
        
        if RICH_AVAILABLE and self._console:
            self._console.print()
            self._console.print("═" * 60, style="bold")
            self._console.print("📋 Mission Phase Artifact Summary", style="bold cyan")
            self._console.print("═" * 60, style="bold")
            
            for phase in phases:
                phase_name = getattr(phase, 'name', str(phase))
                status = getattr(phase, 'status', 'unknown')
                artifacts = getattr(phase, 'artifacts', {})
                
                # Status indicator
                status_icon = "✅" if status == "completed" else "⏳" if status == "running" else "⏸️"
                
                content = Text()
                content.append(f"\n{status_icon} Phase: ", style="bold")
                content.append(f"{phase_name}\n", style="cyan")
                
                if artifacts:
                    for key, value in artifacts.items():
                        truncated = self._truncate(str(value), "artifact_value")
                        content.append(f"   • {key}: ", style="yellow")
                        content.append(f"{truncated}\n", style="dim")
                else:
                    content.append("   (no artifacts)\n", style="dim")
                
                self._console.print(content)
        else:
            print()
            print("═" * 60)
            print("📋 Mission Phase Artifact Summary")
            print("═" * 60)
            
            for phase in phases:
                phase_name = getattr(phase, 'name', str(phase))
                status = getattr(phase, 'status', 'unknown')
                artifacts = getattr(phase, 'artifacts', {})
                
                status_icon = "✅" if status == "completed" else "⏳" if status == "running" else "⏸️"
                print(f"\n{status_icon} Phase: {phase_name}")
                
                if artifacts:
                    for key, value in artifacts.items():
                        truncated = self._truncate(str(value), "artifact_value")
                        print(f"   • {key}: {truncated}")
                else:
                    print("   (no artifacts)")
    
    # =========================================================================
    # ITERATION TRACKING
    # =========================================================================
    
    def log_iteration_summary(self, iteration_info: Dict[str, Any]) -> None:
        """
        Log summary of a single iteration.
        
        Args:
            iteration_info: Dictionary with iteration details
        """
        if not self.enabled:
            return
        
        iteration = iteration_info.get('iteration', 0)
        quality_score = iteration_info.get('quality_score', 0.0)
        passed = iteration_info.get('passed', False)
        notes = iteration_info.get('notes', '')
        
        status_icon = "✅" if passed else "❌"
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append(f"📝 Iteration {iteration}: ", style="bold")
            content.append(f"Quality {quality_score:.2f}/10 ", style="cyan")
            content.append(f"{status_icon}", style="green" if passed else "red")
            if notes:
                content.append(f" - {notes}", style="dim")
            
            self._console.print(content)
        else:
            print(f"📝 Iteration {iteration}: Quality {quality_score:.2f}/10 {status_icon}", end="")
            if notes:
                print(f" - {notes}")
            else:
                print()
    
    def log_iteration_table(self, history: List[Any]) -> None:
        """
        Log a table of all iterations.
        
        Args:
            history: List of IterationResult objects
        """
        if not self.enabled or not history:
            return
        
        if RICH_AVAILABLE and self._console:
            table = Table(
                title="📊 Iteration Summary",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan"
            )
            
            table.add_column("ITER", justify="center", style="bold")
            table.add_column("QUALITY", justify="center")
            table.add_column("DELTA", justify="center")
            table.add_column("PASS", justify="center")
            table.add_column("NOTES", justify="left")
            
            prev_score = 0.0
            for item in history:
                iteration = getattr(item, 'iteration', 0)
                score = getattr(item, 'quality_score', 0.0)
                passed = getattr(item, 'passed', False)
                
                delta = score - prev_score
                delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
                
                pass_icon = "✅" if passed else "❌"
                
                # Determine notes based on convergence
                notes = ""
                if hasattr(item, 'convergence_score') and item.convergence_score > 0.8:
                    notes = "Converged"
                
                table.add_row(
                    str(iteration),
                    f"{score:.2f}",
                    delta_str,
                    pass_icon,
                    notes
                )
                
                prev_score = score
            
            self._console.print()
            self._console.print(table)
        else:
            print()
            print("═" * 60)
            print("📊 Iteration Summary")
            print("═" * 60)
            print(f"{'ITER':^6} | {'QUALITY':^8} | {'DELTA':^8} | {'PASS':^6} | NOTES")
            print("-" * 60)
            
            prev_score = 0.0
            for item in history:
                iteration = getattr(item, 'iteration', 0)
                score = getattr(item, 'quality_score', 0.0)
                passed = getattr(item, 'passed', False)
                
                delta = score - prev_score
                delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
                
                pass_str = "OK" if passed else "X"
                notes = ""
                
                print(f"{iteration:^6} | {score:^8.2f} | {delta_str:^8} | {pass_str:^6} | {notes}")
                
                prev_score = score
    
    # =========================================================================
    # STATE MANAGER SNAPSHOTS
    # =========================================================================
    
    def log_state_manager_snapshot(
        self,
        state_manager: Any,
        label: str = "State Manager"
    ) -> None:
        """
        Log a snapshot of a state manager.
        
        Args:
            state_manager: CouncilStateManager or AgentStateManager
            label: Label for this snapshot
        """
        if not self.enabled:
            return
        
        # Extract state data
        snapshot = self._extract_state_snapshot(state_manager)
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append(f"📸 {label} Snapshot\n\n", style="bold cyan")
            
            for key, value in snapshot.items():
                content.append(f"  {key}: ", style="yellow")
                if isinstance(value, dict):
                    content.append(f"\n")
                    for k, v in value.items():
                        content.append(f"    {k}: {v}\n", style="dim")
                else:
                    content.append(f"{value}\n", style="dim")
            
            panel = Panel(content, border_style="magenta", box=box.ROUNDED)
            self._console.print(panel)
        else:
            self._print_divider()
            print(f"📸 {label} Snapshot")
            self._print_divider()
            
            for key, value in snapshot.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
    
    def _extract_state_snapshot(self, state_manager: Any) -> Dict[str, Any]:
        """Extract relevant data from a state manager."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
        }
        
        # Try common state manager attributes
        if hasattr(state_manager, '_current_workflow_id'):
            snapshot["workflow_id"] = state_manager._current_workflow_id
        
        if hasattr(state_manager, 'get_current_workflow'):
            try:
                workflow = state_manager.get_current_workflow()
                if workflow:
                    snapshot["status"] = workflow.get("status", "unknown")
                    snapshot["current_iteration"] = workflow.get("current_iteration", 0)
                    snapshot["current_phase"] = workflow.get("current_phase")
                    snapshot["current_council"] = workflow.get("current_council")
            except Exception:
                pass
        
        return snapshot
    
    def log_council_execution_complete(
        self,
        council_name: str,
        success: bool,
        duration: Optional[float] = None,
        model_used: Optional[str] = None,
        token_usage: Optional[Dict[str, int]] = None
    ) -> None:
        """
        Log completion of a council execution.
        
        Args:
            council_name: Name of the council
            success: Whether execution succeeded
            duration: Execution duration in seconds
            model_used: Model that was used
            token_usage: Token usage statistics
        """
        if not self.enabled:
            return
        
        # Calculate duration from tracked start time if not provided
        if duration is None and council_name in self._start_times:
            duration = time.time() - self._start_times[council_name]
        
        status_icon = "✅" if success else "❌"
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append(f"{status_icon} Council Complete: ", style="bold")
            content.append(f"{council_name}", style="cyan")
            
            if duration:
                content.append(f" ({duration:.2f}s)", style="dim")
            
            if model_used:
                content.append(f"\n   Model: {model_used}", style="yellow")
            
            if token_usage:
                content.append(f"\n   Tokens: ", style="dim")
                content.append(f"in={token_usage.get('input', 0)}, out={token_usage.get('output', 0)}", style="dim")
            
            self._console.print(content)
        else:
            print(f"{status_icon} Council Complete: {council_name}", end="")
            if duration:
                print(f" ({duration:.2f}s)")
            else:
                print()
            
            if model_used:
                print(f"   Model: {model_used}")
            
            if token_usage:
                print(f"   Tokens: in={token_usage.get('input', 0)}, out={token_usage.get('output', 0)}")
    
    # =========================================================================
    # MULTI-VIEW LOGGING
    # =========================================================================
    
    def log_multi_view_comparison(
        self,
        optimist_output: Any,
        skeptic_output: Any
    ) -> None:
        """
        Log side-by-side comparison of optimist vs skeptic perspectives.
        
        Args:
            optimist_output: Output from OptimistCouncil
            skeptic_output: Output from SkepticCouncil
        """
        if not self.enabled:
            return
        
        # Extract key information
        opt_conf = getattr(optimist_output, 'confidence', 0.5)
        skep_conf = getattr(skeptic_output, 'confidence', 0.5)
        agreement = 1.0 - abs(opt_conf - skep_conf)
        
        opt_opportunities = getattr(optimist_output, 'opportunities', [])
        skep_risks = getattr(skeptic_output, 'risks', [])
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("\n🔀 Multi-View Comparison\n", style="bold cyan")
            content.append("─" * 50 + "\n")
            
            # Agreement score
            agreement_style = "green" if agreement > 0.7 else "yellow" if agreement > 0.4 else "red"
            content.append(f"Agreement: {agreement:.1%}\n", style=agreement_style)
            
            # Optimist summary
            content.append("\n🌟 OPTIMIST ", style="bold green")
            content.append(f"(Confidence: {opt_conf:.2f})\n", style="dim")
            if opt_opportunities:
                for i, opp in enumerate(opt_opportunities[:3]):
                    content.append(f"  + {opp}\n", style="green")
            
            # Skeptic summary
            content.append("\n🔍 SKEPTIC ", style="bold red")
            content.append(f"(Confidence: {skep_conf:.2f})\n", style="dim")
            if skep_risks:
                for i, risk in enumerate(skep_risks[:3]):
                    content.append(f"  - {risk}\n", style="red")
            
            content.append("─" * 50, style="dim")
            
            self._console.print(content)
        else:
            print("\n🔀 Multi-View Comparison")
            print("─" * 50)
            print(f"Agreement: {agreement:.1%}")
            print(f"\n🌟 OPTIMIST (Confidence: {opt_conf:.2f})")
            for opp in opt_opportunities[:3]:
                print(f"  + {opp}")
            print(f"\n🔍 SKEPTIC (Confidence: {skep_conf:.2f})")
            for risk in skep_risks[:3]:
                print(f"  - {risk}")
            print("─" * 50)
    
    def log_convergence_evolution(self, summary: Dict[str, Any]) -> None:
        """
        Log convergence metrics evolution over iterations.
        
        Args:
            summary: Iteration summary dictionary from IterationManager
        """
        if not self.enabled:
            return
        
        scores = summary.get("scores_per_iteration", [])
        converged = summary.get("converged", False)
        reason = summary.get("convergence_reason", "")
        consecutive = summary.get("consecutive_small_improvements", 0)
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("\n📈 Convergence Evolution\n", style="bold cyan")
            
            if scores:
                # Build simple ASCII chart
                max_score = max(scores) if scores else 1
                for i, score in enumerate(scores):
                    bar_len = int((score / max_score) * 20)
                    bar = "█" * bar_len
                    content.append(f"  Iter {i+1}: ", style="dim")
                    content.append(f"{score:.2f} ", style="cyan")
                    content.append(f"{bar}\n", style="green")
            
            # Convergence status
            status = "✅ Converged" if converged else "🔄 Not converged"
            content.append(f"\nStatus: {status}\n", style="bold")
            if reason:
                content.append(f"Reason: {reason}\n", style="dim")
            if consecutive > 0:
                content.append(f"Consecutive small improvements: {consecutive}\n", style="yellow")
            
            self._console.print(content)
        else:
            print("\n📈 Convergence Evolution")
            for i, score in enumerate(scores):
                bar_len = int((score / max(scores)) * 20) if scores else 0
                bar = "█" * bar_len
                print(f"  Iter {i+1}: {score:.2f} {bar}")
            
            status = "✅ Converged" if converged else "🔄 Not converged"
            print(f"\nStatus: {status}")
            if reason:
                print(f"Reason: {reason}")
    
    def log_time_usage(
        self,
        iteration: int,
        elapsed_seconds: float,
        remaining_seconds: float
    ) -> None:
        """
        Log time usage for an iteration.
        
        Args:
            iteration: Current iteration number
            elapsed_seconds: Time elapsed so far
            remaining_seconds: Time remaining
        """
        if not self.enabled:
            return
        
        total = elapsed_seconds + remaining_seconds
        pct_used = (elapsed_seconds / total * 100) if total > 0 else 0
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append(f"⏱️  Iteration {iteration}: ", style="bold")
            content.append(f"{elapsed_seconds:.1f}s elapsed, ", style="cyan")
            content.append(f"{remaining_seconds:.1f}s remaining ", style="green")
            content.append(f"({pct_used:.0f}% used)", style="dim")
            
            self._console.print(content)
        else:
            print(f"⏱️  Iteration {iteration}: {elapsed_seconds:.1f}s elapsed, {remaining_seconds:.1f}s remaining ({pct_used:.0f}% used)")
    
    def log_model_selection(
        self,
        decision: Any,
        confidence: float
    ) -> None:
        """
        Log model selection decision with reasoning.
        
        Args:
            decision: SupervisorDecision object
            confidence: Confidence in the decision
        """
        if not self.enabled:
            return
        
        models = getattr(decision, 'models', [])
        reason = getattr(decision, 'reason', '')
        downgraded = getattr(decision, 'downgraded', False)
        wait = getattr(decision, 'wait_for_capacity', False)
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("🤖 Model Selection: ", style="bold")
            content.append(f"{', '.join(models)}\n", style="cyan")
            
            if downgraded:
                content.append("   ⚠️  Downgraded\n", style="yellow")
            if wait:
                content.append("   ⏳ Waiting for capacity\n", style="blue")
            
            content.append(f"   Reason: {reason}\n", style="dim")
            content.append(f"   Confidence: {confidence:.2f}", style="dim")
            
            self._console.print(content)
        else:
            print(f"🤖 Model Selection: {', '.join(models)}")
            if downgraded:
                print("   ⚠️  Downgraded")
            if wait:
                print("   ⏳ Waiting for capacity")
            print(f"   Reason: {reason}")
            print(f"   Confidence: {confidence:.2f}")
    
    def log_resource_decision(
        self,
        original_models: List[str],
        selected_models: List[str],
        downgraded: bool,
        wait_for_capacity: bool,
        vram_pressure: str,
        vram_info: str,
        time_remaining: float,
        max_wait_minutes: Optional[float] = None
    ) -> None:
        """
        Log resource decision (downgrade/wait) with VRAM pressure and timing.
        
        Phase 8.2: Detailed logging for resource allocation decisions.
        
        Args:
            original_models: Original model selection before downgrade
            selected_models: Final model selection
            downgraded: Whether models were downgraded
            wait_for_capacity: Whether waiting for GPU capacity
            vram_pressure: Current GPU pressure level
            vram_info: VRAM availability information
            time_remaining: Time remaining in mission
            max_wait_minutes: Maximum wait time if waiting for capacity
        """
        if not self.enabled:
            return
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            if downgraded:
                content.append("⚠️  Resource Decision: DOWNGRADE\n", style="yellow")
                content.append(f"   Original: {', '.join(original_models)}\n", style="dim")
                content.append(f"   Selected: {', '.join(selected_models)}\n", style="cyan")
            elif wait_for_capacity:
                content.append("⏳ Resource Decision: WAIT FOR CAPACITY\n", style="blue")
                content.append(f"   Models: {', '.join(selected_models)}\n", style="cyan")
                if max_wait_minutes:
                    content.append(f"   Max wait: {max_wait_minutes:.1f} minutes\n", style="dim")
            else:
                return  # No decision to log
            
            content.append(f"   GPU Pressure: {vram_pressure}\n", style="dim")
            content.append(f"   {vram_info}\n", style="dim")
            content.append(f"   Time remaining: {time_remaining:.1f} minutes", style="dim")
            
            self._console.print(content)
        else:
            if downgraded:
                print(f"⚠️  Resource Decision: DOWNGRADE")
                print(f"   Original: {', '.join(original_models)}")
                print(f"   Selected: {', '.join(selected_models)}")
            elif wait_for_capacity:
                print(f"⏳ Resource Decision: WAIT FOR CAPACITY")
                print(f"   Models: {', '.join(selected_models)}")
                if max_wait_minutes:
                    print(f"   Max wait: {max_wait_minutes:.1f} minutes")
            else:
                return
            
            print(f"   GPU Pressure: {vram_pressure}")
            print(f"   {vram_info}")
            print(f"   Time remaining: {time_remaining:.1f} minutes")
    
    def log_arbiter_synthesis(self, decision: Any) -> None:
        """
        Log full Arbiter synthesis output.
        
        Args:
            decision: ArbiterDecision object
        """
        if not self.enabled:
            return
        
        confidence = getattr(decision, 'confidence', 0)
        meta_analysis = getattr(decision, 'meta_analysis', '')
        ranked_insights = getattr(decision, 'ranked_insights', [])
        optimist_summary = getattr(decision, 'optimist_summary', '')
        skeptic_summary = getattr(decision, 'skeptic_summary', '')
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("\n⚖️ Arbiter Synthesis\n", style="bold magenta")
            content.append("═" * 50 + "\n")
            
            content.append(f"Confidence: {confidence:.2f}\n\n", style="cyan")
            
            if meta_analysis:
                content.append("Meta-Analysis:\n", style="bold yellow")
                truncated = self._truncate(meta_analysis, "council_output")
                content.append(f"{truncated}\n\n", style="dim")
            
            if ranked_insights:
                content.append("Top Insights:\n", style="bold green")
                for i, insight in enumerate(ranked_insights[:5]):
                    content.append(f"  {i+1}. {insight}\n", style="green")
                content.append("\n")
            
            if optimist_summary:
                content.append("🌟 Optimist Path:\n", style="bold green")
                truncated = self._truncate(optimist_summary, "council_output")
                content.append(f"{truncated}\n\n", style="dim")
            
            if skeptic_summary:
                content.append("🔍 Skeptic Path:\n", style="bold red")
                truncated = self._truncate(skeptic_summary, "council_output")
                content.append(f"{truncated}\n", style="dim")
            
            content.append("═" * 50, style="dim")
            
            self._console.print(content)
        else:
            print("\n⚖️ Arbiter Synthesis")
            print("═" * 50)
            print(f"Confidence: {confidence:.2f}\n")
            
            if meta_analysis:
                print("Meta-Analysis:")
                print(self._truncate(meta_analysis, "council_output"))
                print()
            
            if ranked_insights:
                print("Top Insights:")
                for i, insight in enumerate(ranked_insights[:5]):
                    print(f"  {i+1}. {insight}")
                print()
            
            if optimist_summary:
                print("🌟 Optimist Path:")
                print(self._truncate(optimist_summary, "council_output"))
                print()
            
            if skeptic_summary:
                print("🔍 Skeptic Path:")
                print(self._truncate(skeptic_summary, "council_output"))
            
            print("═" * 50)
    
    def log_phase_deepening(
        self,
        phase_name: str,
        councils_activated: List[str]
    ) -> None:
        """
        Log phase deepening loop information.
        
        Args:
            phase_name: Name of the phase being deepened
            councils_activated: List of councils activated for deepening
        """
        if not self.enabled:
            return
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("🔄 Phase Deepening: ", style="bold blue")
            content.append(f"{phase_name}\n", style="cyan")
            content.append(f"   Councils: {', '.join(councils_activated)}", style="dim")
            
            self._console.print(content)
        else:
            print(f"🔄 Phase Deepening: {phase_name}")
            print(f"   Councils: {', '.join(councils_activated)}")
    
    # =========================================================================
    # REASONING SUPERVISOR LOGGING
    # =========================================================================
    
    def log_supervisor_metrics(self, metrics: Any, mission_id: str = None, phase_name: str = None) -> None:
        """
        Log ReasoningSupervisor phase metrics.
        
        Args:
            metrics: PhaseMetrics object with difficulty, uncertainty, etc.
            mission_id: Optional mission ID for SSE publishing
            phase_name: Optional phase name for SSE publishing
        """
        if not self.enabled:
            return
        
        difficulty = getattr(metrics, 'difficulty_score', 0)
        uncertainty = getattr(metrics, 'uncertainty_score', 0)
        progress = getattr(metrics, 'progress_score', 0)
        novelty = getattr(metrics, 'novelty_score', 0)
        confidence = getattr(metrics, 'confidence_score', 0)
        
        # Publish SSE event if mission_id is available
        if mission_id and SSE_AVAILABLE and sse_manager:
            _publish_sse_event(sse_manager.publish_phase_metrics(
                mission_id=mission_id,
                phase_name=phase_name,
                difficulty_score=difficulty,
                uncertainty_score=uncertainty,
                progress_score=progress,
                novelty_score=novelty,
                confidence_score=confidence
            ))
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("📊 Phase Metrics:\n", style="bold cyan")
            
            # Create bar visualizations
            def bar(value: float, width: int = 20) -> str:
                filled = int(value * width)
                return "█" * filled + "░" * (width - filled)
            
            content.append(f"   Difficulty:  {bar(difficulty)} {difficulty:.2f}\n", style="yellow")
            content.append(f"   Uncertainty: {bar(uncertainty)} {uncertainty:.2f}\n", style="red" if uncertainty > 0.7 else "yellow")
            content.append(f"   Progress:    {bar(progress)} {progress:.2f}\n", style="green")
            content.append(f"   Novelty:     {bar(novelty)} {novelty:.2f}\n", style="blue")
            content.append(f"   Confidence:  {bar(confidence)} {confidence:.2f}", style="magenta")
            
            self._console.print(content)
        else:
            print("📊 Phase Metrics:")
            print(f"   Difficulty:  {difficulty:.2f}")
            print(f"   Uncertainty: {uncertainty:.2f}")
            print(f"   Progress:    {progress:.2f}")
            print(f"   Novelty:     {novelty:.2f}")
            print(f"   Confidence:  {confidence:.2f}")
    
    def log_depth_contract(self, contract: Any, phase_name: str) -> None:
        """
        Log a depth contract for a phase.
        
        Args:
            contract: DepthContract object
            phase_name: Name of the phase
        """
        if not self.enabled:
            return
        
        exploration = getattr(contract, 'exploration_depth', 0)
        max_rounds = getattr(contract, 'max_rounds', 1)
        model_tier = getattr(contract, 'model_tier', 'unknown')
        allow_alt = getattr(contract, 'allow_alternatives', False)
        focus = getattr(contract, 'focus_areas', [])
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("📋 Depth Contract: ", style="bold")
            content.append(f"{phase_name}\n", style="cyan")
            content.append(f"   Exploration: {exploration:.2f} | ", style="dim")
            content.append(f"Rounds: {max_rounds} | ", style="dim")
            content.append(f"Tier: {model_tier} | ", style="yellow")
            content.append(f"Alternatives: {'✓' if allow_alt else '✗'}", style="green" if allow_alt else "red")
            
            if focus:
                content.append(f"\n   Focus: {', '.join(focus[:3])}", style="dim")
            
            self._console.print(content)
        else:
            print(f"📋 Depth Contract: {phase_name}")
            print(f"   Exploration: {exploration:.2f} | Rounds: {max_rounds} | Tier: {model_tier}")
            print(f"   Alternatives: {'Yes' if allow_alt else 'No'}")
            if focus:
                print(f"   Focus: {', '.join(focus[:3])}")
    
    def log_multiview_triggered(self, reason: str, metrics: Any) -> None:
        """
        Log when multi-view analysis is triggered.
        
        Args:
            reason: Why multi-view was triggered
            metrics: PhaseMetrics that triggered multi-view
        """
        if not self.enabled:
            return
        
        difficulty = getattr(metrics, 'difficulty_score', 0)
        uncertainty = getattr(metrics, 'uncertainty_score', 0)
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("🔀 Multi-View Triggered\n", style="bold magenta")
            content.append(f"   Reason: {reason}\n", style="dim")
            content.append(f"   Difficulty: {difficulty:.2f} | Uncertainty: {uncertainty:.2f}", style="yellow")
            
            self._console.print(content)
        else:
            print("🔀 Multi-View Triggered")
            print(f"   Reason: {reason}")
            print(f"   Difficulty: {difficulty:.2f} | Uncertainty: {uncertainty:.2f}")
    
    def log_deepening_plan(self, plan: Any) -> None:
        """
        Log a deepening plan from ReasoningSupervisor.
        
        Args:
            plan: DeepeningPlan object
        """
        if not self.enabled:
            return
        
        councils = []
        if getattr(plan, 'run_researcher', False):
            councils.append("Researcher")
        if getattr(plan, 'run_planner', False):
            councils.append("Planner")
        if getattr(plan, 'run_coder', False):
            councils.append("Coder")
        if getattr(plan, 'run_evaluator', False):
            councils.append("Evaluator")
        if getattr(plan, 'run_simulation', False):
            councils.append("Simulation")
        
        reason = getattr(plan, 'reason', '')
        focus = getattr(plan, 'focus_areas', [])
        max_rounds = getattr(plan, 'max_deepening_rounds', 1)
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("🔄 Deepening Plan\n", style="bold blue")
            content.append(f"   Reason: {reason}\n", style="dim")
            content.append(f"   Councils: {', '.join(councils) if councils else 'None'}\n", style="cyan")
            content.append(f"   Rounds: {max_rounds}", style="dim")
            
            if focus:
                content.append(f"\n   Focus: {', '.join(focus[:3])}", style="yellow")
            
            self._console.print(content)
        else:
            print("🔄 Deepening Plan")
            print(f"   Reason: {reason}")
            print(f"   Councils: {', '.join(councils) if councils else 'None'}")
            print(f"   Rounds: {max_rounds}")
            if focus:
                print(f"   Focus: {', '.join(focus[:3])}")
    
    def log_loop_detected(self, detection: Any) -> None:
        """
        Log when a reasoning loop is detected.
        
        Args:
            detection: LoopDetection object
        """
        if not self.enabled:
            return
        
        similarity = getattr(detection, 'similarity_score', 0)
        stagnation = getattr(detection, 'stagnation_count', 0)
        recommendation = getattr(detection, 'recommendation', '')
        details = getattr(detection, 'details', '')
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("⚠️  Loop Detected\n", style="bold red")
            content.append(f"   Similarity: {similarity:.1%} | ", style="yellow")
            content.append(f"Stagnation: {stagnation} iterations\n", style="yellow")
            content.append(f"   Recommendation: {recommendation}\n", style="cyan")
            if details:
                content.append(f"   Details: {details}", style="dim")
            
            self._console.print(content)
        else:
            print("⚠️  Loop Detected")
            print(f"   Similarity: {similarity:.1%} | Stagnation: {stagnation}")
            print(f"   Recommendation: {recommendation}")
            if details:
                print(f"   Details: {details}")
    
    def log_mission_iteration(self, iteration: int, metrics: Any) -> None:
        """
        Log mission iteration information with metrics.
        
        Args:
            iteration: Current iteration number
            metrics: MissionMetrics object
        """
        if not self.enabled:
            return
        
        avg_difficulty = getattr(metrics, 'avg_difficulty', 0)
        avg_uncertainty = getattr(metrics, 'avg_uncertainty', 0)
        progress = getattr(metrics, 'overall_progress', 0)
        convergence = getattr(metrics, 'convergence_score', 0)
        time_remaining = getattr(metrics, 'time_remaining_minutes', 0)
        stagnation = getattr(metrics, 'stagnation_count', 0)
        loop_detected = getattr(metrics, 'loop_detected', False)
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append(f"\n{'═' * 50}\n", style="bold")
            content.append(f"📍 Mission Iteration {iteration}\n", style="bold cyan")
            content.append(f"{'═' * 50}\n", style="bold")
            
            # Progress bar for convergence
            def bar(value: float, width: int = 30) -> str:
                filled = int(value * width)
                return "█" * filled + "░" * (width - filled)
            
            content.append(f"   Convergence: {bar(convergence)} {convergence:.1%}\n", 
                          style="green" if convergence > 0.7 else "yellow")
            content.append(f"   Progress:    {bar(progress)} {progress:.1%}\n", style="blue")
            content.append(f"   Time Left:   {time_remaining:.1f} min\n", style="dim")
            content.append(f"   Difficulty:  {avg_difficulty:.2f} | Uncertainty: {avg_uncertainty:.2f}\n", style="dim")
            
            if stagnation > 0:
                content.append(f"   ⚠️  Stagnation: {stagnation} iteration(s)\n", style="yellow")
            if loop_detected:
                content.append(f"   ⚠️  Loop detected!\n", style="red")
            
            self._console.print(content)
        else:
            print(f"\n{'=' * 50}")
            print(f"📍 Mission Iteration {iteration}")
            print(f"{'=' * 50}")
            print(f"   Convergence: {convergence:.1%}")
            print(f"   Progress: {progress:.1%}")
            print(f"   Time Left: {time_remaining:.1f} min")
            print(f"   Difficulty: {avg_difficulty:.2f} | Uncertainty: {avg_uncertainty:.2f}")
            if stagnation > 0:
                print(f"   ⚠️  Stagnation: {stagnation} iteration(s)")
            if loop_detected:
                print(f"   ⚠️  Loop detected!")
    
    # =========================================================================
    # NEW PANELS: TIMELINE, MEMORY, INTERNET
    # =========================================================================
    
    def log_mission_timeline(
        self,
        phase_times: Dict[str, Tuple[float, float]],
        total_wall: float = 0.0,
        total_council: float = 0.0
    ) -> None:
        """
        Log mission execution timeline showing wall-clock vs council time.
        
        Args:
            phase_times: Dict mapping phase_name -> (wall_seconds, council_seconds)
            total_wall: Total wall-clock time
            total_council: Total council execution time
        """
        if not self.enabled:
            return
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("\n⏱️  Mission Timeline\n", style="bold cyan")
            content.append("─" * 50 + "\n")
            
            for phase_name, (wall, council) in phase_times.items():
                overhead = wall - council if wall > council else 0
                content.append(f"  {phase_name}:\n", style="bold")
                content.append(f"    Wall: {wall:.0f}s | Council: {council:.0f}s | Overhead: {overhead:.0f}s\n", style="dim")
            
            content.append("─" * 50 + "\n")
            total_overhead = total_wall - total_council if total_wall > total_council else 0
            content.append(f"  TOTAL: {total_wall:.0f}s wall, {total_council:.0f}s council, {total_overhead:.0f}s overhead\n", style="bold green")
            
            self._console.print(content)
        else:
            print("\n⏱️  Mission Timeline")
            print("─" * 50)
            
            for phase_name, (wall, council) in phase_times.items():
                overhead = wall - council if wall > council else 0
                print(f"  {phase_name}:")
                print(f"    Wall: {wall:.0f}s | Council: {council:.0f}s | Overhead: {overhead:.0f}s")
            
            print("─" * 50)
            total_overhead = total_wall - total_council if total_wall > total_council else 0
            print(f"  TOTAL: {total_wall:.0f}s wall, {total_council:.0f}s council, {total_overhead:.0f}s overhead")
    
    def log_memory_panel(
        self,
        memory_used_count: int = 0,
        memory_items_titles: Optional[List[str]] = None,
        used_in_prompt: bool = False,
        known_gaps: Optional[List[str]] = None
    ) -> None:
        """
        Log memory usage panel showing what memory was retrieved and used.
        
        Args:
            memory_used_count: Number of memory items retrieved
            memory_items_titles: Short titles of memory items
            used_in_prompt: Whether memory was injected into prompts
            known_gaps: Known gaps identified from memory
        """
        if not self.enabled:
            return
        
        memory_items_titles = memory_items_titles or []
        known_gaps = known_gaps or []
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("\n🧠 Memory Panel\n", style="bold magenta")
            content.append("─" * 50 + "\n")
            
            content.append(f"  Retrieved: {memory_used_count} items\n", style="cyan")
            content.append(f"  Used In Prompt: {'Yes' if used_in_prompt else 'No'}\n", 
                          style="green" if used_in_prompt else "yellow")
            
            if memory_items_titles:
                content.append("  Items:\n", style="bold")
                for title in memory_items_titles[:5]:
                    truncated = title[:60] + "..." if len(title) > 60 else title
                    content.append(f"    • {truncated}\n", style="dim")
            
            if known_gaps:
                content.append("  Known Gaps:\n", style="bold yellow")
                for gap in known_gaps[:3]:
                    truncated = gap[:60] + "..." if len(gap) > 60 else gap
                    content.append(f"    ⚠️  {truncated}\n", style="yellow")
            
            self._console.print(content)
        else:
            print("\n🧠 Memory Panel")
            print("─" * 50)
            print(f"  Retrieved: {memory_used_count} items")
            print(f"  Used In Prompt: {'Yes' if used_in_prompt else 'No'}")
            
            if memory_items_titles:
                print("  Items:")
                for title in memory_items_titles[:5]:
                    truncated = title[:60] + "..." if len(title) > 60 else title
                    print(f"    • {truncated}")
            
            if known_gaps:
                print("  Known Gaps:")
                for gap in known_gaps[:3]:
                    truncated = gap[:60] + "..." if len(gap) > 60 else gap
                    print(f"    ⚠️  {truncated}")
    
    def log_internet_panel(
        self,
        enabled: bool = True,
        searches_used: int = 0,
        quota: int = 10,
        queries_executed: Optional[List[str]] = None,
        search_rationale: Optional[str] = None
    ) -> None:
        """
        Log internet/web search usage panel.
        
        Args:
            enabled: Whether internet search is enabled
            searches_used: Number of searches performed
            quota: Total search quota
            queries_executed: List of queries that were executed
            search_rationale: Rationale for search decisions
        """
        if not self.enabled:
            return
        
        queries_executed = queries_executed or []
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("\n🌐 Internet Panel\n", style="bold blue")
            content.append("─" * 50 + "\n")
            
            status = "Enabled" if enabled else "Disabled"
            status_style = "green" if enabled else "red"
            content.append(f"  Status: {status}\n", style=status_style)
            content.append(f"  Searches Used: {searches_used}/{quota}\n", style="cyan")
            
            if queries_executed:
                content.append("  Queries:\n", style="bold")
                for query in queries_executed[:3]:
                    truncated = query[:50] + "..." if len(query) > 50 else query
                    content.append(f"    🔍 {truncated}\n", style="dim")
            
            if search_rationale:
                content.append(f"  Rationale: {search_rationale}\n", style="dim")
            
            self._console.print(content)
        else:
            print("\n🌐 Internet Panel")
            print("─" * 50)
            print(f"  Status: {'Enabled' if enabled else 'Disabled'}")
            print(f"  Searches Used: {searches_used}/{quota}")
            
            if queries_executed:
                print("  Queries:")
                for query in queries_executed[:3]:
                    truncated = query[:50] + "..." if len(query) > 50 else query
                    print(f"    🔍 {truncated}")
            
            if search_rationale:
                print(f"  Rationale: {search_rationale}")
    
    def log_multi_view_comparison(
        self,
        optimist_output: Any,
        skeptic_output: Any,
        compress_threshold: float = 0.90,
        mission_id: str = None
    ) -> None:
        """
        Log side-by-side comparison of optimist vs skeptic perspectives.
        
        Enhanced with smart compression:
        - If agreement > compress_threshold, show single summary
        - If agreement lower, show meaningful diffs
        
        Args:
            optimist_output: Output from OptimistCouncil
            skeptic_output: Output from SkepticCouncil
            compress_threshold: Agreement threshold for compression
            mission_id: Optional mission ID for SSE publishing
        """
        if not self.enabled:
            return
        
        # Extract key information
        opt_conf = getattr(optimist_output, 'confidence', 0.5)
        skep_conf = getattr(skeptic_output, 'confidence', 0.5)
        agreement = 1.0 - abs(opt_conf - skep_conf)
        
        opt_opportunities = getattr(optimist_output, 'opportunities', [])
        skep_risks = getattr(skeptic_output, 'risks', [])
        opt_reasoning = getattr(optimist_output, 'reasoning', '')
        skep_reasoning = getattr(skeptic_output, 'reasoning', '')
        
        # Check if we should compress
        high_agreement = agreement >= compress_threshold
        
        # Publish SSE event if mission_id is available
        if mission_id and SSE_AVAILABLE and sse_manager:
            # Convert opportunities and risks to strings if they're not already
            opt_opps_list = []
            if opt_opportunities:
                for opp in opt_opportunities[:5]:  # Limit for SSE
                    if isinstance(opp, str):
                        opt_opps_list.append(opp[:200])  # Truncate long strings
                    else:
                        opt_opps_list.append(str(opp)[:200])
            
            skep_risks_list = []
            if skep_risks:
                for risk in skep_risks[:5]:  # Limit for SSE
                    if isinstance(risk, str):
                        skep_risks_list.append(risk[:200])  # Truncate long strings
                    else:
                        skep_risks_list.append(str(risk)[:200])
            
            _publish_sse_event(sse_manager.publish_multi_view_analysis(
                mission_id=mission_id,
                agreement=agreement,
                optimist_confidence=opt_conf,
                skeptic_confidence=skep_conf,
                optimist_opportunities=opt_opps_list,
                skeptic_risks=skep_risks_list,
                high_agreement=high_agreement,
                confidence_gap=abs(opt_conf - skep_conf)
            ))
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("\n🔀 Multi-View Analysis\n", style="bold cyan")
            content.append("─" * 50 + "\n")
            
            # Agreement score with color
            agreement_style = "green" if agreement > 0.8 else "yellow" if agreement > 0.5 else "red"
            content.append(f"Agreement: {agreement:.1%}\n", style=agreement_style)
            
            if high_agreement:
                # Compressed view for high agreement
                content.append("\n✅ High Agreement - Unified Summary:\n", style="bold green")
                content.append(f"  Combined confidence: {(opt_conf + skep_conf) / 2:.2f}\n", style="cyan")
                
                # Merge insights
                if opt_opportunities or skep_risks:
                    content.append("  Key Points:\n", style="bold")
                    all_points = []
                    for opp in opt_opportunities[:2]:
                        all_points.append(f"+ {opp}")
                    for risk in skep_risks[:2]:
                        all_points.append(f"- {risk}")
                    for point in all_points[:4]:
                        content.append(f"    {point}\n", style="dim")
            else:
                # Detailed diff view for disagreement
                content.append("\n⚔️  Perspectives Differ - Detailed View:\n", style="bold yellow")
                
                # Optimist summary
                content.append("\n🌟 OPTIMIST ", style="bold green")
                content.append(f"(Confidence: {opt_conf:.2f})\n", style="dim")
                if opt_opportunities:
                    for i, opp in enumerate(opt_opportunities[:3]):
                        content.append(f"  + {opp}\n", style="green")
                elif opt_reasoning:
                    content.append(f"  {opt_reasoning[:150]}...\n", style="dim")
                
                # Skeptic summary
                content.append("\n🔍 SKEPTIC ", style="bold red")
                content.append(f"(Confidence: {skep_conf:.2f})\n", style="dim")
                if skep_risks:
                    for i, risk in enumerate(skep_risks[:3]):
                        content.append(f"  - {risk}\n", style="red")
                elif skep_reasoning:
                    content.append(f"  {skep_reasoning[:150]}...\n", style="dim")
                
                # Highlight key differences
                content.append("\n📊 Key Differences:\n", style="bold")
                conf_diff = abs(opt_conf - skep_conf)
                content.append(f"  Confidence gap: {conf_diff:.2f}\n", style="yellow")
            
            content.append("─" * 50, style="dim")
            self._console.print(content)
        else:
            print("\n🔀 Multi-View Analysis")
            print("─" * 50)
            print(f"Agreement: {agreement:.1%}")
            
            if high_agreement:
                print("\n✅ High Agreement - Unified Summary:")
                print(f"  Combined confidence: {(opt_conf + skep_conf) / 2:.2f}")
                if opt_opportunities or skep_risks:
                    print("  Key Points:")
                    for opp in opt_opportunities[:2]:
                        print(f"    + {opp}")
                    for risk in skep_risks[:2]:
                        print(f"    - {risk}")
            else:
                print("\n⚔️  Perspectives Differ:")
                print(f"\n🌟 OPTIMIST (Confidence: {opt_conf:.2f})")
                for opp in opt_opportunities[:3]:
                    print(f"  + {opp}")
                print(f"\n🔍 SKEPTIC (Confidence: {skep_conf:.2f})")
                for risk in skep_risks[:3]:
                    print(f"  - {risk}")
                print(f"\n📊 Confidence gap: {abs(opt_conf - skep_conf):.2f}")
            
            print("─" * 50)
    
    def log_mission_summary_panels(
        self,
        timeline: Dict[str, Tuple[float, float]],
        memory_summary: Dict[str, Any],
        internet_stats: Dict[str, Any],
        multiview_agreement: Optional[float] = None
    ) -> None:
        """
        Log all summary panels at mission end.
        
        This is a convenience method that logs timeline, memory, and internet
        panels together for a comprehensive mission summary.
        
        Args:
            timeline: Phase times dict for timeline panel
            memory_summary: Memory usage summary
            internet_stats: Internet usage statistics
            multiview_agreement: Overall multi-view agreement (optional)
        """
        if not self.enabled:
            return
        
        # Calculate totals for timeline
        total_wall = sum(times[0] for times in timeline.values())
        total_council = sum(times[1] for times in timeline.values())
        
        # Log each panel
        self.log_mission_timeline(timeline, total_wall, total_council)
        
        self.log_memory_panel(
            memory_used_count=memory_summary.get("memory_used_count", 0),
            memory_items_titles=memory_summary.get("memory_items_titles", []),
            used_in_prompt=memory_summary.get("used_in_prompt", False),
            known_gaps=memory_summary.get("known_gaps", [])
        )
        
        self.log_internet_panel(
            enabled=internet_stats.get("enabled", True),
            searches_used=internet_stats.get("total_searches", 0),
            quota=internet_stats.get("global_quota", 10),
            queries_executed=internet_stats.get("queries_executed", []),
            search_rationale=internet_stats.get("rationale")
        )
        
        # Add multi-view summary if available
        if multiview_agreement is not None:
            if RICH_AVAILABLE and self._console:
                content = Text()
                agreement_style = "green" if multiview_agreement > 0.8 else "yellow"
                content.append(f"\n🔀 Overall Multi-View Agreement: {multiview_agreement:.1%}\n", style=agreement_style)
                self._console.print(content)
            else:
                print(f"\n🔀 Overall Multi-View Agreement: {multiview_agreement:.1%}")


    # =========================================================================
    # COGNITIVE SPINE LOGGING
    # =========================================================================
    
    def log_schema_correction(
        self,
        council_name: str,
        field: str,
        action: str
    ) -> None:
        """
        Log a schema correction made by the CognitiveSpine.
        
        Args:
            council_name: Name of the council
            field: Field that was corrected
            action: Action taken (e.g., "stripped", "filled default")
        """
        if not self.enabled:
            return
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("🔧 Schema Correction: ", style="bold yellow")
            content.append(f"{council_name}", style="cyan")
            content.append(f" | Field: {field}", style="dim")
            content.append(f" | Action: {action}", style="yellow")
            
            self._console.print(content)
        else:
            print(f"🔧 Schema Correction: {council_name} | Field: {field} | Action: {action}")
    
    def log_output_truncation(
        self,
        council_name: str,
        original_size: int,
        new_size: int
    ) -> None:
        """
        Log output truncation by the CognitiveSpine.
        
        Args:
            council_name: Name of the council
            original_size: Original output size in characters
            new_size: New size after truncation
        """
        if not self.enabled:
            return
        
        reduction = original_size - new_size
        pct = (reduction / original_size * 100) if original_size > 0 else 0
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("✂️  Output Truncated: ", style="bold magenta")
            content.append(f"{council_name}", style="cyan")
            content.append(f" | {original_size:,} → {new_size:,} chars", style="dim")
            content.append(f" ({pct:.0f}% reduction)", style="yellow")
            
            self._console.print(content)
        else:
            print(f"✂️  Output Truncated: {council_name} | {original_size:,} → {new_size:,} chars ({pct:.0f}% reduction)")
    
    def log_fallback_trigger(
        self,
        reason: str,
        action: str
    ) -> None:
        """
        Log a fallback trigger by the CognitiveSpine or Supervisor.
        
        Args:
            reason: Why fallback was triggered
            action: Action being taken
        """
        if not self.enabled:
            return
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("⚡ Fallback Triggered: ", style="bold red")
            content.append(f"{reason}\n", style="yellow")
            content.append(f"   Action: {action}", style="cyan")
            
            self._console.print(content)
        else:
            print(f"⚡ Fallback Triggered: {reason}")
            print(f"   Action: {action}")
    
    def log_council_recovery(
        self,
        council_name: str,
        error: str,
        recovery_action: str
    ) -> None:
        """
        Log a council recovery attempt.
        
        Args:
            council_name: Name of the council
            error: Error that occurred
            recovery_action: Recovery action being taken
        """
        if not self.enabled:
            return
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("🔄 Council Recovery: ", style="bold blue")
            content.append(f"{council_name}\n", style="cyan")
            content.append(f"   Error: {error[:100]}...\n" if len(error) > 100 else f"   Error: {error}\n", style="red")
            content.append(f"   Recovery: {recovery_action}", style="green")
            
            self._console.print(content)
        else:
            print(f"🔄 Council Recovery: {council_name}")
            print(f"   Error: {error[:100]}..." if len(error) > 100 else f"   Error: {error}")
            print(f"   Recovery: {recovery_action}")
    
    def log_spine_decision(
        self,
        decision_type: str,
        details: Dict[str, Any]
    ) -> None:
        """
        Log a CognitiveSpine decision.
        
        This is the primary logging method for spine decisions including:
        - Schema validation results
        - Resource budget decisions
        - Memory compression
        - Contraction mode transitions
        
        Args:
            decision_type: Type of decision (e.g., "contraction_mode_entered")
            details: Details of the decision
        """
        if not self.enabled:
            return
        
        # Color coding by decision type
        style_map = {
            "contraction_mode_entered": ("🔻", "bold red"),
            "contraction_mode_exited": ("🔺", "bold green"),
            "memory_compression": ("📦", "bold blue"),
            "output_size_warning": ("⚠️ ", "bold yellow"),
            "schema_validation": ("✓", "bold green"),
            "resource_exceeded": ("🚫", "bold red"),
            "fallback_activated": ("⚡", "bold yellow"),
        }
        
        icon, style = style_map.get(decision_type, ("📋", "bold cyan"))
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append(f"{icon} Spine Decision: ", style=style)
            content.append(f"{decision_type}\n", style="cyan")
            
            for key, value in details.items():
                content.append(f"   {key}: ", style="dim")
                content.append(f"{value}\n", style="white")
            
            self._console.print(content)
        else:
            print(f"{icon} Spine Decision: {decision_type}")
            for key, value in details.items():
                print(f"   {key}: {value}")
    
    def log_contraction_mode(
        self,
        time_remaining: float,
        trigger: str
    ) -> None:
        """
        Log contraction mode activation.
        
        Args:
            time_remaining: Time remaining in minutes
            trigger: What triggered contraction mode
        """
        if not self.enabled:
            return
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("🔻 CONTRACTION MODE ACTIVATED\n", style="bold red")
            content.append(f"   Time Remaining: {time_remaining:.1f} min\n", style="yellow")
            content.append(f"   Trigger: {trigger}\n", style="dim")
            content.append("   → Skipping non-essential phases\n", style="cyan")
            content.append("   → Compressing memory\n", style="cyan")
            content.append("   → Forcing synthesis", style="cyan")
            
            panel = Panel(content, border_style="red", box=box.ROUNDED) if RICH_AVAILABLE else None
            if panel:
                self._console.print(panel)
            else:
                self._console.print(content)
        else:
            print("🔻 CONTRACTION MODE ACTIVATED")
            print(f"   Time Remaining: {time_remaining:.1f} min")
            print(f"   Trigger: {trigger}")
            print("   → Skipping non-essential phases")
            print("   → Compressing memory")
            print("   → Forcing synthesis")
    
    def log_resource_budget(
        self,
        council_name: str,
        budget: Dict[str, Any]
    ) -> None:
        """
        Log resource budget status for a council.
        
        Args:
            council_name: Name of the council
            budget: Budget dictionary with usage info
        """
        if not self.enabled:
            return
        
        tokens_used = budget.get("tokens_used", 0)
        max_tokens = budget.get("max_tokens", 0)
        chars_used = budget.get("output_chars_used", 0)
        max_chars = budget.get("max_output_chars", 0)
        is_exceeded = budget.get("is_exceeded", False)
        
        status_style = "red" if is_exceeded else "green"
        status_icon = "🚫" if is_exceeded else "✓"
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append(f"{status_icon} Resource Budget: ", style=f"bold {status_style}")
            content.append(f"{council_name}\n", style="cyan")
            
            # Token bar
            token_pct = (tokens_used / max_tokens * 100) if max_tokens > 0 else 0
            token_bar_len = int(token_pct / 5)  # 20 char bar
            token_bar = "█" * token_bar_len + "░" * (20 - token_bar_len)
            content.append(f"   Tokens: [{token_bar}] {token_pct:.0f}%\n", style="dim")
            
            # Chars bar
            char_pct = (chars_used / max_chars * 100) if max_chars > 0 else 0
            char_bar_len = int(char_pct / 5)
            char_bar = "█" * char_bar_len + "░" * (20 - char_bar_len)
            content.append(f"   Chars:  [{char_bar}] {char_pct:.0f}%", style="dim")
            
            self._console.print(content)
        else:
            status = "EXCEEDED" if is_exceeded else "OK"
            print(f"{status_icon} Resource Budget: {council_name} [{status}]")
            print(f"   Tokens: {tokens_used:,} / {max_tokens:,}")
            print(f"   Chars: {chars_used:,} / {max_chars:,}")
    
    def log_phase_validation(
        self,
        phase_name: str,
        is_valid: bool,
        corrections: List[str],
        warnings: List[str]
    ) -> None:
        """
        Log phase boundary validation results.
        
        Args:
            phase_name: Name of the phase
            is_valid: Whether validation passed
            corrections: List of corrections made
            warnings: List of warnings
        """
        if not self.enabled:
            return
        
        status_icon = "✓" if is_valid else "⚠️"
        status_style = "green" if is_valid else "yellow"
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append(f"{status_icon} Phase Validation: ", style=f"bold {status_style}")
            content.append(f"{phase_name}\n", style="cyan")
            
            if corrections:
                content.append("   Corrections:\n", style="yellow")
                for corr in corrections[:5]:
                    content.append(f"     • {corr}\n", style="dim")
            
            if warnings:
                content.append("   Warnings:\n", style="red")
                for warn in warnings[:5]:
                    content.append(f"     ⚠ {warn}\n", style="dim")
            
            self._console.print(content)
        else:
            status = "VALID" if is_valid else "CORRECTED"
            print(f"{status_icon} Phase Validation: {phase_name} [{status}]")
            if corrections:
                print("   Corrections:")
                for corr in corrections[:5]:
                    print(f"     • {corr}")
            if warnings:
                print("   Warnings:")
                for warn in warnings[:5]:
                    print(f"     ⚠ {warn}")


    # =========================================================================
    # DYNAMIC COUNCIL LOGGING
    # =========================================================================
    
    def log_council_definition(self, definition: Dict[str, Any]) -> None:
        """
        Log a dynamically generated council definition.
        
        Args:
            definition: CouncilDefinition dict with models, consensus, etc.
        """
        if not self.enabled:
            return
        
        name = definition.get("name", "unknown")
        council_type = definition.get("council_type", "unknown")
        models = definition.get("models", [])
        consensus_type = definition.get("consensus_type", "unknown")
        phase = definition.get("phase", "")
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("🔧 Dynamic Council Definition\n", style="bold cyan")
            content.append("─" * 50 + "\n")
            
            content.append(f"  Type: ", style="dim")
            content.append(f"{council_type}\n", style="bold yellow")
            
            content.append(f"  Phase: ", style="dim")
            content.append(f"{phase}\n", style="cyan")
            
            content.append(f"  Consensus: ", style="dim")
            content.append(f"{consensus_type}\n", style="green")
            
            content.append("  Models:\n", style="bold")
            for model_info in models:
                model_name = model_info.get("model", "unknown")
                temp = model_info.get("temperature", 0.5)
                persona = model_info.get("persona")
                
                model_str = f"    • {model_name} @ {temp:.2f}"
                content.append(model_str, style="cyan")
                if persona:
                    content.append(f" [{persona}]", style="magenta")
                content.append("\n")
            
            content.append("─" * 50, style="dim")
            
            self._console.print(content)
        else:
            print("🔧 Dynamic Council Definition")
            print("─" * 50)
            print(f"  Type: {council_type}")
            print(f"  Phase: {phase}")
            print(f"  Consensus: {consensus_type}")
            print("  Models:")
            for model_info in models:
                model_name = model_info.get("model", "unknown")
                temp = model_info.get("temperature", 0.5)
                persona = model_info.get("persona")
                persona_str = f" [{persona}]" if persona else ""
                print(f"    • {model_name} @ {temp:.2f}{persona_str}")
            print("─" * 50)
    
    def log_persona_injection(
        self,
        council_name: str,
        model_name: str,
        persona_name: str
    ) -> None:
        """
        Log persona injection into a model's prompt.
        
        Args:
            council_name: Name of the council
            model_name: Name of the model
            persona_name: Name of the persona being injected
        """
        if not self.enabled:
            return
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("🎭 Persona Injection: ", style="bold magenta")
            content.append(f"{council_name}", style="cyan")
            content.append(f" | {model_name}", style="dim")
            content.append(f" ← ", style="white")
            content.append(f"{persona_name}", style="bold green")
            
            self._console.print(content)
        else:
            print(f"🎭 Persona Injection: {council_name} | {model_name} ← {persona_name}")
    
    def log_dynamic_fallback(
        self,
        council_type: str,
        reason: str
    ) -> None:
        """
        Log when dynamic council configuration falls back to static.
        
        Args:
            council_type: Type of council
            reason: Reason for fallback
        """
        if not self.enabled:
            return
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("⚠️  Dynamic Config Fallback: ", style="bold yellow")
            content.append(f"{council_type}\n", style="cyan")
            content.append(f"   Reason: {reason}", style="dim")
            
            self._console.print(content)
        else:
            print(f"⚠️  Dynamic Config Fallback: {council_type}")
            print(f"   Reason: {reason}")
    
    # =========================================================================
    # LAYERED PANEL LOGGING - COMPREHENSIVE SYSTEM VISIBILITY
    # =========================================================================
    
    def log_step_execution_panel(
        self,
        step_name: str,
        step_type: str,
        model_used: str,
        duration_s: float,
        attempts: int,
        status: str,
        output_preview: Optional[str] = None,
        pivot_suggestion: Optional[str] = None,
        error: Optional[str] = None,
        artifacts: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Log detailed step execution panel.
        
        Args:
            step_name: Name of the step
            step_type: Type of step (research, coding, etc.)
            model_used: Model used for execution
            duration_s: Duration in seconds
            attempts: Number of attempts
            status: Final status (completed, failed, skipped)
            output_preview: Preview of step output
            pivot_suggestion: Any pivot suggestions
            error: Error message if failed
            artifacts: Step artifacts
        """
        if not self.enabled:
            return
        
        status_icon = "✅" if status == "completed" else "❌" if status == "failed" else "⏸️"
        status_style = "green" if status == "completed" else "red" if status == "failed" else "yellow"
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append(f"{status_icon} Step Execution: ", style=f"bold {status_style}")
            content.append(f"{step_name}\n", style="bold cyan")
            content.append("─" * 50 + "\n", style="dim")
            
            content.append(f"  Type: {step_type}\n", style="dim")
            content.append(f"  Model: {model_used}\n", style="cyan")
            content.append(f"  Duration: {duration_s:.2f}s\n", style="dim")
            content.append(f"  Attempts: {attempts}\n", style="dim")
            content.append(f"  Status: {status}\n", style=status_style)
            
            if output_preview:
                truncated = self._truncate(output_preview, "council_output")
                content.append(f"\n  Output Preview:\n", style="bold")
                content.append(f"  {truncated}\n", style="dim")
            
            if pivot_suggestion:
                content.append(f"\n  🔄 Pivot Suggestion:\n", style="bold yellow")
                content.append(f"  {pivot_suggestion}\n", style="yellow")
            
            if error:
                content.append(f"\n  ❌ Error:\n", style="bold red")
                content.append(f"  {error}\n", style="red")
            
            if artifacts:
                content.append(f"\n  📦 Artifacts ({len(artifacts)}):\n", style="bold")
                for key, value in list(artifacts.items())[:3]:
                    truncated = self._truncate(str(value), "artifact_value")
                    content.append(f"    • {key}: {truncated}\n", style="dim")
            
            content.append("─" * 50, style="dim")
            
            panel = Panel(content, title="📋 Step Execution", border_style="blue", box=box.ROUNDED)
            self._console.print(panel)
        else:
            print(f"\n{'=' * 50}")
            print(f"{status_icon} Step Execution: {step_name}")
            print(f"{'=' * 50}")
            print(f"  Type: {step_type}")
            print(f"  Model: {model_used}")
            print(f"  Duration: {duration_s:.2f}s")
            print(f"  Attempts: {attempts}")
            print(f"  Status: {status}")
            if output_preview:
                print(f"\n  Output Preview:")
                print(f"  {self._truncate(output_preview, 'council_output')}")
            if pivot_suggestion:
                print(f"\n  🔄 Pivot Suggestion: {pivot_suggestion}")
            if error:
                print(f"\n  ❌ Error: {error}")
            if artifacts:
                print(f"\n  📦 Artifacts ({len(artifacts)}):")
                for key, value in list(artifacts.items())[:3]:
                    print(f"    • {key}: {self._truncate(str(value), 'artifact_value')}")
            print("=" * 50)
    
    def log_model_selection_panel(
        self,
        decision: Any,
        phase_name: str,
        time_remaining: float,
        total_time: float,
        gpu_stats: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log detailed model selection panel with supervisor reasoning.
        
        Args:
            decision: SupervisorDecision object
            phase_name: Name of the phase
            time_remaining: Time remaining in minutes
            total_time: Total time budget in minutes
            gpu_stats: GPU statistics if available
        """
        if not self.enabled:
            return
        
        models = getattr(decision, 'models', [])
        reason = getattr(decision, 'reason', '')
        downgraded = getattr(decision, 'downgraded', False)
        wait = getattr(decision, 'wait_for_capacity', False)
        estimated_vram = getattr(decision, 'estimated_vram', 0)
        phase_importance = getattr(decision, 'phase_importance', 0.5)
        fallback_models = getattr(decision, 'fallback_models', [])
        max_wait_minutes = getattr(decision, 'max_wait_minutes', 0)
        
        time_ratio = (time_remaining / total_time * 100) if total_time > 0 else 0
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("🤖 Model Selection Decision\n", style="bold cyan")
            content.append("─" * 50 + "\n", style="dim")
            
            content.append(f"  Phase: {phase_name}\n", style="bold")
            content.append(f"  Models Selected: ", style="dim")
            content.append(f"{', '.join(models)}\n", style="cyan")
            
            if downgraded:
                content.append(f"  ⚠️  Status: ", style="dim")
                content.append("DOWNGRADED\n", style="bold yellow")
            elif wait:
                content.append(f"  ⏳ Status: ", style="dim")
                content.append(f"WAITING (max {max_wait_minutes:.1f}min)\n", style="bold blue")
            else:
                content.append(f"  ✅ Status: ", style="dim")
                content.append("OPTIMAL\n", style="bold green")
            
            content.append(f"\n  Time Budget:\n", style="bold")
            content.append(f"    Remaining: {time_remaining:.1f} / {total_time:.1f} min ({time_ratio:.0f}%)\n", style="dim")
            
            content.append(f"\n  Reasoning:\n", style="bold")
            content.append(f"  {reason}\n", style="dim")
            
            content.append(f"\n  Phase Importance: {phase_importance:.2f}\n", style="dim")
            
            if estimated_vram > 0:
                content.append(f"  Estimated VRAM: {estimated_vram:,} MB\n", style="dim")
            
            if wait and fallback_models:
                content.append(f"\n  Fallback Models (if timeout):\n", style="bold yellow")
                content.append(f"  {', '.join(fallback_models)}\n", style="yellow")
            
            if gpu_stats:
                content.append(f"\n  GPU Status:\n", style="bold")
                available = gpu_stats.get('available_gpus', 0)
                utilization = gpu_stats.get('utilization_percent', 0)
                content.append(f"    Available GPUs: {available}\n", style="dim")
                content.append(f"    Utilization: {utilization:.1f}%\n", style="dim")
            
            content.append("─" * 50, style="dim")
            
            panel = Panel(content, title="🤖 Model Selection", border_style="magenta", box=box.ROUNDED)
            self._console.print(panel)
        else:
            print(f"\n{'=' * 50}")
            print("🤖 Model Selection Decision")
            print(f"{'=' * 50}")
            print(f"  Phase: {phase_name}")
            print(f"  Models Selected: {', '.join(models)}")
            status = "DOWNGRADED" if downgraded else f"WAITING (max {max_wait_minutes:.1f}min)" if wait else "OPTIMAL"
            print(f"  Status: {status}")
            print(f"  Time Budget: {time_remaining:.1f} / {total_time:.1f} min ({time_ratio:.0f}%)")
            print(f"\n  Reasoning: {reason}")
            print(f"  Phase Importance: {phase_importance:.2f}")
            if estimated_vram > 0:
                print(f"  Estimated VRAM: {estimated_vram:,} MB")
            if wait and fallback_models:
                print(f"\n  Fallback Models: {', '.join(fallback_models)}")
            if gpu_stats:
                print(f"\n  GPU Status:")
                print(f"    Available: {gpu_stats.get('available_gpus', 0)}")
                print(f"    Utilization: {gpu_stats.get('utilization_percent', 0):.1f}%")
            print("=" * 50)
    
    def log_memory_operations_panel(
        self,
        memory_used_count: int = 0,
        memory_items: Optional[List[Dict[str, Any]]] = None,
        used_in_prompt: bool = False,
        known_gaps: Optional[List[str]] = None,
        compression_operations: Optional[List[str]] = None
    ) -> None:
        """
        Log detailed memory operations panel.
        
        Args:
            memory_used_count: Number of memory items retrieved
            memory_items: List of memory items with details
            used_in_prompt: Whether memory was injected into prompts
            known_gaps: Known gaps identified
            compression_operations: Memory compression operations performed
        """
        if not self.enabled:
            return
        
        memory_items = memory_items or []
        known_gaps = known_gaps or []
        compression_operations = compression_operations or []
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("🧠 Memory Operations\n", style="bold magenta")
            content.append("─" * 50 + "\n", style="dim")
            
            content.append(f"  Retrieved: {memory_used_count} items\n", style="cyan")
            content.append(f"  Used In Prompt: ", style="dim")
            content.append(f"{'Yes' if used_in_prompt else 'No'}\n", 
                          style="green" if used_in_prompt else "yellow")
            
            if memory_items:
                content.append(f"\n  Memory Items:\n", style="bold")
                for item in memory_items[:5]:
                    title = item.get('title', item.get('summary', 'Unknown'))[:60]
                    score = item.get('score', 0)
                    content.append(f"    • {title}\n", style="dim")
                    if score > 0:
                        content.append(f"      (relevance: {score:.2f})\n", style="dim")
            
            if known_gaps:
                content.append(f"\n  ⚠️  Known Gaps:\n", style="bold yellow")
                for gap in known_gaps[:3]:
                    truncated = gap[:60] + "..." if len(gap) > 60 else gap
                    content.append(f"    • {truncated}\n", style="yellow")
            
            if compression_operations:
                content.append(f"\n  📦 Compression Operations:\n", style="bold")
                for op in compression_operations[:3]:
                    content.append(f"    • {op}\n", style="dim")
            
            content.append("─" * 50, style="dim")
            
            panel = Panel(content, title="🧠 Memory Operations", border_style="magenta", box=box.ROUNDED)
            self._console.print(panel)
        else:
            print(f"\n{'=' * 50}")
            print("🧠 Memory Operations")
            print(f"{'=' * 50}")
            print(f"  Retrieved: {memory_used_count} items")
            print(f"  Used In Prompt: {'Yes' if used_in_prompt else 'No'}")
            if memory_items:
                print(f"\n  Memory Items:")
                for item in memory_items[:5]:
                    title = item.get('title', item.get('summary', 'Unknown'))[:60]
                    print(f"    • {title}")
            if known_gaps:
                print(f"\n  ⚠️  Known Gaps:")
                for gap in known_gaps[:3]:
                    print(f"    • {gap[:60]}...")
            if compression_operations:
                print(f"\n  📦 Compression Operations:")
                for op in compression_operations[:3]:
                    print(f"    • {op}")
            print("=" * 50)
    
    def log_internet_usage_panel(
        self,
        enabled: bool = True,
        searches_executed: int = 0,
        quota: int = 10,
        queries: Optional[List[Dict[str, Any]]] = None,
        search_rationale: Optional[str] = None,
        results_summary: Optional[Dict[str, int]] = None
    ) -> None:
        """
        Log detailed internet usage panel.
        
        Args:
            enabled: Whether internet search is enabled
            searches_executed: Number of searches performed
            quota: Total search quota
            queries: List of queries with details
            search_rationale: Rationale for search decisions
            results_summary: Summary of results per query
        """
        if not self.enabled:
            return
        
        queries = queries or []
        results_summary = results_summary or {}
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("🌐 Internet Usage\n", style="bold blue")
            content.append("─" * 50 + "\n", style="dim")
            
            status = "Enabled" if enabled else "Disabled"
            status_style = "green" if enabled else "red"
            content.append(f"  Status: {status}\n", style=status_style)
            
            quota_pct = (searches_executed / quota * 100) if quota > 0 else 0
            quota_bar_len = int(quota_pct / 5)
            quota_bar = "█" * quota_bar_len + "░" * (20 - quota_bar_len)
            content.append(f"  Searches: [{quota_bar}] {searches_executed}/{quota} ({quota_pct:.0f}%)\n", style="cyan")
            
            if queries:
                content.append(f"\n  Queries Executed:\n", style="bold")
                for i, query_info in enumerate(queries[:5], 1):
                    query_text = query_info.get('query', query_info.get('text', str(query_info)))[:50]
                    results_count = query_info.get('results_count', results_summary.get(str(i), 0))
                    content.append(f"    {i}. {query_text}...\n", style="dim")
                    if results_count > 0:
                        content.append(f"       → {results_count} results\n", style="green")
            
            if search_rationale:
                content.append(f"\n  Search Rationale:\n", style="bold")
                truncated = self._truncate(search_rationale, "council_output")
                content.append(f"  {truncated}\n", style="dim")
            
            content.append("─" * 50, style="dim")
            
            panel = Panel(content, title="🌐 Internet Usage", border_style="blue", box=box.ROUNDED)
            self._console.print(panel)
        else:
            print(f"\n{'=' * 50}")
            print("🌐 Internet Usage")
            print(f"{'=' * 50}")
            print(f"  Status: {'Enabled' if enabled else 'Disabled'}")
            print(f"  Searches: {searches_executed}/{quota} ({quota_pct:.0f}%)")
            if queries:
                print(f"\n  Queries Executed:")
                for i, query_info in enumerate(queries[:5], 1):
                    query_text = query_info.get('query', query_info.get('text', str(query_info)))[:50]
                    results_count = query_info.get('results_count', results_summary.get(str(i), 0))
                    print(f"    {i}. {query_text}... → {results_count} results")
            if search_rationale:
                print(f"\n  Search Rationale: {self._truncate(search_rationale, 'council_output')}")
            print("=" * 50)
    
    def log_consensus_mechanism_panel(
        self,
        algorithm: str,
        models_participating: List[str],
        agreement_scores: Optional[Dict[str, float]] = None,
        selected_output: Optional[str] = None,
        skip_reason: Optional[str] = None,
        consensus_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log detailed consensus mechanism panel.
        
        Args:
            algorithm: Consensus algorithm used (voting, weighted_blend, etc.)
            models_participating: List of models that participated
            agreement_scores: Agreement scores between outputs
            selected_output: The selected output
            skip_reason: Reason consensus was skipped (if applicable)
            consensus_details: Additional consensus details
        """
        if not self.enabled:
            return
        
        agreement_scores = agreement_scores or {}
        consensus_details = consensus_details or {}
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("⚖️  Consensus Mechanism\n", style="bold cyan")
            content.append("─" * 50 + "\n", style="dim")
            
            if skip_reason:
                content.append(f"  Status: ", style="dim")
                content.append("SKIPPED\n", style="bold yellow")
                content.append(f"  Reason: {skip_reason}\n", style="yellow")
            else:
                content.append(f"  Algorithm: ", style="dim")
                content.append(f"{algorithm}\n", style="bold cyan")
                content.append(f"  Models: {len(models_participating)}\n", style="dim")
                content.append(f"    {', '.join(models_participating)}\n", style="cyan")
            
            if agreement_scores:
                content.append(f"\n  Agreement Scores:\n", style="bold")
                for pair, score in list(agreement_scores.items())[:5]:
                    score_style = "green" if score > 0.8 else "yellow" if score > 0.5 else "red"
                    content.append(f"    {pair}: ", style="dim")
                    content.append(f"{score:.2f}\n", style=score_style)
            
            if selected_output:
                truncated = self._truncate(selected_output, "council_output")
                content.append(f"\n  Selected Output:\n", style="bold")
                content.append(f"  {truncated}\n", style="dim")
            
            if consensus_details:
                voting_details = consensus_details.get('voting', {})
                if voting_details:
                    winner = voting_details.get('winner')
                    votes = voting_details.get('votes', {})
                    if winner:
                        content.append(f"\n  Voting Results:\n", style="bold")
                        content.append(f"    Winner: {winner}\n", style="green")
                        if votes:
                            content.append(f"    Votes: {votes}\n", style="dim")
            
            content.append("─" * 50, style="dim")
            
            panel = Panel(content, title="⚖️  Consensus Mechanism", border_style="cyan", box=box.ROUNDED)
            self._console.print(panel)
        else:
            print(f"\n{'=' * 50}")
            print("⚖️  Consensus Mechanism")
            print(f"{'=' * 50}")
            if skip_reason:
                print(f"  Status: SKIPPED")
                print(f"  Reason: {skip_reason}")
            else:
                print(f"  Algorithm: {algorithm}")
                print(f"  Models: {len(models_participating)} - {', '.join(models_participating)}")
            if agreement_scores:
                print(f"\n  Agreement Scores:")
                for pair, score in list(agreement_scores.items())[:5]:
                    print(f"    {pair}: {score:.2f}")
            if selected_output:
                print(f"\n  Selected Output: {self._truncate(selected_output, 'council_output')}")
            print("=" * 50)
    
    def log_model_execution_panel(
        self,
        model_name: str,
        success: bool,
        output_preview: Optional[str] = None,
        token_usage: Optional[Dict[str, int]] = None,
        duration_s: Optional[float] = None,
        error: Optional[str] = None,
        persona: Optional[str] = None
    ) -> None:
        """
        Log per-model execution panel.
        
        Args:
            model_name: Name of the model
            success: Whether execution succeeded
            output_preview: Preview of model output
            token_usage: Token usage statistics
            duration_s: Duration in seconds
            error: Error message if failed
            persona: Persona applied to model (if any)
        """
        if not self.enabled:
            return
        
        status_icon = "✅" if success else "❌"
        status_style = "green" if success else "red"
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append(f"{status_icon} Model: ", style=f"bold {status_style}")
            content.append(f"{model_name}\n", style="bold cyan")
            content.append("─" * 30 + "\n", style="dim")
            
            if persona:
                content.append(f"  Persona: {persona}\n", style="magenta")
            
            if duration_s:
                content.append(f"  Duration: {duration_s:.2f}s\n", style="dim")
            
            if token_usage:
                input_tokens = token_usage.get('input', 0)
                output_tokens = token_usage.get('output', 0)
                total = input_tokens + output_tokens
                content.append(f"  Tokens: {total:,} (in: {input_tokens:,}, out: {output_tokens:,})\n", style="dim")
            
            if output_preview:
                truncated = self._truncate(output_preview, "council_output")
                content.append(f"\n  Output:\n", style="bold")
                content.append(f"  {truncated}\n", style="dim")
            
            if error:
                content.append(f"\n  ❌ Error: {error}\n", style="red")
            
            content.append("─" * 30, style="dim")
            
            panel = Panel(content, border_style="blue", box=box.ROUNDED)
            self._console.print(panel)
        else:
            print(f"\n{status_icon} Model: {model_name}")
            if persona:
                print(f"  Persona: {persona}")
            if duration_s:
                print(f"  Duration: {duration_s:.2f}s")
            if token_usage:
                total = token_usage.get('input', 0) + token_usage.get('output', 0)
                print(f"  Tokens: {total:,}")
            if output_preview:
                print(f"  Output: {self._truncate(output_preview, 'council_output')}")
            if error:
                print(f"  ❌ Error: {error}")
    
    def log_resource_management_panel(
        self,
        gpu_stats: Optional[Dict[str, Any]] = None,
        token_usage: Optional[Dict[str, int]] = None,
        time_remaining: Optional[float] = None,
        time_total: Optional[float] = None,
        resource_budgets: Optional[Dict[str, Dict[str, Any]]] = None,
        contraction_mode: bool = False
    ) -> None:
        """
        Log resource management panel.
        
        Args:
            gpu_stats: GPU statistics
            token_usage: Token usage per council/model
            time_remaining: Time remaining in minutes
            time_total: Total time budget in minutes
            resource_budgets: Resource budgets per council
            contraction_mode: Whether contraction mode is active
        """
        if not self.enabled:
            return
        
        resource_budgets = resource_budgets or {}
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("📊 Resource Management\n", style="bold yellow")
            content.append("─" * 50 + "\n", style="dim")
            
            if contraction_mode:
                content.append("  🔻 CONTRACTION MODE ACTIVE\n", style="bold red")
            
            if time_remaining is not None and time_total is not None:
                time_pct = (time_remaining / time_total * 100) if time_total > 0 else 0
                time_bar_len = int(time_pct / 5)
                time_bar = "█" * time_bar_len + "░" * (20 - time_bar_len)
                content.append(f"  Time Budget: [{time_bar}] {time_remaining:.1f}/{time_total:.1f} min ({time_pct:.0f}%)\n", style="cyan")
            
            if gpu_stats:
                content.append(f"\n  GPU Status:\n", style="bold")
                available = gpu_stats.get('available_gpus', 0)
                utilization = gpu_stats.get('utilization_percent', 0)
                vram_used = gpu_stats.get('vram_used_mb', 0)
                vram_total = gpu_stats.get('vram_total_mb', 0)
                content.append(f"    Available: {available}\n", style="dim")
                content.append(f"    Utilization: {utilization:.1f}%\n", style="dim")
                if vram_total > 0:
                    vram_pct = (vram_used / vram_total * 100)
                    content.append(f"    VRAM: {vram_used:,}/{vram_total:,} MB ({vram_pct:.0f}%)\n", style="dim")
            
            if token_usage:
                content.append(f"\n  Token Usage:\n", style="bold")
                for council, tokens in list(token_usage.items())[:5]:
                    content.append(f"    {council}: {tokens:,}\n", style="dim")
            
            if resource_budgets:
                content.append(f"\n  Resource Budgets:\n", style="bold")
                for council, budget in list(resource_budgets.items())[:5]:
                    is_exceeded = budget.get('is_exceeded', False)
                    status_icon = "🚫" if is_exceeded else "✓"
                    status_style = "red" if is_exceeded else "green"
                    content.append(f"    {status_icon} {council}: ", style=status_style)
                    tokens_used = budget.get('tokens_used', 0)
                    max_tokens = budget.get('max_tokens', 0)
                    if max_tokens > 0:
                        pct = (tokens_used / max_tokens * 100)
                        content.append(f"{tokens_used:,}/{max_tokens:,} ({pct:.0f}%)\n", style="dim")
                    else:
                        content.append(f"{tokens_used:,}\n", style="dim")
            
            content.append("─" * 50, style="dim")
            
            panel = Panel(content, title="📊 Resource Management", border_style="yellow", box=box.ROUNDED)
            self._console.print(panel)
        else:
            print(f"\n{'=' * 50}")
            print("📊 Resource Management")
            print(f"{'=' * 50}")
            if contraction_mode:
                print("  🔻 CONTRACTION MODE ACTIVE")
            if time_remaining is not None and time_total is not None:
                time_pct = (time_remaining / time_total * 100) if time_total > 0 else 0
                print(f"  Time Budget: {time_remaining:.1f}/{time_total:.1f} min ({time_pct:.0f}%)")
            if gpu_stats:
                print(f"\n  GPU Status:")
                print(f"    Available: {gpu_stats.get('available_gpus', 0)}")
                print(f"    Utilization: {gpu_stats.get('utilization_percent', 0):.1f}%")
            if token_usage:
                print(f"\n  Token Usage:")
                for council, tokens in list(token_usage.items())[:5]:
                    print(f"    {council}: {tokens:,}")
            print("=" * 50)
    
    def log_security_execution_panel(
        self,
        security_issues: Optional[List[Dict[str, Any]]] = None,
        execution_environment: Optional[str] = None,
        safety_checks: Optional[List[str]] = None,
        code_scanned: bool = False,
        execution_allowed: bool = True
    ) -> None:
        """
        Log security and execution panel.
        
        Args:
            security_issues: List of security issues found
            execution_environment: Execution environment (docker, browser, local)
            safety_checks: List of safety checks performed
            code_scanned: Whether code was scanned
            execution_allowed: Whether execution was allowed
        """
        if not self.enabled:
            return
        
        security_issues = security_issues or []
        safety_checks = safety_checks or []
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("🔒 Security & Execution\n", style="bold red")
            content.append("─" * 50 + "\n", style="dim")
            
            content.append(f"  Code Scanned: {'Yes' if code_scanned else 'No'}\n", style="dim")
            content.append(f"  Execution Allowed: ", style="dim")
            content.append(f"{'Yes' if execution_allowed else 'No'}\n", 
                          style="green" if execution_allowed else "red")
            
            if execution_environment:
                content.append(f"  Environment: {execution_environment}\n", style="cyan")
            
            if safety_checks:
                content.append(f"\n  Safety Checks:\n", style="bold")
                for check in safety_checks[:5]:
                    content.append(f"    ✓ {check}\n", style="green")
            
            if security_issues:
                content.append(f"\n  ⚠️  Security Issues ({len(security_issues)}):\n", style="bold yellow")
                for issue in security_issues[:5]:
                    risk_level = issue.get('risk_level', 'unknown')
                    category = issue.get('category', 'unknown')
                    description = issue.get('description', '')[:50]
                    risk_style = "red" if risk_level in ['high', 'critical'] else "yellow"
                    content.append(f"    [{risk_level.upper()}] {category}: {description}...\n", style=risk_style)
            
            content.append("─" * 50, style="dim")
            
            panel = Panel(content, title="🔒 Security & Execution", border_style="red", box=box.ROUNDED)
            self._console.print(panel)
        else:
            print(f"\n{'=' * 50}")
            print("🔒 Security & Execution")
            print(f"{'=' * 50}")
            print(f"  Code Scanned: {'Yes' if code_scanned else 'No'}")
            print(f"  Execution Allowed: {'Yes' if execution_allowed else 'No'}")
            if execution_environment:
                print(f"  Environment: {execution_environment}")
            if safety_checks:
                print(f"\n  Safety Checks:")
                for check in safety_checks[:5]:
                    print(f"    ✓ {check}")
            if security_issues:
                print(f"\n  ⚠️  Security Issues ({len(security_issues)}):")
                for issue in security_issues[:5]:
                    risk_level = issue.get('risk_level', 'unknown')
                    category = issue.get('category', 'unknown')
                    description = issue.get('description', '')[:50]
                    print(f"    [{risk_level.upper()}] {category}: {description}...")
            print("=" * 50)
    
    def log_phase_layer_summary(
        self,
        phase_name: str,
        layers: Dict[str, bool],
        duration_s: float,
        quality_score: Optional[float] = None
    ) -> None:
        """
        Log summary panel showing all layers for a completed phase.
        
        Args:
            phase_name: Name of the phase
            layers: Dict mapping layer names to whether they were active
            duration_s: Phase duration in seconds
            quality_score: Quality score if available
        """
        if not self.enabled:
            return
        
        active_layers = [name for name, active in layers.items() if active]
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append(f"✅ Phase Complete: {phase_name}\n", style="bold green")
            content.append("─" * 50 + "\n", style="dim")
            
            content.append(f"  Duration: {duration_s:.2f}s\n", style="dim")
            
            if quality_score is not None:
                score_style = "green" if quality_score > 0.7 else "yellow" if quality_score > 0.4 else "red"
                content.append(f"  Quality Score: ", style="dim")
                content.append(f"{quality_score:.2f}\n", style=score_style)
            
            content.append(f"\n  Active Layers ({len(active_layers)}):\n", style="bold")
            layer_icons = {
                "Model Selection": "🤖",
                "Memory Operations": "🧠",
                "Internet Usage": "🌐",
                "Step Execution": "📋",
                "Model Execution": "🤖",
                "Consensus Mechanism": "⚖️",
                "Multi-View": "🔀",
                "Resource Management": "📊",
                "Security & Execution": "🔒",
                "Arbiter": "⚖️"
            }
            for layer in active_layers:
                icon = layer_icons.get(layer, "•")
                content.append(f"    {icon} {layer}\n", style="cyan")
            
            content.append("─" * 50, style="dim")
            
            panel = Panel(content, title="📊 Phase Layer Summary", border_style="green", box=box.ROUNDED)
            self._console.print(panel)
        else:
            print(f"\n{'=' * 50}")
            print(f"✅ Phase Complete: {phase_name}")
            print(f"{'=' * 50}")
            print(f"  Duration: {duration_s:.2f}s")
            if quality_score is not None:
                print(f"  Quality Score: {quality_score:.2f}")
            print(f"\n  Active Layers ({len(active_layers)}):")
            for layer in active_layers:
                print(f"    • {layer}")
            print("=" * 50)
    
    # =========================================================================
    # ML PREDICTION & GOVERNANCE PANELS
    # =========================================================================
    
    def log_ml_predictions_panel(
        self,
        phase_name: str,
        cost_prediction: Optional[Dict[str, Any]] = None,
        risk_prediction: Optional[Dict[str, Any]] = None,
        web_search_prediction: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log ML predictions panel before phase execution.
        
        Displays predictions from all ML predictors in a unified panel:
        - Cost/Time: wall_time, gpu_seconds, vram_peak
        - Risk: retry_probability, expected_retries, failure_mode
        - Web Search: search_required, expected_queries, hallucination_risk
        
        Args:
            phase_name: Name of the phase being predicted
            cost_prediction: Cost/time prediction dict (from CostTimePrediction.to_dict())
            risk_prediction: Risk prediction dict (from PhaseRiskPrediction.to_dict())
            web_search_prediction: Web search prediction dict (from WebSearchPrediction.to_dict())
        """
        if not self.enabled:
            return
        
        # Skip if no predictions available
        if not any([cost_prediction, risk_prediction, web_search_prediction]):
            return
        
        def _confidence_bar(conf: float, width: int = 10) -> str:
            """Generate a visual confidence bar."""
            filled = int(conf * width)
            return "█" * filled + "░" * (width - filled)
        
        def _model_label(version: str, used_fallback: bool) -> str:
            """Generate model label with fallback indicator."""
            if used_fallback:
                return "fallback"
            return f"ML {version}" if version else "ML"
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("🤖 ML Predictions\n", style="bold magenta")
            content.append("─" * 50 + "\n", style="dim")
            
            # Cost/Time prediction
            if cost_prediction:
                version = cost_prediction.get("model_version", "")
                fallback = cost_prediction.get("used_fallback", False)
                conf = cost_prediction.get("confidence", 0.0)
                label = _model_label(version, fallback)
                
                style = "yellow" if fallback else "cyan"
                content.append(f"  ⏱️  Cost/Time ", style="bold")
                content.append(f"({label}, conf: {conf:.2f})\n", style=style)
                
                wall_time = cost_prediction.get("wall_time_seconds", 0)
                gpu_sec = cost_prediction.get("gpu_seconds", 0)
                vram = cost_prediction.get("vram_peak_mb", 0)
                content.append(f"     Wall: {wall_time:.1f}s   GPU: {gpu_sec:.1f}s   VRAM: {vram:,} MB\n", style="dim")
                content.append(f"     Confidence: [{_confidence_bar(conf)}] {conf:.0%}\n", style="dim")
                content.append("\n")
            
            # Risk prediction
            if risk_prediction:
                version = risk_prediction.get("model_version", "")
                fallback = risk_prediction.get("used_fallback", False)
                conf = risk_prediction.get("confidence", 0.0)
                label = _model_label(version, fallback)
                
                style = "yellow" if fallback else "cyan"
                content.append(f"  ⚠️  Risk ", style="bold")
                content.append(f"({label}, conf: {conf:.2f})\n", style=style)
                
                retry_prob = risk_prediction.get("retry_probability", 0) * 100
                expected = risk_prediction.get("expected_retries", 0)
                mode = risk_prediction.get("dominant_failure_mode", "unknown")
                content.append(f"     Retry Prob: {retry_prob:.0f}%   Expected: {expected:.1f}   Mode: {mode}\n", style="dim")
                content.append(f"     Confidence: [{_confidence_bar(conf)}] {conf:.0%}\n", style="dim")
                content.append("\n")
            
            # Web Search prediction
            if web_search_prediction:
                version = web_search_prediction.get("model_version", "")
                fallback = web_search_prediction.get("used_fallback", False)
                conf = web_search_prediction.get("confidence", 0.0)
                label = _model_label(version, fallback)
                
                style = "yellow" if fallback else "cyan"
                content.append(f"  🔍 Web Search ", style="bold")
                content.append(f"({label}, conf: {conf:.2f})\n", style=style)
                
                required = "Yes" if web_search_prediction.get("search_required", False) else "No"
                queries = web_search_prediction.get("expected_queries", 0)
                halluc_risk = web_search_prediction.get("hallucination_risk_without_search", 0) * 100
                content.append(f"     Required: {required}   Queries: {queries}   Halluc. Risk: {halluc_risk:.0f}%\n", style="dim")
                content.append(f"     Confidence: [{_confidence_bar(conf)}] {conf:.0%}\n", style="dim")
            
            content.append("─" * 50, style="dim")
            
            panel = Panel(
                content,
                title=f"🤖 ML Predictions: {phase_name}",
                border_style="magenta",
                box=box.ROUNDED
            )
            self._console.print(panel)
        else:
            # Plain text fallback
            print(f"\n{'=' * 55}")
            print(f"🤖 ML Predictions: {phase_name}")
            print(f"{'=' * 55}")
            
            if cost_prediction:
                version = cost_prediction.get("model_version", "")
                fallback = cost_prediction.get("used_fallback", False)
                conf = cost_prediction.get("confidence", 0.0)
                label = _model_label(version, fallback)
                
                print(f"\n  ⏱️  Cost/Time ({label}, conf: {conf:.2f})")
                wall_time = cost_prediction.get("wall_time_seconds", 0)
                gpu_sec = cost_prediction.get("gpu_seconds", 0)
                vram = cost_prediction.get("vram_peak_mb", 0)
                print(f"     Wall: {wall_time:.1f}s   GPU: {gpu_sec:.1f}s   VRAM: {vram:,} MB")
            
            if risk_prediction:
                version = risk_prediction.get("model_version", "")
                fallback = risk_prediction.get("used_fallback", False)
                conf = risk_prediction.get("confidence", 0.0)
                label = _model_label(version, fallback)
                
                print(f"\n  ⚠️  Risk ({label}, conf: {conf:.2f})")
                retry_prob = risk_prediction.get("retry_probability", 0) * 100
                expected = risk_prediction.get("expected_retries", 0)
                mode = risk_prediction.get("dominant_failure_mode", "unknown")
                print(f"     Retry Prob: {retry_prob:.0f}%   Expected: {expected:.1f}   Mode: {mode}")
            
            if web_search_prediction:
                version = web_search_prediction.get("model_version", "")
                fallback = web_search_prediction.get("used_fallback", False)
                conf = web_search_prediction.get("confidence", 0.0)
                label = _model_label(version, fallback)
                
                print(f"\n  🔍 Web Search ({label}, conf: {conf:.2f})")
                required = "Yes" if web_search_prediction.get("search_required", False) else "No"
                queries = web_search_prediction.get("expected_queries", 0)
                halluc_risk = web_search_prediction.get("hallucination_risk_without_search", 0) * 100
                print(f"     Required: {required}   Queries: {queries}   Halluc. Risk: {halluc_risk:.0f}%")
            
            print(f"{'=' * 55}")
    
    def log_ml_governance_panel(
        self,
        system_health: Optional[float] = None,
        predictor_status: Optional[Dict[str, Dict[str, Any]]] = None,
        recent_alerts: Optional[List[Dict[str, Any]]] = None,
        advisory_readiness: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log ML governance and influence monitoring panel.
        
        Displays system-wide ML monitoring metrics:
        - System health score
        - Per-predictor status and metrics
        - Recent drift alerts
        - Advisory mode readiness
        
        Args:
            system_health: Overall system health score (0-1)
            predictor_status: Dict mapping predictor names to their status reports
            recent_alerts: List of recent drift alert dicts
            advisory_readiness: Advisory mode readiness assessment
        """
        if not self.enabled:
            return
        
        # Skip if no data available
        if system_health is None and not predictor_status and not recent_alerts:
            return
        
        def _health_bar(score: float, width: int = 10) -> str:
            """Generate a visual health bar."""
            filled = int(score * width)
            return "█" * filled + "░" * (width - filled)
        
        def _health_status(score: float) -> Tuple[str, str]:
            """Get health status label and style."""
            if score >= 0.7:
                return "HEALTHY", "green"
            elif score >= 0.4:
                return "WARNING", "yellow"
            else:
                return "CRITICAL", "red"
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("📊 ML Governance\n", style="bold blue")
            content.append("─" * 50 + "\n", style="dim")
            
            # System health
            if system_health is not None:
                status_label, status_style = _health_status(system_health)
                content.append(f"  System Health: [{_health_bar(system_health)}] ", style="bold")
                content.append(f"{system_health:.2f} ", style="white")
                content.append(f"({status_label})\n", style=status_style)
                content.append("\n")
            
            # Per-predictor status
            if predictor_status:
                content.append("  Predictor Status:\n", style="bold")
                for pred_name, pred_report in predictor_status.items():
                    metrics = pred_report.get("metrics", {})
                    health_score = pred_report.get("health_score", 0.5)
                    status = pred_report.get("status", "unknown")
                    
                    # Icon based on status
                    if status == "healthy":
                        icon = "✓"
                        style = "green"
                    elif status == "warning":
                        icon = "⚠"
                        style = "yellow"
                    else:
                        icon = "✗"
                        style = "red"
                    
                    pred_count = metrics.get("prediction_count", 0)
                    fallback_rate = metrics.get("fallback_rate", 0) * 100
                    avg_conf = metrics.get("avg_confidence", 0)
                    
                    content.append(f"    {icon} ", style=style)
                    content.append(f"{pred_name}: ", style="cyan")
                    content.append(f"{pred_count} predictions, {fallback_rate:.0f}% fallback, conf {avg_conf:.2f}\n", style="dim")
                content.append("\n")
            
            # Advisory readiness
            if advisory_readiness:
                overall_score = advisory_readiness.get("overall_score", 0)
                is_ready = advisory_readiness.get("ready", False)
                ready_label = "Ready" if is_ready else "Not Ready"
                ready_style = "green" if is_ready else "yellow"
                
                content.append(f"  Advisory Readiness: ", style="bold")
                content.append(f"{overall_score:.2f}/1.00 ", style="white")
                content.append(f"({ready_label})\n", style=ready_style)
                content.append("\n")
            
            # Recent alerts
            if recent_alerts:
                content.append(f"  Alerts ({len(recent_alerts)}):\n", style="bold")
                for alert in recent_alerts[:3]:  # Show max 3
                    severity = alert.get("severity", 0)
                    drift_type = alert.get("drift_type", "unknown")
                    pred_name = alert.get("predictor_name", "unknown")
                    
                    if severity > 0.8:
                        icon = "🔴"
                        style = "red"
                    elif severity > 0.5:
                        icon = "🟡"
                        style = "yellow"
                    else:
                        icon = "🟢"
                        style = "green"
                    
                    content.append(f"    {icon} ", style=style)
                    content.append(f"{drift_type} ({pred_name}): severity {severity:.2f}\n", style="dim")
            else:
                content.append("  Alerts: ", style="bold")
                content.append("None ✓\n", style="green")
            
            content.append("─" * 50, style="dim")
            
            panel = Panel(
                content,
                title="📊 ML Governance Summary",
                border_style="blue",
                box=box.ROUNDED
            )
            self._console.print(panel)
        else:
            # Plain text fallback
            print(f"\n{'=' * 55}")
            print("📊 ML Governance Summary")
            print(f"{'=' * 55}")
            
            if system_health is not None:
                status_label, _ = _health_status(system_health)
                print(f"\n  System Health: [{_health_bar(system_health)}] {system_health:.2f} ({status_label})")
            
            if predictor_status:
                print("\n  Predictor Status:")
                for pred_name, pred_report in predictor_status.items():
                    metrics = pred_report.get("metrics", {})
                    status = pred_report.get("status", "unknown")
                    icon = "✓" if status == "healthy" else "⚠" if status == "warning" else "✗"
                    
                    pred_count = metrics.get("prediction_count", 0)
                    fallback_rate = metrics.get("fallback_rate", 0) * 100
                    avg_conf = metrics.get("avg_confidence", 0)
                    
                    print(f"    {icon} {pred_name}: {pred_count} predictions, {fallback_rate:.0f}% fallback, conf {avg_conf:.2f}")
            
            if advisory_readiness:
                overall_score = advisory_readiness.get("overall_score", 0)
                is_ready = advisory_readiness.get("ready", False)
                ready_label = "Ready" if is_ready else "Not Ready"
                print(f"\n  Advisory Readiness: {overall_score:.2f}/1.00 ({ready_label})")
            
            if recent_alerts:
                print(f"\n  Alerts ({len(recent_alerts)}):")
                for alert in recent_alerts[:3]:
                    severity = alert.get("severity", 0)
                    drift_type = alert.get("drift_type", "unknown")
                    pred_name = alert.get("predictor_name", "unknown")
                    icon = "🔴" if severity > 0.8 else "🟡" if severity > 0.5 else "🟢"
                    print(f"    {icon} {drift_type} ({pred_name}): severity {severity:.2f}")
            else:
                print("\n  Alerts: None ✓")
            
            print(f"{'=' * 55}")

    # =========================================================================
    # ENHANCED UX PANELS - RETRIEVAL, CONSENSUS, STEP PROGRESS
    # =========================================================================

    def log_retrieval_panel(
        self,
        query: str,
        source: str,
        results: List[Tuple[Dict[str, Any], float]],
        reranked: bool = False
    ) -> None:
        """
        Log RAG retrieval operation with query, source, and results.
        
        Args:
            query: The search query used
            source: Source of retrieval ("mission_rag", "global_rag", "general_knowledge")
            results: List of (document_dict, score) tuples
            reranked: Whether results were reranked
        """
        if not self.enabled:
            return
        
        # Truncate query for display
        query_display = query[:50] + "..." if len(query) > 50 else query
        result_count = len(results)
        rerank_str = "yes" if reranked else "no"
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("📚 Retrieval: ", style="bold")
            content.append(f'"{query_display}"\n', style="white")
            content.append(f"   Source: ", style="dim")
            content.append(f"{source}", style="cyan")
            content.append(f" │ {result_count} results │ reranked: {rerank_str}\n", style="dim")
            content.append("   " + "─" * 45 + "\n", style="dim")
            
            for doc, score in results[:5]:  # Show top 5
                # Extract meaningful text from document
                text = doc.get("text", doc.get("summary", doc.get("content", "")))[:60]
                if len(doc.get("text", doc.get("summary", doc.get("content", "")))) > 60:
                    text += "..."
                
                # Show source info if available
                doc_source = doc.get("source_doc", doc.get("artifact_type", doc.get("phase", "")))
                if doc_source:
                    text = f"{doc_source} - {text}"
                
                content.append(f"   {score:.2f}  ", style="white")
                content.append(f"{text}\n", style="dim")
            
            if result_count > 5:
                content.append(f"   ... and {result_count - 5} more\n", style="dim")
            
            self._console.print(content)
        else:
            print(f'📚 Retrieval: "{query_display}"')
            print(f"   Source: {source} │ {result_count} results │ reranked: {rerank_str}")
            print("   " + "─" * 45)
            
            for doc, score in results[:5]:
                text = doc.get("text", doc.get("summary", doc.get("content", "")))[:60]
                if len(doc.get("text", doc.get("summary", doc.get("content", "")))) > 60:
                    text += "..."
                doc_source = doc.get("source_doc", doc.get("artifact_type", doc.get("phase", "")))
                if doc_source:
                    text = f"{doc_source} - {text}"
                print(f"   {score:.2f}  {text}")
            
            if result_count > 5:
                print(f"   ... and {result_count - 5} more")

    def log_consensus_panel(
        self,
        council_name: str,
        winner: Optional[str] = None,
        winner_model: Optional[str] = None,
        confidence: float = 0.0,
        vote_counts: Optional[Dict[str, int]] = None,
        cluster_assignments: Optional[Dict[str, int]] = None,
        skipped: bool = False,
        skip_reason: str = ""
    ) -> None:
        """
        Log consensus/voting result or skip decision.
        
        Args:
            council_name: Name of the council
            winner: Winning output text (truncated for display)
            winner_model: Model that produced winning output
            confidence: Confidence in consensus
            vote_counts: Dict mapping cluster_id -> count
            cluster_assignments: Dict mapping model_name -> cluster_id
            skipped: Whether consensus was skipped
            skip_reason: Reason for skipping (if skipped)
        """
        if not self.enabled:
            return
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append("⚖️  Consensus: ", style="bold")
            content.append(f"{council_name}\n", style="cyan")
            
            if skipped:
                content.append(f"   Skipped: ", style="dim")
                content.append(f"{skip_reason}\n", style="yellow")
            else:
                content.append(f"   Winner: ", style="dim")
                content.append(f"{winner_model or 'unknown'}", style="cyan")
                content.append(f" │ confidence: {confidence:.2f}\n", style="dim")
                
                if cluster_assignments and vote_counts:
                    content.append("   " + "─" * 45 + "\n", style="dim")
                    
                    # Group models by cluster
                    clusters: Dict[int, List[str]] = {}
                    for model, cluster_id in cluster_assignments.items():
                        if cluster_id not in clusters:
                            clusters[cluster_id] = []
                        clusters[cluster_id].append(model)
                    
                    # Find winning cluster
                    winning_cluster = None
                    if winner_model and winner_model in cluster_assignments:
                        winning_cluster = cluster_assignments[winner_model]
                    
                    for cluster_id, models in sorted(clusters.items()):
                        count = len(models)
                        models_str = ", ".join(models)
                        is_winner = cluster_id == winning_cluster
                        
                        content.append(f"   Cluster {cluster_id + 1} ({count} model{'s' if count > 1 else ''}): ", style="dim")
                        content.append(f"{models_str}", style="cyan" if is_winner else "dim")
                        if is_winner:
                            content.append("  ← winner", style="green")
                        content.append("\n")
            
            self._console.print(content)
        else:
            print(f"⚖️  Consensus: {council_name}")
            
            if skipped:
                print(f"   Skipped: {skip_reason}")
            else:
                print(f"   Winner: {winner_model or 'unknown'} │ confidence: {confidence:.2f}")
                
                if cluster_assignments and vote_counts:
                    print("   " + "─" * 45)
                    
                    clusters: Dict[int, List[str]] = {}
                    for model, cluster_id in cluster_assignments.items():
                        if cluster_id not in clusters:
                            clusters[cluster_id] = []
                        clusters[cluster_id].append(model)
                    
                    winning_cluster = None
                    if winner_model and winner_model in cluster_assignments:
                        winning_cluster = cluster_assignments[winner_model]
                    
                    for cluster_id, models in sorted(clusters.items()):
                        count = len(models)
                        models_str = ", ".join(models)
                        is_winner = cluster_id == winning_cluster
                        suffix = "  ← winner" if is_winner else ""
                        print(f"   Cluster {cluster_id + 1} ({count} model{'s' if count > 1 else ''}): {models_str}{suffix}")

    def log_step_progress(
        self,
        step_idx: int,
        total_steps: int,
        step_name: str,
        status: str,
        model: str,
        duration_s: float = 0.0
    ) -> None:
        """
        Log step progress within a phase (inline, compact format).
        
        Args:
            step_idx: Current step index (1-based)
            total_steps: Total number of steps
            step_name: Name of the step
            status: "running", "completed", or "failed"
            model: Model being used
            duration_s: Duration in seconds (for completed/failed)
        """
        if not self.enabled:
            return
        
        # Truncate step name if too long
        step_display = step_name[:25] + "..." if len(step_name) > 25 else step_name
        model_display = model.split("/")[-1] if "/" in model else model  # Remove registry prefix
        
        if status == "running":
            status_str = "running..."
            status_style = "yellow"
        elif status == "completed":
            status_str = f"done ({duration_s:.1f}s) ✓"
            status_style = "green"
        else:  # failed
            status_str = f"failed ({duration_s:.1f}s) ✗"
            status_style = "red"
        
        if RICH_AVAILABLE and self._console:
            content = Text()
            content.append(f"  [{step_idx}/{total_steps}] ", style="dim")
            content.append(f"{step_display}", style="white")
            content.append(f" ─ ", style="dim")
            content.append(f"{model_display}", style="cyan")
            content.append(f" ─ ", style="dim")
            content.append(f"{status_str}", style=status_style)
            
            # Use carriage return for "running" to allow overwrite, newline for final states
            if status == "running":
                self._console.print(content, end="\r")
            else:
                self._console.print(content)
        else:
            line = f"  [{step_idx}/{total_steps}] {step_display} ─ {model_display} ─ {status_str}"
            if status == "running":
                print(line, end="\r", flush=True)
            else:
                print(line)


# Global verbose logger instance
verbose_logger = VerboseLogger()

# Re-export RICH_AVAILABLE for consumers
__all__ = ['VerboseLogger', 'verbose_logger', 'configure_verbose_logging', 'RICH_AVAILABLE']


def configure_verbose_logging(enabled: bool = False, full_mode: bool = False) -> VerboseLogger:
    """
    Configure the global verbose logger.
    
    Args:
        enabled: Whether verbose logging is enabled
        full_mode: If True, disable truncation
        
    Returns:
        The configured VerboseLogger instance
    """
    verbose_logger.configure(enabled=enabled, full_mode=full_mode)
    return verbose_logger

