import time
from contextlib import contextmanager
from datetime import datetime

import pytz

# Define California timezone
CALIFORNIA_TZ = pytz.timezone("America/Los_Angeles")


class EventLog:
    def __init__(self, event_name):
        self.event_name = event_name
        self.start_time = None
        self.end_time = None
        self.child_events = {}

    def record_start_time(self):
        self.start_time = datetime.now(
            pytz.utc
        )  # Store in UTC for consistent timezone conversion

    def record_end_time(self):
        self.end_time = datetime.now(
            pytz.utc
        )  # Store in UTC for consistent timezone conversion

    def get_total_time(self):
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0

    def get_start_time(self):
        if self.start_time:
            # Format to milliseconds
            return self.start_time.astimezone(CALIFORNIA_TZ).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]
        return None

    def get_end_time(self):
        if self.end_time:
            # Format to milliseconds
            return self.end_time.astimezone(CALIFORNIA_TZ).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]
        return None

    def add_child_event(self, child_event):
        self.child_events[child_event.event_name] = child_event

    def get_child_events(self):
        return self.child_events


class LoggingWrapper:
    def __init__(self, lm_config):
        self.logging_dict = {}
        self.lm_config = lm_config
        self.current_pipeline_stage = None
        self.event_stack = []
        self.pipeline_stage_active = False
        self.pipeline_stage_stack = []  # Track nested pipeline stages

    def _pipeline_stage_start(self, pipeline_stage: str):
        # Save the current state if a pipeline stage is already active
        if self.pipeline_stage_active:
            self.pipeline_stage_stack.append(
                {
                    "stage": self.current_pipeline_stage,
                    "event_stack": self.event_stack.copy(),
                }
            )

        # Initialize new pipeline stage
        self.current_pipeline_stage = pipeline_stage
        if pipeline_stage not in self.logging_dict:
            self.logging_dict[pipeline_stage] = {
                "time_usage": {},
                "lm_usage": {},
                "lm_history": [],
                "query_count": 0,
                "total_wall_time": 0,
            }
        self.pipeline_stage_active = True
        self.event_stack = []  # Reset event stack for the new pipeline stage

    def _event_start(self, event_name: str):
        # Silently return if no pipeline stage is active
        if not self.pipeline_stage_active:
            return

        try:
            if not self.event_stack and self.current_pipeline_stage:
                # Top-level event (directly under the pipeline stage)
                if (
                    event_name
                    not in self.logging_dict[self.current_pipeline_stage]["time_usage"]
                ):
                    event = EventLog(event_name=event_name)
                    event.record_start_time()
                    self.logging_dict[self.current_pipeline_stage]["time_usage"][
                        event_name
                    ] = event
                    self.event_stack.append(event)
                else:
                    self.logging_dict[self.current_pipeline_stage]["time_usage"][
                        event_name
                    ].record_start_time()
                    self.event_stack.append(
                        self.logging_dict[self.current_pipeline_stage]["time_usage"][
                            event_name
                        ]
                    )
            elif self.event_stack:
                # Nested event (under another event)
                parent_event = self.event_stack[-1]
                if event_name not in parent_event.get_child_events():
                    event = EventLog(event_name=event_name)
                    event.record_start_time()
                    parent_event.add_child_event(event)
                    self.event_stack.append(event)
                else:
                    child_event = parent_event.get_child_events()[event_name]
                    child_event.record_start_time()
                    self.event_stack.append(child_event)
            else:
                # Create a new top-level event if there's no event stack
                event = EventLog(event_name=event_name)
                event.record_start_time()
                self.logging_dict[self.current_pipeline_stage]["time_usage"][
                    event_name
                ] = event
                self.event_stack.append(event)
        except Exception as e:
            # Log the error but don't disrupt workflow
            print(f"Warning: Error starting event '{event_name}': {e}")

    def _event_end(self, event_name: str):
        # Silently return if no pipeline stage is active
        if not self.pipeline_stage_active:
            return

        try:
            # If event stack is empty, just return rather than raising an error
            if not self.event_stack:
                return

            current_event = self.event_stack[-1]
            # Only pop if the event name matches to maintain proper nesting
            if current_event.event_name == event_name:
                current_event.record_end_time()
                self.event_stack.pop()
            else:
                # Try to find and end the event by name, but don't throw errors
                for i, event in enumerate(reversed(self.event_stack)):
                    if event.event_name == event_name:
                        event.record_end_time()
                        # Don't pop events we're skipping over - just record the end time
                        break
        except Exception as e:
            # Log the error but don't disrupt workflow
            print(f"Warning: Error ending event '{event_name}': {e}")

    def _pipeline_stage_end(self):
        if not self.pipeline_stage_active:
            return

        try:
            # Record LM usage
            self.logging_dict[self.current_pipeline_stage]["lm_usage"] = (
                self.lm_config.collect_and_reset_lm_usage()
                if hasattr(self.lm_config, "collect_and_reset_lm_usage")
                else {}
            )

            # Record LM history if the method exists
            if hasattr(self.lm_config, "collect_and_reset_lm_history"):
                self.logging_dict[self.current_pipeline_stage][
                    "lm_history"
                ] = self.lm_config.collect_and_reset_lm_history()

            # Restore previous pipeline stage if there was one
            if self.pipeline_stage_stack:
                previous = self.pipeline_stage_stack.pop()
                self.current_pipeline_stage = previous["stage"]
                self.event_stack = previous["event_stack"]
            else:
                self.pipeline_stage_active = False
                self.current_pipeline_stage = None
                self.event_stack = []
        except Exception as e:
            # Log the error but don't disrupt workflow
            print(f"Warning: Error ending pipeline stage: {e}")
            # Ensure we reset the state even if there's an error
            self.pipeline_stage_active = False
            self.event_stack = []

    def add_query_count(self, count):
        if not self.pipeline_stage_active:
            return  # Silently ignore if no pipeline stage active

        try:
            self.logging_dict[self.current_pipeline_stage]["query_count"] += count
        except Exception as e:
            # Log the error but don't disrupt workflow
            print(f"Warning: Error adding query count: {e}")

    @contextmanager
    def log_event(self, event_name):
        try:
            self._event_start(event_name)
            yield
        except Exception as e:
            # Re-raise the original exception after recording the event end
            original_exception = e
            raise_exception = True
        else:
            raise_exception = False
        finally:
            try:
                self._event_end(event_name)
            except Exception as e:
                print(f"Warning: Error in log_event cleanup for '{event_name}': {e}")

            if raise_exception:
                raise original_exception

    @contextmanager
    def log_pipeline_stage(self, pipeline_stage):
        start_time = time.time()
        try:
            self._pipeline_stage_start(pipeline_stage)
            yield
        except Exception as e:
            # Re-raise the original exception after recording pipeline end
            original_exception = e
            raise_exception = True
        else:
            raise_exception = False
        finally:
            try:
                # Calculate wall time before ending the pipeline stage
                if (
                    self.pipeline_stage_active
                    and self.current_pipeline_stage == pipeline_stage
                ):
                    self.logging_dict[pipeline_stage]["total_wall_time"] = (
                        time.time() - start_time
                    )
                self._pipeline_stage_end()
            except Exception as e:
                print(
                    f"Warning: Error in log_pipeline_stage cleanup for '{pipeline_stage}': {e}"
                )

            if raise_exception:
                raise original_exception

    def dump_logging_and_reset(self, reset_logging=True):
        log_dump = {}
        for pipeline_stage, pipeline_log in self.logging_dict.items():
            time_stamp_log = {}

            # Handle time usage entries
            for event_name, event in pipeline_log.get("time_usage", {}).items():
                if event:  # Ensure the event exists
                    time_stamp_log[event_name] = {
                        "total_time_seconds": event.get_total_time(),
                        "start_time": event.get_start_time(),
                        "end_time": event.get_end_time(),
                    }

            # Ensure all required keys exist
            for key in ["lm_usage", "lm_history", "query_count", "total_wall_time"]:
                if key not in pipeline_log:
                    pipeline_log[key] = {} if key in ["lm_usage", "lm_history"] else 0

            log_dump[pipeline_stage] = {
                "time_usage": time_stamp_log,
                "lm_usage": pipeline_log.get("lm_usage", {}),
                "lm_history": pipeline_log.get("lm_history", []),
                "query_count": pipeline_log.get("query_count", 0),
                "total_wall_time": pipeline_log.get("total_wall_time", 0),
            }

        if reset_logging:
            self.logging_dict.clear()
            self.pipeline_stage_active = False
            self.current_pipeline_stage = None
            self.event_stack = []
            self.pipeline_stage_stack = []

        return log_dump
