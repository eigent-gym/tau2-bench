"""
DEPRECATED: This module has been superseded by voice_analysis.py.

All functionality from this module is now available in voice_analysis.py with
additional features:
- Batch processing of multiple results.json files
- VAD (Voice Activity Detection) latency analysis
- More detailed event-level analysis
- Consistent output format (raw.csv, analysis.csv, *.pdf)

To run voice/conversation dynamics analysis, use:
    python -m experiments.tau_voice.exp.voice_analysis --data-dir <path>

This module is kept for backwards compatibility but will be removed in a future version.
"""

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from tau2.data_model.message import Tick
from tau2.data_model.simulation import Results

# Issue deprecation warning when module is imported
warnings.warn(
    "voice_metrics module is deprecated. Use voice_analysis module instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Configuration
TICK_DURATION_SECONDS = 0.2  # Default tick duration
print(f"USING DEFAULT TICK DURATION: {TICK_DURATION_SECONDS} seconds")

# =============================================================================
# Data Models
# =============================================================================


class Speaker(str, Enum):
    """Enum for speaker roles."""

    USER = "user"
    AGENT = "agent"


class UserActionType(str, Enum):
    """Types of user turn-taking actions."""

    KEEP_TALKING = "keep_talking"
    STOP_TALKING = "stop_talking"
    GENERATE_MESSAGE = "generate_message"
    WAIT = "wait"
    BACKCHANNEL = "backchannel"
    UNKNOWN = "unknown"


@dataclass
class SpeechEvent:
    """Represents a contiguous speech activity event."""

    speaker: Speaker
    start_time: float  # Start time in seconds
    duration: float  # Duration in seconds
    start_tick: int = 0  # Starting tick index
    end_tick: int = 0  # Ending tick index (exclusive)

    @property
    def end_time(self) -> float:
        """End time of the speech event."""
        return self.start_time + self.duration

    def overlaps_with(self, other: "SpeechEvent") -> bool:
        """Check if this event overlaps with another event."""
        return self.start_time < other.end_time and other.start_time < self.end_time

    def get_overlap(self, other: "SpeechEvent") -> Optional["SpeechEvent"]:
        """Get the overlapping region between two events, if any."""
        if not self.overlaps_with(other):
            return None
        overlap_start = max(self.start_time, other.start_time)
        overlap_end = min(self.end_time, other.end_time)
        return SpeechEvent(
            speaker=Speaker.USER,  # Placeholder, represents overlap
            start_time=overlap_start,
            duration=overlap_end - overlap_start,
        )


@dataclass
class UserSpeechEvent(SpeechEvent):
    """User speech event with turn-taking action information."""

    action_type: UserActionType = UserActionType.UNKNOWN
    is_interruption: bool = False  # True if user interrupted agent
    is_backchannel: bool = False  # True if this was a backchannel


@dataclass
class UserInterruptionEvent:
    """Represents an interruption where USER started speaking while AGENT was talking."""

    start_time: float  # When user started interrupting
    agent_yield_time: float  # Time it took agent to stop after interruption started
    user_speech_duration: (
        float  # Total duration of user speech during/after interruption
    )
    is_backchannel: bool = False  # Was this a backchannel rather than interruption?
    agent_stopped: bool = True  # Did the agent stop speaking?


@dataclass
class AgentInterruptionEvent:
    """Represents an interruption where AGENT started speaking while USER was talking."""

    start_time: float  # When agent started interrupting
    user_yield_time: float  # Time it took user to stop after interruption started
    agent_speech_duration: (
        float  # Total duration of agent speech during/after interruption
    )
    user_stopped: bool = True  # Did the user stop speaking?


@dataclass
class ResponseLatencyEvent:
    """Represents the latency between user finishing and agent starting to speak."""

    user_end_time: float  # When user stopped speaking
    agent_start_time: float  # When agent started speaking
    latency: float  # Time gap between user end and agent start


@dataclass
class VADEvent:
    """Represents a VAD (Voice Activity Detection) event from the provider."""

    event_type: str  # "speech_started", "speech_stopped", or "interrupted"
    tick_idx: int  # Tick index where the event occurred
    time: float  # Time in seconds when the event occurred


@dataclass
class VADLatencyEvent:
    """Represents VAD detection latency for a single user speech start.

    Measures the delay between when the user actually starts speaking
    (based on contains_speech flag) and when the provider's VAD detects it.
    """

    user_speech_start_tick: (
        int  # Tick where user started speaking (contains_speech=True)
    )
    user_speech_start_time: float  # Time in seconds
    vad_detected_tick: int  # Tick where VAD speech_started event arrived
    vad_detected_time: float  # Time in seconds
    latency_ticks: int  # Delay in ticks
    latency_seconds: float  # Delay in seconds


@dataclass
class VADMissEvent:
    """Represents a missed VAD detection - user spoke but no VAD event received.

    This indicates the provider's VAD failed to detect user speech within
    the expected time window (e.g., within 5 seconds / 25 ticks).
    """

    user_speech_start_tick: int  # Tick where user started speaking
    user_speech_start_time: float  # Time in seconds
    user_speech_duration_ticks: int  # How long user spoke (in ticks)
    user_speech_duration_seconds: float  # How long user spoke (in seconds)


@dataclass
class TurnTakingMetrics:
    """Comprehensive turn-taking metrics."""

    # Turn durations
    user_turn_durations: list[float] = field(default_factory=list)
    agent_turn_durations: list[float] = field(default_factory=list)

    # Response latencies
    response_latencies: list[ResponseLatencyEvent] = field(default_factory=list)

    # User interrupts agent
    user_interruption_events: list[UserInterruptionEvent] = field(default_factory=list)
    backchannel_events: list[UserInterruptionEvent] = field(default_factory=list)

    # Agent interrupts user
    agent_interruption_events: list[AgentInterruptionEvent] = field(
        default_factory=list
    )

    @property
    def user_turn_avg(self) -> float:
        return np.mean(self.user_turn_durations) if self.user_turn_durations else 0.0

    @property
    def user_turn_std(self) -> float:
        return np.std(self.user_turn_durations) if self.user_turn_durations else 0.0

    @property
    def agent_turn_avg(self) -> float:
        return np.mean(self.agent_turn_durations) if self.agent_turn_durations else 0.0

    @property
    def agent_turn_std(self) -> float:
        return np.std(self.agent_turn_durations) if self.agent_turn_durations else 0.0

    @property
    def response_latency_avg(self) -> float:
        latencies = [e.latency for e in self.response_latencies]
        return np.mean(latencies) if latencies else 0.0

    @property
    def response_latency_std(self) -> float:
        latencies = [e.latency for e in self.response_latencies]
        return np.std(latencies) if latencies else 0.0

    # User interrupts agent metrics
    @property
    def user_interruption_yield_times(self) -> list[float]:
        """Time for agent to stop after user interruption (excluding backchannels)."""
        return [
            e.agent_yield_time for e in self.user_interruption_events if e.agent_stopped
        ]

    @property
    def user_interruption_yield_avg(self) -> float:
        times = self.user_interruption_yield_times
        return np.mean(times) if times else 0.0

    @property
    def user_interruption_yield_std(self) -> float:
        times = self.user_interruption_yield_times
        return np.std(times) if times else 0.0

    @property
    def backchannel_agent_stopped_count(self) -> int:
        """Number of backchannels where agent incorrectly stopped."""
        return sum(1 for e in self.backchannel_events if e.agent_stopped)

    @property
    def backchannel_agent_continued_count(self) -> int:
        """Number of backchannels where agent correctly continued."""
        return sum(1 for e in self.backchannel_events if not e.agent_stopped)

    # Agent interrupts user metrics
    @property
    def agent_interruption_yield_times(self) -> list[float]:
        """Time for user to stop after agent interruption."""
        return [
            e.user_yield_time for e in self.agent_interruption_events if e.user_stopped
        ]

    @property
    def agent_interruption_yield_avg(self) -> float:
        times = self.agent_interruption_yield_times
        return np.mean(times) if times else 0.0

    @property
    def agent_interruption_yield_std(self) -> float:
        times = self.agent_interruption_yield_times
        return np.std(times) if times else 0.0


@dataclass
class SpeechActivitySummary:
    """Summary statistics for speech activity."""

    total_duration: float
    user_speaking_time: float
    agent_speaking_time: float
    overlap_time: float
    user_events: list[SpeechEvent]
    agent_events: list[SpeechEvent]
    overlap_events: list[SpeechEvent]
    turn_taking_metrics: Optional[TurnTakingMetrics] = None
    vad_events: list[VADEvent] = field(default_factory=list)
    vad_latencies: list[VADLatencyEvent] = field(default_factory=list)
    vad_misses: list[VADMissEvent] = field(default_factory=list)

    @property
    def vad_latency_avg(self) -> float:
        """Average VAD detection latency in seconds."""
        if not self.vad_latencies:
            return 0.0
        return np.mean([e.latency_seconds for e in self.vad_latencies])

    @property
    def vad_latency_std(self) -> float:
        """Standard deviation of VAD detection latency in seconds."""
        if not self.vad_latencies:
            return 0.0
        return np.std([e.latency_seconds for e in self.vad_latencies])

    @property
    def user_speaking_pct(self) -> float:
        """Percentage of time user is speaking."""
        return (
            100 * self.user_speaking_time / self.total_duration
            if self.total_duration > 0
            else 0
        )

    @property
    def agent_speaking_pct(self) -> float:
        """Percentage of time agent is speaking."""
        return (
            100 * self.agent_speaking_time / self.total_duration
            if self.total_duration > 0
            else 0
        )

    @property
    def overlap_pct(self) -> float:
        """Percentage of time with overlapping speech."""
        return (
            100 * self.overlap_time / self.total_duration
            if self.total_duration > 0
            else 0
        )

    @property
    def silence_time(self) -> float:
        """Time with no speech activity."""
        return (
            self.total_duration
            - self.user_speaking_time
            - self.agent_speaking_time
            + self.overlap_time
        )

    @property
    def silence_pct(self) -> float:
        """Percentage of time with no speech activity."""
        return (
            100 * self.silence_time / self.total_duration
            if self.total_duration > 0
            else 0
        )


# =============================================================================
# Event Extraction Functions
# =============================================================================


def get_user_action_type(tick: Tick) -> UserActionType:
    """Extract the user action type from a tick."""
    if tick.user_chunk and tick.user_chunk.turn_taking_action:
        action_str = tick.user_chunk.turn_taking_action.action
        try:
            return UserActionType(action_str)
        except ValueError:
            return UserActionType.UNKNOWN
    return UserActionType.UNKNOWN


def extract_speech_flags(ticks: list[Tick]) -> tuple[list[bool], list[bool]]:
    """
    Extract per-tick speech flags for user and agent.

    Args:
        ticks: List of Tick objects from a simulation

    Returns:
        Tuple of (user_speaking, agent_speaking) boolean lists
    """
    user_speaking = []
    agent_speaking = []

    for tick in ticks:
        user_speech = tick.user_chunk.contains_speech if tick.user_chunk else False
        agent_speech = tick.agent_chunk.contains_speech if tick.agent_chunk else False
        user_speaking.append(user_speech)
        agent_speaking.append(agent_speech)

    return user_speaking, agent_speaking


def extract_vad_events(ticks: list[Tick], tick_duration: float) -> list[VADEvent]:
    """
    Extract VAD (Voice Activity Detection) events from ticks.

    VAD events are stored in tick.agent_chunk.raw_data["vad_events"] and include:
    - "speech_started": User started speaking (detected by provider VAD)
    - "speech_stopped": User stopped speaking (detected by provider VAD)
    - "interrupted": User interrupted agent (Gemini-specific)

    Args:
        ticks: List of Tick objects from a simulation
        tick_duration: Duration of each tick in seconds

    Returns:
        List of VADEvent objects with timing information
    """
    vad_events = []

    for i, tick in enumerate(ticks):
        # VAD events are stored in agent_chunk.raw_data["vad_events"]
        if tick.agent_chunk and tick.agent_chunk.raw_data:
            raw_vad_events = tick.agent_chunk.raw_data.get("vad_events", [])
            for event_type in raw_vad_events:
                vad_events.append(
                    VADEvent(
                        event_type=event_type,
                        tick_idx=i,
                        time=i * tick_duration,
                    )
                )

    return vad_events


def compute_vad_latencies_and_misses(
    ticks: list[Tick],
    vad_events: list[VADEvent],
    tick_duration: float,
) -> tuple[list[VADLatencyEvent], list[VADMissEvent]]:
    """
    Compute VAD detection latencies and identify missed VAD detections.

    Measures the delay between when the user actually starts speaking
    (based on contains_speech transitioning to True) and when the
    provider's VAD emits a speech_started event. Also identifies cases
    where no VAD event was received within the expected window.

    Args:
        ticks: List of Tick objects from a simulation
        vad_events: List of VAD events extracted from ticks
        tick_duration: Duration of each tick in seconds

    Returns:
        Tuple of (latencies, misses):
        - latencies: List of VADLatencyEvent objects with latency measurements
        - misses: List of VADMissEvent objects for failed VAD detections
    """
    latencies = []
    misses = []

    # Get speech_started VAD events
    vad_starts = [e for e in vad_events if e.event_type == "speech_started"]

    # Find user speech start/end transitions (contains_speech: False -> True -> False)
    user_speech_segments = []  # List of (start_tick, end_tick)
    prev_speaking = False
    segment_start = None
    for i, tick in enumerate(ticks):
        is_speaking = tick.user_chunk.contains_speech if tick.user_chunk else False
        if is_speaking and not prev_speaking:
            segment_start = i
        elif not is_speaking and prev_speaking and segment_start is not None:
            user_speech_segments.append((segment_start, i))
            segment_start = None
        prev_speaking = is_speaking
    # Handle speech continuing to end
    if segment_start is not None:
        user_speech_segments.append((segment_start, len(ticks)))

    # If no VAD events, all user speech segments are misses
    if not vad_starts:
        for start_tick, end_tick in user_speech_segments:
            duration_ticks = end_tick - start_tick
            misses.append(
                VADMissEvent(
                    user_speech_start_tick=start_tick,
                    user_speech_start_time=start_tick * tick_duration,
                    user_speech_duration_ticks=duration_ticks,
                    user_speech_duration_seconds=duration_ticks * tick_duration,
                )
            )
        return latencies, misses

    # For each user speech segment, find the next VAD speech_started event
    vad_start_idx = 0
    for start_tick, end_tick in user_speech_segments:
        # Find the next VAD event at or after this user speech start
        while (
            vad_start_idx < len(vad_starts)
            and vad_starts[vad_start_idx].tick_idx < start_tick
        ):
            vad_start_idx += 1

        # Check if we have a matching VAD event within the window
        matched = False
        if vad_start_idx < len(vad_starts):
            vad_event = vad_starts[vad_start_idx]
            # Only match if the VAD event is reasonably close (within 25 ticks / 5 seconds)
            if vad_event.tick_idx - start_tick <= 25:
                latency_ticks = vad_event.tick_idx - start_tick
                latencies.append(
                    VADLatencyEvent(
                        user_speech_start_tick=start_tick,
                        user_speech_start_time=start_tick * tick_duration,
                        vad_detected_tick=vad_event.tick_idx,
                        vad_detected_time=vad_event.time,
                        latency_ticks=latency_ticks,
                        latency_seconds=latency_ticks * tick_duration,
                    )
                )
                vad_start_idx += 1  # Move to next VAD event for next user speech
                matched = True

        # If no match, record as a miss
        if not matched:
            duration_ticks = end_tick - start_tick
            misses.append(
                VADMissEvent(
                    user_speech_start_tick=start_tick,
                    user_speech_start_time=start_tick * tick_duration,
                    user_speech_duration_ticks=duration_ticks,
                    user_speech_duration_seconds=duration_ticks * tick_duration,
                )
            )

    return latencies, misses


def flags_to_events(
    flags: list[bool], speaker: Speaker, tick_duration: float
) -> list[SpeechEvent]:
    """
    Convert a boolean flag list to a list of SpeechEvent objects.

    Args:
        flags: Boolean list where True indicates speech activity
        speaker: The speaker role for these events
        tick_duration: Duration of each tick in seconds

    Returns:
        List of SpeechEvent objects representing contiguous speech regions
    """
    events = []
    start_tick = None

    for i, is_speaking in enumerate(flags):
        if is_speaking and start_tick is None:
            start_tick = i
        elif not is_speaking and start_tick is not None:
            events.append(
                SpeechEvent(
                    speaker=speaker,
                    start_time=start_tick * tick_duration,
                    duration=(i - start_tick) * tick_duration,
                    start_tick=start_tick,
                    end_tick=i,
                )
            )
            start_tick = None

    # Handle case where speaking continues to the end
    if start_tick is not None:
        events.append(
            SpeechEvent(
                speaker=speaker,
                start_time=start_tick * tick_duration,
                duration=(len(flags) - start_tick) * tick_duration,
                start_tick=start_tick,
                end_tick=len(flags),
            )
        )

    return events


def compute_overlap_events(
    user_flags: list[bool], agent_flags: list[bool], tick_duration: float
) -> list[SpeechEvent]:
    """
    Compute overlap events where both user and agent are speaking.

    Args:
        user_flags: Boolean list for user speech activity
        agent_flags: Boolean list for agent speech activity
        tick_duration: Duration of each tick in seconds

    Returns:
        List of SpeechEvent objects representing overlap regions
    """
    overlap_flags = [u and a for u, a in zip(user_flags, agent_flags)]
    # Use USER as placeholder speaker for overlaps
    return flags_to_events(overlap_flags, Speaker.USER, tick_duration)


def extract_user_speech_events(
    ticks: list[Tick], tick_duration: float
) -> list[UserSpeechEvent]:
    """
    Extract user speech events with action type information.

    Args:
        ticks: List of Tick objects
        tick_duration: Duration of each tick in seconds

    Returns:
        List of UserSpeechEvent objects with action information
    """
    events = []
    start_tick = None
    first_action: UserActionType = UserActionType.UNKNOWN

    for i, tick in enumerate(ticks):
        user_speech = tick.user_chunk.contains_speech if tick.user_chunk else False

        if user_speech and start_tick is None:
            start_tick = i
            first_action = get_user_action_type(tick)
        elif not user_speech and start_tick is not None:
            # Determine if this was an interruption or backchannel
            is_backchannel = first_action == UserActionType.BACKCHANNEL
            # Check if agent was speaking when user started
            agent_was_speaking = (
                ticks[start_tick].agent_chunk.contains_speech
                if ticks[start_tick].agent_chunk
                else False
            )
            is_interruption = agent_was_speaking and not is_backchannel

            events.append(
                UserSpeechEvent(
                    speaker=Speaker.USER,
                    start_time=start_tick * tick_duration,
                    duration=(i - start_tick) * tick_duration,
                    start_tick=start_tick,
                    end_tick=i,
                    action_type=first_action,
                    is_interruption=is_interruption,
                    is_backchannel=is_backchannel,
                )
            )
            start_tick = None

    # Handle case where speaking continues to the end
    if start_tick is not None:
        first_action = get_user_action_type(ticks[start_tick])
        is_backchannel = first_action == UserActionType.BACKCHANNEL
        agent_was_speaking = (
            ticks[start_tick].agent_chunk.contains_speech
            if ticks[start_tick].agent_chunk
            else False
        )
        is_interruption = agent_was_speaking and not is_backchannel

        events.append(
            UserSpeechEvent(
                speaker=Speaker.USER,
                start_time=start_tick * tick_duration,
                duration=(len(ticks) - start_tick) * tick_duration,
                start_tick=start_tick,
                end_tick=len(ticks),
                action_type=first_action,
                is_interruption=is_interruption,
                is_backchannel=is_backchannel,
            )
        )

    return events


def compute_turn_taking_metrics(
    ticks: list[Tick],
    user_events: list[SpeechEvent],
    agent_events: list[SpeechEvent],
    tick_duration: float,
) -> TurnTakingMetrics:
    """
    Compute detailed turn-taking metrics.

    Args:
        ticks: List of Tick objects
        user_events: List of user speech events
        agent_events: List of agent speech events
        tick_duration: Duration of each tick in seconds

    Returns:
        TurnTakingMetrics with all computed metrics
    """
    metrics = TurnTakingMetrics()

    # Turn durations
    metrics.user_turn_durations = [e.duration for e in user_events]
    metrics.agent_turn_durations = [e.duration for e in agent_events]

    # Extract user events with action info
    user_speech_events = extract_user_speech_events(ticks, tick_duration)

    # Compute response latencies (agent responding after user finishes)
    for user_event in user_events:
        # Find the next agent event that starts after this user event ends
        for agent_event in agent_events:
            if agent_event.start_time >= user_event.end_time:
                latency = agent_event.start_time - user_event.end_time
                metrics.response_latencies.append(
                    ResponseLatencyEvent(
                        user_end_time=user_event.end_time,
                        agent_start_time=agent_event.start_time,
                        latency=latency,
                    )
                )
                break

    # Compute interruption and backchannel events
    for user_event in user_speech_events:
        if not (user_event.is_interruption or user_event.is_backchannel):
            continue

        # Find if agent was speaking at user start
        start_idx = user_event.start_tick
        agent_speaking_at_start = (
            ticks[start_idx].agent_chunk.contains_speech
            if ticks[start_idx].agent_chunk
            else False
        )

        if not agent_speaking_at_start:
            continue

        # Find when agent stopped speaking after user started
        agent_stopped = False
        agent_yield_ticks = 0
        for i in range(
            start_idx, min(start_idx + 50, len(ticks))
        ):  # Look ahead up to 50 ticks
            agent_speaking = (
                ticks[i].agent_chunk.contains_speech if ticks[i].agent_chunk else False
            )
            if not agent_speaking:
                agent_stopped = True
                agent_yield_ticks = i - start_idx
                break

        agent_yield_time = agent_yield_ticks * tick_duration

        user_interruption_event = UserInterruptionEvent(
            start_time=user_event.start_time,
            agent_yield_time=agent_yield_time,
            user_speech_duration=user_event.duration,
            is_backchannel=user_event.is_backchannel,
            agent_stopped=agent_stopped,
        )

        if user_event.is_backchannel:
            metrics.backchannel_events.append(user_interruption_event)
        else:
            metrics.user_interruption_events.append(user_interruption_event)

    # Compute agent interruption events (agent starts speaking while user is talking)
    for agent_event in agent_events:
        start_time_ticks = int(agent_event.start_time / tick_duration)

        # Check if user was speaking when agent started
        if start_time_ticks >= len(ticks):
            continue

        user_speaking_at_start = (
            ticks[start_time_ticks].user_chunk.contains_speech
            if ticks[start_time_ticks].user_chunk
            else False
        )

        if not user_speaking_at_start:
            continue

        # This is an agent interruption - agent started speaking while user was talking
        # Find when user stopped speaking after agent started
        user_stopped = False
        user_yield_ticks = 0
        for i in range(
            start_time_ticks, min(start_time_ticks + 50, len(ticks))
        ):  # Look ahead up to 50 ticks
            user_speaking = (
                ticks[i].user_chunk.contains_speech if ticks[i].user_chunk else False
            )
            if not user_speaking:
                user_stopped = True
                user_yield_ticks = i - start_time_ticks
                break

        user_yield_time = user_yield_ticks * tick_duration

        agent_interruption_event = AgentInterruptionEvent(
            start_time=agent_event.start_time,
            user_yield_time=user_yield_time,
            agent_speech_duration=agent_event.duration,
            user_stopped=user_stopped,
        )
        metrics.agent_interruption_events.append(agent_interruption_event)

    return metrics


def extract_speech_activity(
    ticks: list[Tick], tick_duration: float = TICK_DURATION_SECONDS
) -> SpeechActivitySummary:
    """
    Extract complete speech activity summary from ticks.

    Args:
        ticks: List of Tick objects from a simulation
        tick_duration: Duration of each tick in seconds

    Returns:
        SpeechActivitySummary with all events and statistics
    """
    # Extract per-tick flags
    user_flags, agent_flags = extract_speech_flags(ticks)

    # Convert to events
    user_events = flags_to_events(user_flags, Speaker.USER, tick_duration)
    agent_events = flags_to_events(agent_flags, Speaker.AGENT, tick_duration)
    overlap_events = compute_overlap_events(user_flags, agent_flags, tick_duration)

    # Calculate times
    total_duration = len(ticks) * tick_duration
    user_time = sum(e.duration for e in user_events)
    agent_time = sum(e.duration for e in agent_events)
    overlap_time = sum(e.duration for e in overlap_events)

    # Compute turn-taking metrics
    turn_taking_metrics = compute_turn_taking_metrics(
        ticks, user_events, agent_events, tick_duration
    )

    # Extract VAD events and compute VAD latencies/misses
    vad_events = extract_vad_events(ticks, tick_duration)
    vad_latencies, vad_misses = compute_vad_latencies_and_misses(
        ticks, vad_events, tick_duration
    )

    return SpeechActivitySummary(
        total_duration=total_duration,
        user_speaking_time=user_time,
        agent_speaking_time=agent_time,
        overlap_time=overlap_time,
        user_events=user_events,
        agent_events=agent_events,
        overlap_events=overlap_events,
        turn_taking_metrics=turn_taking_metrics,
        vad_events=vad_events,
        vad_latencies=vad_latencies,
        vad_misses=vad_misses,
    )


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_speech_timeline(
    activity: SpeechActivitySummary,
    sim_idx: int = 0,
    figsize: tuple[float, float] = (14, 3),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a timeline visualization showing when user and agent are speaking.

    Args:
        activity: SpeechActivitySummary with extracted events
        sim_idx: Simulation index (for title)
        figsize: Figure size as (width, height)

    Returns:
        Tuple of (figure, axes)
    """
    import matplotlib.lines as mlines

    # Create figure with light background
    fig, ax = plt.subplots(figsize=figsize, facecolor="#fafafa")
    ax.set_facecolor("#fafafa")

    # Color palette (light theme)
    colors = {
        "user": "#2563eb",  # Vibrant blue
        "agent": "#dc2626",  # Vibrant red
        "overlap": "#7c3aed",  # Purple for overlap
        "grid": "#e5e7eb",  # Light gray grid
        "text": "#1f2937",  # Dark text
        "user_interrupt": "#f59e0b",  # Orange for user interruption
        "agent_interrupt": "#ec4899",  # Pink for agent interruption
        "backchannel": "#10b981",  # Green for backchannel
        "legend_bg": "#ffffff",  # White legend background
        "vad_started": "#06b6d4",  # Cyan for VAD speech_started
        "vad_stopped": "#8b5cf6",  # Violet for VAD speech_stopped
    }

    # Y positions - closer together for easier comparison
    y_user = 0.55
    y_agent = 0.0
    bar_height = 0.35

    # Convert events to broken_barh format: list of (start, duration)
    user_intervals = [(e.start_time, e.duration) for e in activity.user_events]
    agent_intervals = [(e.start_time, e.duration) for e in activity.agent_events]

    # Plot speaking regions
    if user_intervals:
        ax.broken_barh(
            user_intervals,
            (y_user - bar_height / 2, bar_height),
            facecolors=colors["user"],
            edgecolors="none",
            alpha=0.85,
        )

    if agent_intervals:
        ax.broken_barh(
            agent_intervals,
            (y_agent - bar_height / 2, bar_height),
            facecolors=colors["agent"],
            edgecolors="none",
            alpha=0.85,
        )

    # Highlight overlaps with vertical spans
    for event in activity.overlap_events:
        ax.axvspan(
            event.start_time,
            event.end_time,
            alpha=0.2,
            color=colors["overlap"],
            zorder=0,
        )

    # Add event markers from turn-taking metrics
    metrics = activity.turn_taking_metrics
    marker_y_offset = bar_height / 2 + 0.08

    if metrics:
        # User interruption markers (triangles pointing down at user track)
        for event in metrics.user_interruption_events:
            ax.plot(
                event.start_time,
                y_user + marker_y_offset,
                marker="v",
                color=colors["user_interrupt"],
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=10,
            )

        # Agent interruption markers (triangles pointing up at agent track)
        for event in metrics.agent_interruption_events:
            ax.plot(
                event.start_time,
                y_agent - marker_y_offset,
                marker="^",
                color=colors["agent_interrupt"],
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=10,
            )

        # Backchannel markers (circles at user track)
        for event in metrics.backchannel_events:
            ax.plot(
                event.start_time,
                y_user + marker_y_offset,
                marker="o",
                color=colors["backchannel"],
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=10,
            )

    # Add VAD event markers (diamonds on a separate track below agent)
    y_vad = -0.25  # Position below agent track
    for vad_event in activity.vad_events:
        if vad_event.event_type == "speech_started":
            ax.plot(
                vad_event.time,
                y_vad,
                marker="D",
                color=colors["vad_started"],
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=10,
            )
        elif vad_event.event_type == "speech_stopped":
            ax.plot(
                vad_event.time,
                y_vad,
                marker="s",
                color=colors["vad_stopped"],
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=10,
            )
        elif vad_event.event_type == "interrupted":
            # Show Gemini interruption events
            ax.plot(
                vad_event.time,
                y_vad,
                marker="X",
                color=colors["vad_started"],
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=10,
            )

    # Add subtle grid
    ax.set_xlim(0, activity.total_duration)
    ax.set_ylim(-0.35, 0.95)

    # Grid lines
    for t in np.arange(0, activity.total_duration, 5):
        ax.axvline(t, color=colors["grid"], linewidth=0.5, zorder=0)

    # Styling
    ax.set_yticks([y_agent, y_user])
    ax.set_yticklabels(
        ["Agent", "User"], fontsize=11, fontweight="medium", color=colors["text"]
    )
    ax.set_xlabel("Time (seconds)", fontsize=10, color=colors["text"], labelpad=8)

    # Add horizontal reference lines
    ax.axhline(y_user, color=colors["grid"], linewidth=0.8, linestyle="-", alpha=0.4)
    ax.axhline(y_agent, color=colors["grid"], linewidth=0.8, linestyle="-", alpha=0.4)

    # Title
    ax.set_title(
        f"Speech Activity Timeline — Simulation {sim_idx}",
        fontsize=13,
        fontweight="bold",
        color=colors["text"],
        pad=10,
    )

    # Clean up spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=colors["text"], which="both")

    # Legend with event markers
    legend_elements = [
        mpatches.Patch(facecolor=colors["user"], alpha=0.85, label="User Speaking"),
        mpatches.Patch(facecolor=colors["agent"], alpha=0.85, label="Agent Speaking"),
        mpatches.Patch(facecolor=colors["overlap"], alpha=0.3, label="Overlap"),
        mlines.Line2D(
            [],
            [],
            color=colors["user_interrupt"],
            marker="v",
            linestyle="None",
            markersize=7,
            label="User Interrupts",
        ),
        mlines.Line2D(
            [],
            [],
            color=colors["agent_interrupt"],
            marker="^",
            linestyle="None",
            markersize=7,
            label="Agent Interrupts",
        ),
        mlines.Line2D(
            [],
            [],
            color=colors["backchannel"],
            marker="o",
            linestyle="None",
            markersize=6,
            label="Backchannel",
        ),
        mlines.Line2D(
            [],
            [],
            color=colors["vad_started"],
            marker="D",
            linestyle="None",
            markersize=5,
            label="VAD Start",
        ),
        mlines.Line2D(
            [],
            [],
            color=colors["vad_stopped"],
            marker="s",
            linestyle="None",
            markersize=5,
            label="VAD Stop",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        framealpha=0.95,
        facecolor=colors["legend_bg"],
        edgecolor=colors["grid"],
        fontsize=8,
        labelcolor=colors["text"],
        ncol=2,
    )

    # Stats annotation
    stats_text = (
        f"Duration: {activity.total_duration:.1f}s  |  "
        f"User: {activity.user_speaking_time:.1f}s ({activity.user_speaking_pct:.1f}%)  |  "
        f"Agent: {activity.agent_speaking_time:.1f}s ({activity.agent_speaking_pct:.1f}%)  |  "
        f"Overlap: {activity.overlap_time:.1f}s ({activity.overlap_pct:.1f}%)"
    )
    fig.text(
        0.5,
        0.02,
        stats_text,
        fontsize=9,
        color=colors["text"],
        alpha=0.8,
        ha="center",
        va="bottom",
        fontfamily="monospace",
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig, ax


def print_turn_taking_metrics(metrics: TurnTakingMetrics) -> None:
    """Print turn-taking metrics in a formatted way."""
    print("\n" + "=" * 60)
    print("TURN-TAKING METRICS")
    print("=" * 60)

    print("\n📊 Turn Durations:")
    print(f"  User turns:  {metrics.user_turn_avg:.2f}s ± {metrics.user_turn_std:.2f}s")
    print(f"               (n={len(metrics.user_turn_durations)})")
    print(
        f"  Agent turns: {metrics.agent_turn_avg:.2f}s ± {metrics.agent_turn_std:.2f}s"
    )
    print(f"               (n={len(metrics.agent_turn_durations)})")

    print("\n⏱️  Response Latency (agent response after user stops):")
    print(
        f"  Average: {metrics.response_latency_avg:.2f}s ± {metrics.response_latency_std:.2f}s"
    )
    print(f"  (n={len(metrics.response_latencies)})")

    print("\n🔄 User Interrupts Agent:")
    print(f"  Total interruptions: {len(metrics.user_interruption_events)}")
    if metrics.user_interruption_events:
        print(
            f"  Agent yield time: {metrics.user_interruption_yield_avg:.2f}s ± {metrics.user_interruption_yield_std:.2f}s"
        )
        stopped = sum(1 for e in metrics.user_interruption_events if e.agent_stopped)
        print(f"  Agent stopped: {stopped}/{len(metrics.user_interruption_events)}")

    print("\n🔄 Agent Interrupts User:")
    print(f"  Total interruptions: {len(metrics.agent_interruption_events)}")
    if metrics.agent_interruption_events:
        print(
            f"  User yield time: {metrics.agent_interruption_yield_avg:.2f}s ± {metrics.agent_interruption_yield_std:.2f}s"
        )
        user_stopped = sum(
            1 for e in metrics.agent_interruption_events if e.user_stopped
        )
        print(
            f"  User stopped: {user_stopped}/{len(metrics.agent_interruption_events)}"
        )

    print("\n💬 Backchannels:")
    print(f"  Total backchannels: {len(metrics.backchannel_events)}")
    if metrics.backchannel_events:
        print(
            f"  Agent correctly continued: {metrics.backchannel_agent_continued_count}"
        )
        print(f"  Agent incorrectly stopped: {metrics.backchannel_agent_stopped_count}")


def activity_to_metrics_row(
    activity: SpeechActivitySummary, sim_idx: int, task_id: str
) -> dict:
    """Convert a SpeechActivitySummary to a flat dictionary for CSV export."""
    metrics = activity.turn_taking_metrics
    row = {
        "sim_idx": sim_idx,
        "task_id": task_id,
        # Basic metrics
        "total_duration": activity.total_duration,
        "user_speaking_time": activity.user_speaking_time,
        "user_speaking_pct": activity.user_speaking_pct,
        "agent_speaking_time": activity.agent_speaking_time,
        "agent_speaking_pct": activity.agent_speaking_pct,
        "overlap_time": activity.overlap_time,
        "overlap_pct": activity.overlap_pct,
        "silence_time": activity.silence_time,
        "silence_pct": activity.silence_pct,
        "user_events_count": len(activity.user_events),
        "agent_events_count": len(activity.agent_events),
        "overlap_events_count": len(activity.overlap_events),
    }

    if metrics:
        row.update(
            {
                # Turn durations
                "user_turn_avg": metrics.user_turn_avg,
                "user_turn_std": metrics.user_turn_std,
                "user_turn_count": len(metrics.user_turn_durations),
                "agent_turn_avg": metrics.agent_turn_avg,
                "agent_turn_std": metrics.agent_turn_std,
                "agent_turn_count": len(metrics.agent_turn_durations),
                # Response latency
                "response_latency_avg": metrics.response_latency_avg,
                "response_latency_std": metrics.response_latency_std,
                "response_latency_count": len(metrics.response_latencies),
                # User interrupts agent
                "user_interruption_count": len(metrics.user_interruption_events),
                "user_interruption_yield_avg": metrics.user_interruption_yield_avg,
                "user_interruption_yield_std": metrics.user_interruption_yield_std,
                "user_interruption_agent_stopped": sum(
                    1 for e in metrics.user_interruption_events if e.agent_stopped
                ),
                # Agent interrupts user
                "agent_interruption_count": len(metrics.agent_interruption_events),
                "agent_interruption_yield_avg": metrics.agent_interruption_yield_avg,
                "agent_interruption_yield_std": metrics.agent_interruption_yield_std,
                "agent_interruption_user_stopped": sum(
                    1 for e in metrics.agent_interruption_events if e.user_stopped
                ),
                # Backchannels
                "backchannel_count": len(metrics.backchannel_events),
                "backchannel_agent_continued": metrics.backchannel_agent_continued_count,
                "backchannel_agent_stopped": metrics.backchannel_agent_stopped_count,
            }
        )

    # Add VAD metrics
    vad_started = sum(
        1 for e in activity.vad_events if e.event_type == "speech_started"
    )
    vad_stopped = sum(
        1 for e in activity.vad_events if e.event_type == "speech_stopped"
    )
    row.update(
        {
            "vad_speech_started_count": vad_started,
            "vad_speech_stopped_count": vad_stopped,
            "vad_latency_avg": activity.vad_latency_avg,
            "vad_latency_std": activity.vad_latency_std,
            "vad_latency_count": len(activity.vad_latencies),
            "vad_miss_count": len(activity.vad_misses),
        }
    )

    return row


def create_simulation_report(
    activity: SpeechActivitySummary,
    sim_idx: int,
    task_id: str,
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    Create a comprehensive report figure with timeline and metrics.

    Args:
        activity: SpeechActivitySummary with extracted events
        sim_idx: Simulation index
        task_id: Task ID
        figsize: Figure size as (width, height)

    Returns:
        Figure with timeline plot and metrics text
    """
    import matplotlib.lines as mlines

    # Create figure with two subplots: timeline on top, metrics text below
    fig = plt.figure(figsize=figsize, facecolor="#fafafa")

    # Create grid spec for layout
    gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.2], hspace=0.25)

    # Timeline plot (top)
    ax_timeline = fig.add_subplot(gs[0])
    ax_timeline.set_facecolor("#fafafa")

    # Color palette
    colors = {
        "user": "#2563eb",
        "agent": "#dc2626",
        "overlap": "#7c3aed",
        "grid": "#e5e7eb",
        "text": "#1f2937",
        "legend_bg": "#ffffff",
        "user_interrupt": "#f59e0b",  # Orange for user interruption
        "agent_interrupt": "#ec4899",  # Pink for agent interruption
        "backchannel": "#10b981",  # Green for backchannel
        "vad_started": "#06b6d4",  # Cyan for VAD speech_started
        "vad_stopped": "#8b5cf6",  # Violet for VAD speech_stopped
    }

    # Y positions - closer together for easier comparison
    y_user = 0.55
    y_agent = 0.0
    bar_height = 0.35

    user_intervals = [(e.start_time, e.duration) for e in activity.user_events]
    agent_intervals = [(e.start_time, e.duration) for e in activity.agent_events]

    if user_intervals:
        ax_timeline.broken_barh(
            user_intervals,
            (y_user - bar_height / 2, bar_height),
            facecolors=colors["user"],
            edgecolors="none",
            alpha=0.85,
        )

    if agent_intervals:
        ax_timeline.broken_barh(
            agent_intervals,
            (y_agent - bar_height / 2, bar_height),
            facecolors=colors["agent"],
            edgecolors="none",
            alpha=0.85,
        )

    for event in activity.overlap_events:
        ax_timeline.axvspan(
            event.start_time,
            event.end_time,
            alpha=0.2,
            color=colors["overlap"],
            zorder=0,
        )

    # Add event markers
    metrics = activity.turn_taking_metrics
    marker_y_offset = bar_height / 2 + 0.08

    if metrics:
        # User interruption markers
        for event in metrics.user_interruption_events:
            ax_timeline.plot(
                event.start_time,
                y_user + marker_y_offset,
                marker="v",
                color=colors["user_interrupt"],
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=10,
            )

        # Agent interruption markers
        for event in metrics.agent_interruption_events:
            ax_timeline.plot(
                event.start_time,
                y_agent - marker_y_offset,
                marker="^",
                color=colors["agent_interrupt"],
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=10,
            )

        # Backchannel markers
        for event in metrics.backchannel_events:
            ax_timeline.plot(
                event.start_time,
                y_user + marker_y_offset,
                marker="o",
                color=colors["backchannel"],
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=10,
            )

    # Add VAD event markers (diamonds on a separate track below agent)
    y_vad = -0.25  # Position below agent track
    for vad_event in activity.vad_events:
        if vad_event.event_type == "speech_started":
            ax_timeline.plot(
                vad_event.time,
                y_vad,
                marker="D",
                color=colors["vad_started"],
                markersize=4,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=10,
            )
        elif vad_event.event_type == "speech_stopped":
            ax_timeline.plot(
                vad_event.time,
                y_vad,
                marker="s",
                color=colors["vad_stopped"],
                markersize=4,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=10,
            )
        elif vad_event.event_type == "interrupted":
            ax_timeline.plot(
                vad_event.time,
                y_vad,
                marker="X",
                color=colors["vad_started"],
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=10,
            )

    ax_timeline.set_xlim(0, activity.total_duration)
    ax_timeline.set_ylim(-0.35, 0.95)

    for t in np.arange(0, activity.total_duration, 5):
        ax_timeline.axvline(t, color=colors["grid"], linewidth=0.5, zorder=0)

    ax_timeline.set_yticks([y_agent, y_user])
    ax_timeline.set_yticklabels(
        ["Agent", "User"], fontsize=10, fontweight="medium", color=colors["text"]
    )
    ax_timeline.set_xlabel("Time (seconds)", fontsize=9, color=colors["text"])
    ax_timeline.axhline(y_user, color=colors["grid"], linewidth=0.8, alpha=0.4)
    ax_timeline.axhline(y_agent, color=colors["grid"], linewidth=0.8, alpha=0.4)

    ax_timeline.set_title(
        f"Speech Activity Timeline — Simulation {sim_idx} (Task: {task_id})",
        fontsize=11,
        fontweight="bold",
        color=colors["text"],
        pad=8,
    )

    for spine in ax_timeline.spines.values():
        spine.set_visible(False)
    ax_timeline.tick_params(colors=colors["text"], which="both")

    # Legend with event markers
    legend_elements = [
        mpatches.Patch(facecolor=colors["user"], alpha=0.85, label="User"),
        mpatches.Patch(facecolor=colors["agent"], alpha=0.85, label="Agent"),
        mpatches.Patch(facecolor=colors["overlap"], alpha=0.3, label="Overlap"),
        mlines.Line2D(
            [],
            [],
            color=colors["user_interrupt"],
            marker="v",
            linestyle="None",
            markersize=6,
            label="User Interrupts",
        ),
        mlines.Line2D(
            [],
            [],
            color=colors["agent_interrupt"],
            marker="^",
            linestyle="None",
            markersize=6,
            label="Agent Interrupts",
        ),
        mlines.Line2D(
            [],
            [],
            color=colors["backchannel"],
            marker="o",
            linestyle="None",
            markersize=5,
            label="Backchannel",
        ),
        mlines.Line2D(
            [],
            [],
            color=colors["vad_started"],
            marker="D",
            linestyle="None",
            markersize=4,
            label="VAD Start",
        ),
        mlines.Line2D(
            [],
            [],
            color=colors["vad_stopped"],
            marker="s",
            linestyle="None",
            markersize=4,
            label="VAD Stop",
        ),
    ]
    ax_timeline.legend(
        handles=legend_elements,
        loc="upper right",
        framealpha=0.95,
        facecolor=colors["legend_bg"],
        edgecolor=colors["grid"],
        fontsize=7,
        ncol=3,
    )

    # Metrics text (bottom)
    ax_metrics = fig.add_subplot(gs[1])
    ax_metrics.set_facecolor("#fafafa")
    ax_metrics.axis("off")

    metrics = activity.turn_taking_metrics

    # Build metrics text
    metrics_lines = []
    metrics_lines.append("─" * 80)
    metrics_lines.append("BASIC METRICS")
    metrics_lines.append("─" * 80)
    metrics_lines.append(f"Total Duration:     {activity.total_duration:.1f}s")
    metrics_lines.append(
        f"User Speaking:      {activity.user_speaking_time:.1f}s ({activity.user_speaking_pct:.1f}%)"
    )
    metrics_lines.append(
        f"Agent Speaking:     {activity.agent_speaking_time:.1f}s ({activity.agent_speaking_pct:.1f}%)"
    )
    metrics_lines.append(
        f"Overlap:            {activity.overlap_time:.1f}s ({activity.overlap_pct:.1f}%)"
    )
    metrics_lines.append(
        f"Silence:            {activity.silence_time:.1f}s ({activity.silence_pct:.1f}%)"
    )
    metrics_lines.append(
        f"Events:             User={len(activity.user_events)}, Agent={len(activity.agent_events)}, Overlap={len(activity.overlap_events)}"
    )
    metrics_lines.append("")
    metrics_lines.append("─" * 80)
    metrics_lines.append("TURN-TAKING METRICS")
    metrics_lines.append("─" * 80)

    if metrics:
        metrics_lines.append(
            f"User Turn Duration:     {metrics.user_turn_avg:.2f}s ± {metrics.user_turn_std:.2f}s (n={len(metrics.user_turn_durations)})"
        )
        metrics_lines.append(
            f"Agent Turn Duration:    {metrics.agent_turn_avg:.2f}s ± {metrics.agent_turn_std:.2f}s (n={len(metrics.agent_turn_durations)})"
        )
        metrics_lines.append(
            f"Response Latency:       {metrics.response_latency_avg:.2f}s ± {metrics.response_latency_std:.2f}s (n={len(metrics.response_latencies)})"
        )
        metrics_lines.append("")
        metrics_lines.append(
            f"User Interrupts Agent:  {len(metrics.user_interruption_events)} total"
        )
        if metrics.user_interruption_events:
            stopped = sum(
                1 for e in metrics.user_interruption_events if e.agent_stopped
            )
            metrics_lines.append(
                f"  Agent Yield Time:     {metrics.user_interruption_yield_avg:.2f}s ± {metrics.user_interruption_yield_std:.2f}s"
            )
            metrics_lines.append(
                f"  Agent Stopped:        {stopped}/{len(metrics.user_interruption_events)}"
            )
        metrics_lines.append("")
        metrics_lines.append(
            f"Agent Interrupts User:  {len(metrics.agent_interruption_events)} total"
        )
        if metrics.agent_interruption_events:
            user_stopped = sum(
                1 for e in metrics.agent_interruption_events if e.user_stopped
            )
            metrics_lines.append(
                f"  User Yield Time:      {metrics.agent_interruption_yield_avg:.2f}s ± {metrics.agent_interruption_yield_std:.2f}s"
            )
            metrics_lines.append(
                f"  User Stopped:         {user_stopped}/{len(metrics.agent_interruption_events)}"
            )
        metrics_lines.append("")
        metrics_lines.append(
            f"Backchannels:           {len(metrics.backchannel_events)} total"
        )
        if metrics.backchannel_events:
            metrics_lines.append(
                f"  Agent Continued:      {metrics.backchannel_agent_continued_count}"
            )
            metrics_lines.append(
                f"  Agent Stopped:        {metrics.backchannel_agent_stopped_count}"
            )

    # Add VAD events section
    if activity.vad_events:
        metrics_lines.append("")
        metrics_lines.append("─" * 80)
        metrics_lines.append("VAD EVENTS (from provider)")
        metrics_lines.append("─" * 80)
        vad_started = sum(
            1 for e in activity.vad_events if e.event_type == "speech_started"
        )
        vad_stopped = sum(
            1 for e in activity.vad_events if e.event_type == "speech_stopped"
        )
        vad_interrupted = sum(
            1 for e in activity.vad_events if e.event_type == "interrupted"
        )
        metrics_lines.append(f"Speech Started:     {vad_started}")
        metrics_lines.append(f"Speech Stopped:     {vad_stopped}")
        if vad_interrupted > 0:
            metrics_lines.append(f"Interrupted:        {vad_interrupted}")
        # VAD detection latency and misses
        metrics_lines.append("")
        if activity.vad_latencies:
            metrics_lines.append(
                f"VAD Detection Latency: {activity.vad_latency_avg:.3f}s ± {activity.vad_latency_std:.3f}s (n={len(activity.vad_latencies)})"
            )
        if activity.vad_misses:
            metrics_lines.append(
                f"[!] VAD Misses:         {len(activity.vad_misses)} (user spoke but no VAD event)"
            )

    metrics_text = "\n".join(metrics_lines)

    ax_metrics.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax_metrics.transAxes,
        fontsize=10,
        fontfamily="monospace",
        color=colors["text"],
        verticalalignment="top",
        horizontalalignment="left",
    )

    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    return fig


def analyze_all_simulations(
    results: Results,
    output_dir: str,
    tick_duration: float = TICK_DURATION_SECONDS,
) -> None:
    """
    Analyze all simulations and save results.

    Args:
        results: Results object containing all simulations
        output_dir: Directory to save outputs (will create figs/ subdirectory)
        tick_duration: Duration of each tick in seconds
    """
    import csv
    from pathlib import Path

    # Create output directory
    figs_dir = Path(output_dir) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing {len(results.simulations)} simulations...")
    print(f"Output directory: {figs_dir}")

    # Collect all metrics rows for CSV
    all_metrics_rows = []

    for i, sim in enumerate(results.simulations):
        if sim.ticks is None:
            print(f"  Simulation {i}: No ticks data, skipping...")
            continue

        print(f"  Processing simulation {i} (task: {sim.task_id})...")

        # Extract speech activity
        activity = extract_speech_activity(sim.ticks, tick_duration)

        # Collect metrics for CSV
        row = activity_to_metrics_row(activity, i, sim.task_id)
        all_metrics_rows.append(row)

        # Create and save PDF report
        fig = create_simulation_report(activity, i, sim.task_id)
        pdf_path = figs_dir / f"sim_{i:03d}_{sim.task_id[:8]}.pdf"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="#fafafa")
        plt.close(fig)

    # Save metrics CSV
    if all_metrics_rows:
        import pandas as pd

        csv_path = figs_dir / "metrics.csv"
        fieldnames = all_metrics_rows[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics_rows)
        print(f"\n✓ Saved metrics to: {csv_path}")

        # Create summary plots
        metrics_df = pd.DataFrame(all_metrics_rows)
        create_summary_plots(metrics_df, figs_dir)

    print(f"✓ Saved {len(all_metrics_rows)} simulation reports to: {figs_dir}")


def create_summary_plots(
    metrics_df,
    output_dir: str,
) -> None:
    """
    Create summary plots showing average and std for key metrics.

    Args:
        metrics_df: pandas DataFrame with all simulation metrics
        output_dir: Directory to save plots
    """
    from pathlib import Path

    figs_dir = Path(output_dir)

    # Define metric groups for plotting
    metric_groups = {
        "speech_activity": {
            "title": "Speech Activity (% of total duration)",
            "metrics": [
                "user_speaking_pct",
                "agent_speaking_pct",
                "overlap_pct",
                "silence_pct",
            ],
            "labels": ["User Speaking", "Agent Speaking", "Overlap", "Silence"],
            "colors": ["#2563eb", "#dc2626", "#7c3aed", "#9ca3af"],
            "unit": "%",
        },
        "turn_duration": {
            "title": "Turn Duration (seconds)",
            "metrics": ["user_turn_avg", "agent_turn_avg"],
            "labels": ["User Turns", "Agent Turns"],
            "colors": ["#2563eb", "#dc2626"],
            "unit": "s",
        },
        "response_latency": {
            "title": "Response Latency (seconds)",
            "metrics": ["response_latency_avg"],
            "labels": ["Agent Response Latency"],
            "colors": ["#059669"],
            "unit": "s",
        },
        "interruptions": {
            "title": "Interruption Counts",
            "metrics": [
                "user_interruption_count",
                "agent_interruption_count",
                "backchannel_count",
            ],
            "labels": [
                "User Interrupts Agent",
                "Agent Interrupts User",
                "Backchannels",
            ],
            "colors": ["#2563eb", "#dc2626", "#f59e0b"],
            "unit": "count",
        },
        "interruption_yield": {
            "title": "Interruption Yield Time (seconds)",
            "metrics": ["user_interruption_yield_avg", "agent_interruption_yield_avg"],
            "labels": [
                "Agent Yield (user interrupts)",
                "User Yield (agent interrupts)",
            ],
            "colors": ["#2563eb", "#dc2626"],
            "unit": "s",
        },
    }

    # Create individual bar plots for each metric group
    for group_name, group_config in metric_groups.items():
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#fafafa")
        ax.set_facecolor("#fafafa")

        metrics = group_config["metrics"]
        labels = group_config["labels"]
        colors = group_config["colors"]

        # Calculate means and stds
        means = []
        stds = []
        for metric in metrics:
            if metric in metrics_df.columns:
                means.append(metrics_df[metric].mean())
                stds.append(metrics_df[metric].std())
            else:
                means.append(0)
                stds.append(0)

        x = np.arange(len(labels))
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            color=colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.5,
        )

        ax.set_xlabel("Metric", fontsize=11, color="#1f2937")
        ax.set_ylabel(f"Value ({group_config['unit']})", fontsize=11, color="#1f2937")
        ax.set_title(
            group_config["title"],
            fontsize=14,
            fontweight="bold",
            color="#1f2937",
            pad=15,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.annotate(
                f"{mean:.2f}±{std:.2f}",
                xy=(
                    bar.get_x() + bar.get_width() / 2,
                    height + std + 0.02 * max(means),
                ),
                ha="center",
                va="bottom",
                fontsize=9,
                color="#374151",
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#e5e7eb")
        ax.spines["bottom"].set_color("#e5e7eb")
        ax.tick_params(colors="#1f2937")
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        fig.savefig(
            figs_dir / f"summary_{group_name}.pdf",
            format="pdf",
            bbox_inches="tight",
            facecolor="#fafafa",
        )
        plt.close(fig)

    # Create a combined overview figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor="#fafafa")
    axes = axes.flatten()

    for idx, (group_name, group_config) in enumerate(metric_groups.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        ax.set_facecolor("#fafafa")

        metrics = group_config["metrics"]
        labels = group_config["labels"]
        colors = group_config["colors"]

        means = []
        stds = []
        for metric in metrics:
            if metric in metrics_df.columns:
                means.append(metrics_df[metric].mean())
                stds.append(metrics_df[metric].std())
            else:
                means.append(0)
                stds.append(0)

        x = np.arange(len(labels))
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            color=colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=1,
        )

        ax.set_ylabel(f"({group_config['unit']})", fontsize=9, color="#1f2937")
        ax.set_title(
            group_config["title"],
            fontsize=11,
            fontweight="bold",
            color="#1f2937",
            pad=8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#e5e7eb")
        ax.spines["bottom"].set_color("#e5e7eb")
        ax.tick_params(colors="#1f2937", labelsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    # Hide unused subplot
    if len(metric_groups) < len(axes):
        for idx in range(len(metric_groups), len(axes)):
            axes[idx].set_visible(False)

    fig.suptitle(
        f"Metrics Summary (n={len(metrics_df)} simulations)",
        fontsize=16,
        fontweight="bold",
        color="#1f2937",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(
        figs_dir / "summary_overview.pdf",
        format="pdf",
        bbox_inches="tight",
        facecolor="#fafafa",
    )
    plt.close(fig)

    # Create a box plot showing distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#fafafa")

    # Box plot 1: Speaking percentages
    ax = axes[0]
    ax.set_facecolor("#fafafa")
    box_data = [
        metrics_df["user_speaking_pct"],
        metrics_df["agent_speaking_pct"],
        metrics_df["overlap_pct"],
    ]
    bp = ax.boxplot(
        box_data, tick_labels=["User %", "Agent %", "Overlap %"], patch_artist=True
    )
    colors_box = ["#2563eb", "#dc2626", "#7c3aed"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(
        "Speaking Time Distribution", fontsize=12, fontweight="bold", color="#1f2937"
    )
    ax.set_ylabel("Percentage", fontsize=10, color="#1f2937")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    # Box plot 2: Turn durations
    ax = axes[1]
    ax.set_facecolor("#fafafa")
    box_data = [metrics_df["user_turn_avg"], metrics_df["agent_turn_avg"]]
    bp = ax.boxplot(
        box_data, tick_labels=["User Turns", "Agent Turns"], patch_artist=True
    )
    colors_box = ["#2563eb", "#dc2626"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(
        "Turn Duration Distribution", fontsize=12, fontweight="bold", color="#1f2937"
    )
    ax.set_ylabel("Duration (s)", fontsize=10, color="#1f2937")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    # Box plot 3: Response latency
    ax = axes[2]
    ax.set_facecolor("#fafafa")
    box_data = [metrics_df["response_latency_avg"]]
    bp = ax.boxplot(box_data, tick_labels=["Response Latency"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#059669")
    bp["boxes"][0].set_alpha(0.7)
    ax.set_title(
        "Response Latency Distribution", fontsize=12, fontweight="bold", color="#1f2937"
    )
    ax.set_ylabel("Duration (s)", fontsize=10, color="#1f2937")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(
        f"Metrics Distributions (n={len(metrics_df)} simulations)",
        fontsize=14,
        fontweight="bold",
        color="#1f2937",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(
        figs_dir / "summary_distributions.pdf",
        format="pdf",
        bbox_inches="tight",
        facecolor="#fafafa",
    )
    plt.close(fig)

    print(f"✓ Saved summary plots to: {figs_dir}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Analyze voice metrics from simulation results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze results in a specific directory
  python -m experiments.tau_voice.exp.voice_metrics data/exp/2026_01_15_15_58_10/retail_regular_openai

  # Specify tick duration (default: 0.2s)
  python -m experiments.tau_voice.exp.voice_metrics data/exp/my_experiment --tick-duration 0.2

  # Specify output directory
  python -m experiments.tau_voice.exp.voice_metrics data/exp/my_experiment --output /tmp/analysis
        """,
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Path to simulation directory containing results.json",
    )
    parser.add_argument(
        "--tick-duration",
        type=float,
        default=TICK_DURATION_SECONDS,
        help=f"Duration of each tick in seconds. Overridden by value in results.json if present (default: {TICK_DURATION_SECONDS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for analysis results (default: same as input directory)",
    )

    args = parser.parse_args()

    # Resolve paths
    input_dir = Path(args.directory)
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir

    result_file = input_dir / "results.json"
    output_dir = Path(args.output) if args.output else input_dir

    if not result_file.exists():
        print(f"Error: results.json not found at {result_file}")
        exit(1)

    results = Results.load(result_file)

    # Determine tick duration: prefer from results.json, fall back to CLI arg
    tick_duration = args.tick_duration
    if (
        results.info.audio_native_config is not None
        and results.info.audio_native_config.tick_duration_seconds is not None
    ):
        tick_duration = results.info.audio_native_config.tick_duration_seconds
        print(f"Loading results from: {result_file}")
        print(f"Tick duration: {tick_duration}s (from results.json)")
    else:
        print(f"Loading results from: {result_file}")
        print(f"Tick duration: {tick_duration}s (default/CLI)")
    print(f"Output directory: {output_dir}")
    print()

    # Analyze all simulations
    analyze_all_simulations(results, output_dir, tick_duration)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
