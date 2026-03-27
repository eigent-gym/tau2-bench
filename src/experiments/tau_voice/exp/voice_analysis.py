#!/usr/bin/env python3
"""
Voice Analysis for Tau Voice Experiments.

This module provides detailed voice/speech analysis following the same
architectural principles as performance_analysis.py:
- Each analysis outputs to a dedicated subdirectory
- Consistent pattern: raw.csv, analysis.csv, *.pdf
- Unified styling system

The foundation is the extraction of speech segments from simulation ticks,
with rich metadata for turn-taking, interruption, and timing analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from loguru import logger

if TYPE_CHECKING:
    import argparse

    import matplotlib.pyplot as plt
    import pandas as pd

    from tau2.data_model.simulation import Results, SimulationRun

# Import only the data models we need (avoid loading full tau2 package)
from tau2.data_model.message import Tick

# Delay Results import to avoid loading full registry
# Results is only needed in CLI, not for segment extraction


# =============================================================================
# Data Models for Audio Effects in Segments
# =============================================================================


@dataclass
class AudioEffectSegment:
    """
    Represents a contiguous audio effect segment during speech.

    Audio effects are applied to simulate realistic acoustic conditions:
    - Source effects: burst noise (e.g., cough, door slam)
    - Speech effects: vocal tics ("um", "uh"), non-directed speech ("one sec"),
                      dynamic muffling (hand over mic)

    Effects that span multiple consecutive ticks are merged into a single segment.
    """

    # Effect type classification
    effect_type: str  # "burst_noise", "vocal_tic", "non_directed_speech", "muffling"

    # Timing (tick-based)
    start_tick: int  # First tick where effect occurred
    end_tick: int  # Exclusive (first tick without effect)

    # Timing (seconds)
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0
    duration_sec: float = 0.0

    # Effect details
    text: Optional[str] = None  # Text content for vocal tics/non-directed speech
    file_path: Optional[str] = None  # Path to burst noise file (if applicable)

    # Whether effect was muffled (non-directed speech is typically muffled)
    is_muffled: bool = False

    @property
    def tick_count(self) -> int:
        """Number of ticks this effect spans."""
        return self.end_tick - self.start_tick


# Backwards compatibility alias
AudioEffectEvent = AudioEffectSegment


@dataclass
class FrameDropEvent:
    """
    Represents a frame drop event (simulated network packet loss).

    Frame drops are stored in channel_effects with:
    - frame_drops_enabled: Boolean indicating if frame drop is active
    - frame_drop_ms: Duration of the frame drop in milliseconds
    """

    tick_idx: int
    time_sec: float
    duration_ms: int  # Duration of the frame drop in ms
    during_speech: bool = False  # Whether user was speaking during this frame drop


# =============================================================================
# Data Models for Speech Segments
# =============================================================================


@dataclass
class SpeechSegment:
    """
    A contiguous speech segment extracted from ticks.

    This is the base class containing timing and content information
    common to both user and agent segments.
    """

    # Identification
    role: str  # "user" or "agent"

    # Tick-based timing
    start_tick: int
    end_tick: int  # exclusive (first non-speaking tick)

    # Precise timing from simulation (milliseconds)
    start_time_ms: Optional[float] = None  # From cumulative_user_audio_at_tick_start_ms
    end_time_ms: Optional[float] = None
    duration_ms: Optional[float] = None

    # Computed timing (seconds) - fallback if precise timing not available
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0
    duration_sec: float = 0.0

    # Content
    transcript: str = ""  # Concatenated proportional transcripts
    utterance_ids: List[str] = field(default_factory=list)

    # Context
    other_speaking_at_start: bool = (
        False  # Was the other party speaking when this started?
    )
    other_speaking_at_end: bool = False  # Was the other party speaking when this ended?

    @property
    def tick_count(self) -> int:
        """Number of ticks in this segment."""
        return self.end_tick - self.start_tick


@dataclass
class UserSpeechSegment(SpeechSegment):
    """
    User speech segment with turn-taking information.

    Extends SpeechSegment with user-specific metadata about
    the turn-taking action and interruption/backchannel classification.
    """

    # Turn-taking action (from first tick of segment)
    action: str = ""  # keep_talking, stop_talking, backchannel, wait, generate_message
    action_info: str = ""  # Human-readable description

    # Classification
    is_interruption: bool = False  # User started speaking while agent was speaking
    is_backchannel: bool = False  # This was classified as a backchannel

    # Performance timing (when available)
    interrupt_check_seconds: Optional[float] = None
    backchannel_check_seconds: Optional[float] = None
    llm_generation_seconds: Optional[float] = None
    tts_synthesis_seconds: Optional[float] = None

    # Audio effects applied during this segment
    audio_effects: List[AudioEffectEvent] = field(default_factory=list)

    # Aggregate effect flags (for quick filtering/analysis)
    has_burst_noise: bool = False
    has_vocal_tic: bool = False
    has_non_directed_speech: bool = False
    has_muffling: bool = False


@dataclass
class AgentSpeechSegment(SpeechSegment):
    """
    Agent speech segment with interruption and VAD information.

    Extends SpeechSegment with agent-specific metadata about
    being interrupted and VAD events during the segment.
    """

    # Interruption info
    was_interrupted: bool = False  # Was agent interrupted during this segment?
    truncated_audio_bytes: int = 0  # Audio bytes truncated due to interruption
    interruption_audio_start_ms: Optional[float] = (
        None  # When interruption was detected
    )

    # VAD events that occurred during this segment
    vad_events: List[str] = field(
        default_factory=list
    )  # speech_started, speech_stopped, interrupted

    # Audio effects applied during this segment
    audio_effects: List[AudioEffectEvent] = field(default_factory=list)

    # Aggregate effect flags (for quick filtering/analysis)
    has_burst_noise: bool = False
    has_vocal_tic: bool = False
    has_non_directed_speech: bool = False
    has_muffling: bool = False


# =============================================================================
# VAD (Voice Activity Detection) Data Models
# =============================================================================


@dataclass
class VADEvent:
    """
    Represents a VAD (Voice Activity Detection) event from the provider.

    VAD events are emitted by the audio provider when it detects speech
    activity changes. These are distinct from the contains_speech flags
    which are computed from actual audio content.
    """

    event_type: str  # "speech_started", "speech_stopped", or "interrupted"
    tick_idx: int  # Tick index where the event occurred
    time_sec: float  # Time in seconds when the event occurred


@dataclass
class VADLatencyEvent:
    """
    Measures VAD detection latency for a single user speech start.

    This captures the delay between when the user actually starts speaking
    (based on contains_speech flag becoming True) and when the provider's
    VAD detects it (speech_started event).
    """

    # Experiment metadata
    llm: str = ""
    domain: str = ""
    speech_complexity: str = ""
    provider: str = ""

    # Simulation metadata
    simulation_id: str = ""
    task_id: str = ""

    # User speech start info
    user_speech_start_tick: int = 0  # Tick where user started speaking
    user_speech_start_time_sec: float = 0.0  # Time in seconds

    # VAD detection info
    vad_detected_tick: int = 0  # Tick where VAD speech_started event arrived
    vad_detected_time_sec: float = 0.0  # Time in seconds

    # Latency measurements
    latency_ticks: int = 0  # Delay in ticks
    latency_sec: float = 0.0  # Delay in seconds


@dataclass
class VADMissEvent:
    """
    Represents a missed VAD detection - user spoke but no VAD event received.

    This indicates the provider's VAD failed to detect user speech within
    the expected time window (e.g., within 5 seconds / 25 ticks).
    """

    # Experiment metadata
    llm: str = ""
    domain: str = ""
    speech_complexity: str = ""
    provider: str = ""

    # Simulation metadata
    simulation_id: str = ""
    task_id: str = ""

    # User speech info
    user_speech_start_tick: int = 0  # Tick where user started speaking
    user_speech_start_time_sec: float = 0.0  # Time in seconds
    user_speech_duration_ticks: int = 0  # How long user spoke (in ticks)
    user_speech_duration_sec: float = 0.0  # How long user spoke (in seconds)


# =============================================================================
# Audio Effects Extraction Helpers
# =============================================================================


@dataclass
class _ActiveEffect:
    """Temporary state for tracking an active effect during segment extraction."""

    effect_type: str
    text: Optional[str] = None
    file_path: Optional[str] = None
    is_muffled: bool = False

    def matches(self, other: "_ActiveEffect") -> bool:
        """Check if this effect matches another (same type and details)."""
        return (
            self.effect_type == other.effect_type
            and self.text == other.text
            and self.file_path == other.file_path
        )


def _get_active_effects_from_chunk(chunk) -> List[_ActiveEffect]:
    """
    Get the list of currently active effects from a message chunk.

    This returns what effects are "on" in this tick, without timing info.
    Used to track effect segments across consecutive ticks.

    Args:
        chunk: A user_chunk or agent_chunk from a Tick

    Returns:
        List of _ActiveEffect representing currently active effects
    """
    active: List[_ActiveEffect] = []

    if chunk is None:
        return active

    # Extract source effects (burst noise, speech inserts from source)
    source_effects = getattr(chunk, "source_effects", None)
    if source_effects is not None:
        # Burst noise
        burst_file = getattr(source_effects, "burst_noise_file", None)
        if burst_file:
            active.append(
                _ActiveEffect(
                    effect_type="burst_noise",
                    file_path=burst_file,
                )
            )

        # Speech insert from source effects (less common)
        source_insert = getattr(source_effects, "speech_insert", None)
        if source_insert is not None:
            insert_type = getattr(source_insert, "type", "vocal_tic")
            insert_text = getattr(source_insert, "text", "")
            is_muffled = getattr(source_insert, "is_muffled", False)
            effect_type = (
                "vocal_tic" if insert_type == "vocal_tic" else "non_directed_speech"
            )

            active.append(
                _ActiveEffect(
                    effect_type=effect_type,
                    text=insert_text,
                    is_muffled=is_muffled,
                )
            )

    # Extract speech effects (vocal tics, non-directed speech, muffling)
    speech_effects = getattr(chunk, "speech_effects", None)
    if speech_effects is not None:
        # Dynamic muffling
        if getattr(speech_effects, "dynamic_muffling_enabled", False):
            active.append(
                _ActiveEffect(
                    effect_type="muffling",
                    is_muffled=True,
                )
            )

        # Speech insert (vocal tic or non-directed phrase)
        speech_insert = getattr(speech_effects, "speech_insert", None)
        if speech_insert is not None:
            insert_type = getattr(speech_insert, "type", "vocal_tic")
            insert_text = getattr(speech_insert, "text", "")
            is_muffled = getattr(speech_insert, "is_muffled", False)
            effect_type = (
                "vocal_tic" if insert_type == "vocal_tic" else "non_directed_speech"
            )

            active.append(
                _ActiveEffect(
                    effect_type=effect_type,
                    text=insert_text,
                    is_muffled=is_muffled,
                )
            )

    return active


def _start_effect_segment(
    active: _ActiveEffect,
    tick_idx: int,
    tick_duration_sec: float,
) -> AudioEffectSegment:
    """Create a new AudioEffectSegment from an active effect."""
    return AudioEffectSegment(
        effect_type=active.effect_type,
        start_tick=tick_idx,
        end_tick=tick_idx + 1,
        start_time_sec=tick_idx * tick_duration_sec,
        end_time_sec=(tick_idx + 1) * tick_duration_sec,
        duration_sec=tick_duration_sec,
        text=active.text,
        file_path=active.file_path,
        is_muffled=active.is_muffled,
    )


def _extend_effect_segment(
    segment: AudioEffectSegment,
    tick_idx: int,
    tick_duration_sec: float,
) -> None:
    """Extend an existing AudioEffectSegment by one tick."""
    segment.end_tick = tick_idx + 1
    segment.end_time_sec = (tick_idx + 1) * tick_duration_sec
    segment.duration_sec = segment.end_time_sec - segment.start_time_sec


def _update_audio_effect_segments(
    current_segments: List[AudioEffectSegment],
    prev_active: List[_ActiveEffect],
    curr_active: List[_ActiveEffect],
    tick_idx: int,
    tick_duration_sec: float,
) -> Tuple[List[AudioEffectSegment], List[_ActiveEffect]]:
    """
    Update audio effect segments based on current vs previous active effects.

    - Effects that continue: extend existing segment
    - Effects that start: create new segment
    - Effects that end: segment is already closed (end_tick was set)

    Returns:
        Tuple of (updated_segments, new_prev_active)
    """
    # Find which previous effects are still active (to extend)
    # and which current effects are new (to start)
    new_segments = list(current_segments)

    # Track which current effects matched a previous one
    matched_curr = [False] * len(curr_active)

    for prev in prev_active:
        # Find matching current effect
        for i, curr in enumerate(curr_active):
            if not matched_curr[i] and prev.matches(curr):
                matched_curr[i] = True
                # Extend the most recent segment of this type
                for seg in reversed(new_segments):
                    if (
                        seg.effect_type == prev.effect_type
                        and seg.end_tick == tick_idx  # Still open
                        and seg.text == prev.text
                        and seg.file_path == prev.file_path
                    ):
                        _extend_effect_segment(seg, tick_idx, tick_duration_sec)
                        break
                break

    # Start new segments for unmatched current effects
    for i, curr in enumerate(curr_active):
        if not matched_curr[i]:
            new_segments.append(
                _start_effect_segment(curr, tick_idx, tick_duration_sec)
            )

    return new_segments, curr_active


def _compute_effect_flags(
    segments: List[AudioEffectSegment],
) -> Tuple[bool, bool, bool, bool]:
    """Compute aggregate effect flags from segments."""
    has_burst = any(s.effect_type == "burst_noise" for s in segments)
    has_vocal_tic = any(s.effect_type == "vocal_tic" for s in segments)
    has_non_directed = any(s.effect_type == "non_directed_speech" for s in segments)
    has_muffling = any(s.effect_type == "muffling" for s in segments)
    return has_burst, has_vocal_tic, has_non_directed, has_muffling


def extract_out_of_turn_effects(
    ticks: List[Tick],
    tick_duration_sec: float = 0.2,
) -> List[AudioEffectSegment]:
    """
    Extract audio effects that occur outside of speech segments (out-of-turn).

    These are effects like non-directed speech ("Hold on", "One sec") or burst
    noise that occur when contains_speech=False. Such effects represent audio
    events during conversation gaps.

    Args:
        ticks: List of Tick objects from a simulation
        tick_duration_sec: Duration of each tick in seconds

    Returns:
        List of AudioEffectSegment for effects occurring during non-speech ticks
    """
    effect_segments: List[AudioEffectSegment] = []
    prev_active: List[_ActiveEffect] = []

    for i, tick in enumerate(ticks):
        user_chunk = tick.user_chunk

        # Only look at ticks where user is NOT speaking
        is_speaking = user_chunk is not None and getattr(
            user_chunk, "contains_speech", False
        )

        if is_speaking:
            # Reset tracking when speech starts
            prev_active = []
            continue

        # Get active effects from this non-speech tick
        curr_active = _get_active_effects_from_chunk(user_chunk)

        if not curr_active:
            prev_active = []
            continue

        # Update effect segments (same logic as speech segments)
        effect_segments, prev_active = _update_audio_effect_segments(
            effect_segments, prev_active, curr_active, i, tick_duration_sec
        )

    return effect_segments


def extract_frame_drops(
    ticks: List[Tick],
    tick_duration_sec: float = 0.2,
) -> List[FrameDropEvent]:
    """
    Extract frame drop events from simulation ticks.

    Frame drops simulate network packet loss and are stored in channel_effects.

    Args:
        ticks: List of Tick objects from a simulation
        tick_duration_sec: Duration of each tick in seconds

    Returns:
        List of FrameDropEvent objects
    """
    frame_drops: List[FrameDropEvent] = []

    for i, tick in enumerate(ticks):
        user_chunk = tick.user_chunk
        if user_chunk is None:
            continue

        channel_effects = getattr(user_chunk, "channel_effects", None)
        if channel_effects is None:
            continue

        frame_drops_enabled = getattr(channel_effects, "frame_drops_enabled", False)
        if not frame_drops_enabled:
            continue

        frame_drop_ms = getattr(channel_effects, "frame_drop_ms", 0)
        if frame_drop_ms <= 0:
            continue

        during_speech = getattr(user_chunk, "contains_speech", False)

        frame_drops.append(
            FrameDropEvent(
                tick_idx=i,
                time_sec=i * tick_duration_sec,
                duration_ms=frame_drop_ms,
                during_speech=during_speech,
            )
        )

    return frame_drops


# =============================================================================
# Segment Extraction Functions
# =============================================================================


def extract_user_segments(
    ticks: List[Tick],
    tick_duration_sec: float = 0.2,
) -> List[UserSpeechSegment]:
    """
    Extract user speech segments from simulation ticks.

    Groups contiguous ticks where user.contains_speech=True into segments,
    enriched with turn-taking metadata.

    Args:
        ticks: List of Tick objects from a simulation
        tick_duration_sec: Duration of each tick in seconds (for fallback timing)

    Returns:
        List of UserSpeechSegment objects
    """
    segments: List[UserSpeechSegment] = []
    current_segment: Optional[UserSpeechSegment] = None
    prev_active_effects: List[_ActiveEffect] = []

    for i, tick in enumerate(ticks):
        user_chunk = tick.user_chunk
        has_speech = user_chunk is not None and getattr(
            user_chunk, "contains_speech", False
        )

        if has_speech:
            if current_segment is None:
                # Start new segment
                current_segment, prev_active_effects = _create_user_segment_start(
                    tick, i, ticks, tick_duration_sec
                )
            else:
                # Extend current segment
                prev_active_effects = _extend_user_segment(
                    current_segment, tick, i, tick_duration_sec, prev_active_effects
                )
        else:
            # No speech - finalize current segment if any
            if current_segment is not None:
                _finalize_segment(current_segment, i, tick, ticks, tick_duration_sec)
                segments.append(current_segment)
                current_segment = None
                prev_active_effects = []

    # Handle segment continuing to end
    if current_segment is not None:
        _finalize_segment(current_segment, len(ticks), None, ticks, tick_duration_sec)
        segments.append(current_segment)

    return segments


def _create_user_segment_start(
    tick: Tick,
    tick_idx: int,
    ticks: List[Tick],
    tick_duration_sec: float,
) -> Tuple[UserSpeechSegment, List[_ActiveEffect]]:
    """Create a new UserSpeechSegment at the start of speech."""
    user_chunk = tick.user_chunk

    # Get turn-taking action from first tick
    action = ""
    action_info = ""
    is_backchannel = False
    interrupt_check_sec = None
    backchannel_check_sec = None
    llm_gen_sec = None
    tts_sec = None

    if user_chunk and user_chunk.turn_taking_action:
        tta = user_chunk.turn_taking_action
        action = getattr(tta, "action", "") or ""
        action_info = getattr(tta, "info", "") or ""
        is_backchannel = action == "backchannel"
        interrupt_check_sec = getattr(tta, "interrupt_check_seconds", None)
        backchannel_check_sec = getattr(tta, "backchannel_check_seconds", None)
        llm_gen_sec = getattr(tta, "llm_generation_seconds", None)
        tts_sec = getattr(tta, "tts_synthesis_seconds", None)

    # Check if agent was speaking when user started (interruption detection)
    agent_speaking_at_start = False
    if tick.agent_chunk:
        agent_speaking_at_start = getattr(tick.agent_chunk, "contains_speech", False)

    is_interruption = agent_speaking_at_start and not is_backchannel

    # Get precise timing if available
    start_time_ms = None
    if tick.agent_chunk and tick.agent_chunk.raw_data:
        raw_data = tick.agent_chunk.raw_data
        if isinstance(raw_data, dict):
            start_time_ms = raw_data.get("cumulative_user_audio_at_tick_start_ms")

    # Get text content
    text = ""
    if user_chunk and user_chunk.content:
        text = user_chunk.content

    # Get utterance IDs
    utterance_ids = []
    if user_chunk and user_chunk.utterance_ids:
        utterance_ids = list(user_chunk.utterance_ids)

    # Extract audio effects - start new effect segments for all active effects
    curr_active = _get_active_effects_from_chunk(user_chunk)
    effect_segments: List[AudioEffectSegment] = []
    for active in curr_active:
        effect_segments.append(
            _start_effect_segment(active, tick_idx, tick_duration_sec)
        )

    segment = UserSpeechSegment(
        role="user",
        start_tick=tick_idx,
        end_tick=tick_idx + 1,  # Will be updated as segment extends
        start_time_ms=start_time_ms,
        start_time_sec=tick_idx * tick_duration_sec,
        transcript=text,
        utterance_ids=utterance_ids,
        other_speaking_at_start=agent_speaking_at_start,
        action=action,
        action_info=action_info,
        is_interruption=is_interruption,
        is_backchannel=is_backchannel,
        interrupt_check_seconds=interrupt_check_sec,
        backchannel_check_seconds=backchannel_check_sec,
        llm_generation_seconds=llm_gen_sec,
        tts_synthesis_seconds=tts_sec,
        audio_effects=effect_segments,
    )

    return segment, curr_active


def _extend_user_segment(
    segment: UserSpeechSegment,
    tick: Tick,
    tick_idx: int,
    tick_duration_sec: float,
    prev_active: List[_ActiveEffect],
) -> List[_ActiveEffect]:
    """Extend an existing UserSpeechSegment with data from a new tick."""
    segment.end_tick = tick_idx + 1

    # Append text content
    user_chunk = tick.user_chunk
    if user_chunk and user_chunk.content:
        segment.transcript += user_chunk.content

    # Collect unique utterance IDs
    if user_chunk and user_chunk.utterance_ids:
        for uid in user_chunk.utterance_ids:
            if uid not in segment.utterance_ids:
                segment.utterance_ids.append(uid)

    # Update audio effect segments
    curr_active = _get_active_effects_from_chunk(user_chunk)
    segment.audio_effects, new_prev = _update_audio_effect_segments(
        segment.audio_effects, prev_active, curr_active, tick_idx, tick_duration_sec
    )

    return new_prev


def extract_agent_segments(
    ticks: List[Tick],
    tick_duration_sec: float = 0.2,
) -> List[AgentSpeechSegment]:
    """
    Extract agent speech segments from simulation ticks.

    Groups contiguous ticks where agent.contains_speech=True into segments,
    enriched with interruption and VAD metadata.

    Args:
        ticks: List of Tick objects from a simulation
        tick_duration_sec: Duration of each tick in seconds (for fallback timing)

    Returns:
        List of AgentSpeechSegment objects
    """
    segments: List[AgentSpeechSegment] = []
    current_segment: Optional[AgentSpeechSegment] = None
    prev_active_effects: List[_ActiveEffect] = []

    for i, tick in enumerate(ticks):
        agent_chunk = tick.agent_chunk
        has_speech = agent_chunk is not None and getattr(
            agent_chunk, "contains_speech", False
        )

        if has_speech:
            if current_segment is None:
                # Start new segment
                current_segment, prev_active_effects = _create_agent_segment_start(
                    tick, i, ticks, tick_duration_sec
                )
            else:
                # Extend current segment
                prev_active_effects = _extend_agent_segment(
                    current_segment, tick, i, tick_duration_sec, prev_active_effects
                )
        else:
            # No speech - finalize current segment if any
            if current_segment is not None:
                _finalize_segment(current_segment, i, tick, ticks, tick_duration_sec)
                segments.append(current_segment)
                current_segment = None
                prev_active_effects = []

    # Handle segment continuing to end
    if current_segment is not None:
        _finalize_segment(current_segment, len(ticks), None, ticks, tick_duration_sec)
        segments.append(current_segment)

    return segments


def _create_agent_segment_start(
    tick: Tick,
    tick_idx: int,
    ticks: List[Tick],
    tick_duration_sec: float,
) -> Tuple[AgentSpeechSegment, List[_ActiveEffect]]:
    """Create a new AgentSpeechSegment at the start of speech."""
    agent_chunk = tick.agent_chunk

    # Get precise timing if available
    start_time_ms = None
    if agent_chunk and agent_chunk.raw_data:
        raw_data = agent_chunk.raw_data
        if isinstance(raw_data, dict):
            start_time_ms = raw_data.get("cumulative_user_audio_at_tick_start_ms")

    # Check if user was speaking when agent started
    user_speaking_at_start = False
    if tick.user_chunk:
        user_speaking_at_start = getattr(tick.user_chunk, "contains_speech", False)

    # Get text content (proportional transcript from raw_data or content)
    text = ""
    if agent_chunk:
        if agent_chunk.content:
            text = agent_chunk.content
        elif agent_chunk.raw_data and isinstance(agent_chunk.raw_data, dict):
            text = agent_chunk.raw_data.get("proportional_transcript", "")

    # Get utterance IDs
    utterance_ids = []
    if agent_chunk and agent_chunk.utterance_ids:
        utterance_ids = list(agent_chunk.utterance_ids)

    # Get VAD events from this tick
    vad_events = []
    if agent_chunk and agent_chunk.raw_data and isinstance(agent_chunk.raw_data, dict):
        vad_events = list(agent_chunk.raw_data.get("vad_events", []))

    # Extract audio effects - start new effect segments for all active effects
    curr_active = _get_active_effects_from_chunk(agent_chunk)
    effect_segments: List[AudioEffectSegment] = []
    for active in curr_active:
        effect_segments.append(
            _start_effect_segment(active, tick_idx, tick_duration_sec)
        )

    segment = AgentSpeechSegment(
        role="agent",
        start_tick=tick_idx,
        end_tick=tick_idx + 1,
        start_time_ms=start_time_ms,
        start_time_sec=tick_idx * tick_duration_sec,
        transcript=text,
        utterance_ids=utterance_ids,
        other_speaking_at_start=user_speaking_at_start,
        vad_events=vad_events,
        audio_effects=effect_segments,
    )

    return segment, curr_active


def _extend_agent_segment(
    segment: AgentSpeechSegment,
    tick: Tick,
    tick_idx: int,
    tick_duration_sec: float,
    prev_active: List[_ActiveEffect],
) -> List[_ActiveEffect]:
    """Extend an existing AgentSpeechSegment with data from a new tick."""
    segment.end_tick = tick_idx + 1

    agent_chunk = tick.agent_chunk

    # Append text content
    if agent_chunk:
        if agent_chunk.content:
            segment.transcript += agent_chunk.content
        elif agent_chunk.raw_data and isinstance(agent_chunk.raw_data, dict):
            segment.transcript += agent_chunk.raw_data.get(
                "proportional_transcript", ""
            )

    # Collect unique utterance IDs
    if agent_chunk and agent_chunk.utterance_ids:
        for uid in agent_chunk.utterance_ids:
            if uid not in segment.utterance_ids:
                segment.utterance_ids.append(uid)

    # Collect VAD events
    if agent_chunk and agent_chunk.raw_data and isinstance(agent_chunk.raw_data, dict):
        for vad_event in agent_chunk.raw_data.get("vad_events", []):
            segment.vad_events.append(vad_event)

    # Check for interruption
    if agent_chunk and agent_chunk.raw_data and isinstance(agent_chunk.raw_data, dict):
        if agent_chunk.raw_data.get("was_truncated", False):
            segment.was_interrupted = True
            segment.truncated_audio_bytes += agent_chunk.raw_data.get(
                "truncated_audio_bytes", 0
            )
            if segment.interruption_audio_start_ms is None:
                segment.interruption_audio_start_ms = agent_chunk.raw_data.get(
                    "interruption_audio_start_ms"
                )

    # Update audio effect segments
    curr_active = _get_active_effects_from_chunk(agent_chunk)
    segment.audio_effects, new_prev = _update_audio_effect_segments(
        segment.audio_effects, prev_active, curr_active, tick_idx, tick_duration_sec
    )

    return new_prev


def _finalize_segment(
    segment: SpeechSegment,
    end_tick_idx: int,
    end_tick: Optional[Tick],
    ticks: List[Tick],
    tick_duration_sec: float,
) -> None:
    """Finalize a segment by computing end timing and context."""
    segment.end_tick = end_tick_idx

    # Compute duration
    segment.duration_sec = (segment.end_tick - segment.start_tick) * tick_duration_sec
    segment.end_time_sec = segment.start_time_sec + segment.duration_sec

    # Compute precise timing if start_time_ms is available
    if segment.start_time_ms is not None:
        tick_duration_ms = tick_duration_sec * 1000
        segment.duration_ms = (segment.end_tick - segment.start_tick) * tick_duration_ms
        segment.end_time_ms = segment.start_time_ms + segment.duration_ms

    # Check if other party was speaking at segment end
    if end_tick is not None:
        if segment.role == "user" and end_tick.agent_chunk:
            segment.other_speaking_at_end = getattr(
                end_tick.agent_chunk, "contains_speech", False
            )
        elif segment.role == "agent" and end_tick.user_chunk:
            segment.other_speaking_at_end = getattr(
                end_tick.user_chunk, "contains_speech", False
            )

    # Compute aggregate effect flags from effect segments
    if hasattr(segment, "audio_effects"):
        has_burst, has_vocal_tic, has_non_directed, has_muffling = (
            _compute_effect_flags(segment.audio_effects)
        )
        segment.has_burst_noise = has_burst
        segment.has_vocal_tic = has_vocal_tic
        segment.has_non_directed_speech = has_non_directed
        segment.has_muffling = has_muffling


# =============================================================================
# High-Level Extraction Function
# =============================================================================


def filter_end_of_conversation_ticks(ticks: List[Tick]) -> List[Tick]:
    """
    Filter out end-of-conversation artifact ticks.

    Conversations end when the user outputs ###STOP###. This creates a 1-tick
    user segment at the very end that triggers false positive "No Yield" events.

    This function removes the last tick if:
    - It's the last tick in the simulation
    - The user has speech in that tick (contains_speech=True)
    - The previous tick did NOT have user speech (i.e., this is a 1-tick segment)

    Args:
        ticks: List of Tick objects from a simulation

    Returns:
        Filtered list of ticks with end-of-conversation artifact removed
    """
    if len(ticks) < 2:
        return ticks

    last_tick = ticks[-1]
    prev_tick = ticks[-2]

    # Check if last tick has user speech
    last_has_user_speech = last_tick.user_chunk is not None and getattr(
        last_tick.user_chunk, "contains_speech", False
    )

    # Check if previous tick did NOT have user speech (making this a 1-tick segment)
    prev_has_user_speech = prev_tick.user_chunk is not None and getattr(
        prev_tick.user_chunk, "contains_speech", False
    )

    if last_has_user_speech and not prev_has_user_speech:
        logger.debug(
            f"Filtering out last tick (end-of-conversation artifact): "
            f"1-tick user segment at tick {len(ticks) - 1}"
        )
        return ticks[:-1]

    return ticks


def extract_all_segments(
    ticks: List[Tick],
    tick_duration_sec: float = 0.2,
) -> Tuple[List[UserSpeechSegment], List[AgentSpeechSegment]]:
    """
    Extract all speech segments from simulation ticks.

    Automatically filters out end-of-conversation artifacts (1-tick user segments
    at the very end caused by ###STOP###).

    Args:
        ticks: List of Tick objects from a simulation
        tick_duration_sec: Duration of each tick in seconds

    Returns:
        Tuple of (user_segments, agent_segments)
    """
    # Filter out end-of-conversation artifact
    ticks = filter_end_of_conversation_ticks(ticks)

    user_segments = extract_user_segments(ticks, tick_duration_sec)
    agent_segments = extract_agent_segments(ticks, tick_duration_sec)

    logger.debug(
        f"Extracted {len(user_segments)} user segments and {len(agent_segments)} agent segments"
    )

    return user_segments, agent_segments


# =============================================================================
# DataFrame Conversion (for CSV export)
# =============================================================================


def user_segments_to_dataframe(
    segments: List[UserSpeechSegment],
    simulation_id: str = "",
    task_id: str = "",
    llm: str = "",
    domain: str = "",
    speech_complexity: str = "",
    provider: str = "",
) -> "pd.DataFrame":
    """
    Convert user speech segments to a pandas DataFrame.

    Each row represents one user speech segment with all metadata.

    Args:
        segments: List of UserSpeechSegment objects
        simulation_id: Simulation identifier
        task_id: Task identifier
        llm: LLM model name (e.g., "openai:gpt-4o-realtime")
        domain: Domain name (e.g., "retail")
        speech_complexity: Speech complexity level (e.g., "regular")
        provider: Provider name (e.g., "openai")

    Returns:
        DataFrame with one row per segment
    """
    import pandas as pd

    rows = []
    for i, seg in enumerate(segments):
        rows.append(
            {
                # Experiment metadata
                "llm": llm,
                "domain": domain,
                "speech_complexity": speech_complexity,
                "provider": provider,
                # Simulation metadata
                "simulation_id": simulation_id,
                "task_id": task_id,
                # Segment identification
                "segment_idx": i,
                "role": seg.role,
                # Timing (tick-based)
                "start_tick": seg.start_tick,
                "end_tick": seg.end_tick,
                "tick_count": seg.tick_count,
                # Timing (seconds)
                "start_time_sec": seg.start_time_sec,
                "end_time_sec": seg.end_time_sec,
                "duration_sec": seg.duration_sec,
                # Timing (milliseconds, precise)
                "start_time_ms": seg.start_time_ms,
                "end_time_ms": seg.end_time_ms,
                "duration_ms": seg.duration_ms,
                # Content
                "transcript": seg.transcript,
                "utterance_ids": (
                    ",".join(seg.utterance_ids) if seg.utterance_ids else ""
                ),
                # Context
                "other_speaking_at_start": seg.other_speaking_at_start,
                "other_speaking_at_end": seg.other_speaking_at_end,
                # User-specific: Turn-taking
                "action": seg.action,
                "action_info": seg.action_info,
                "is_interruption": seg.is_interruption,
                "is_backchannel": seg.is_backchannel,
                # User-specific: Performance timing
                "interrupt_check_seconds": seg.interrupt_check_seconds,
                "backchannel_check_seconds": seg.backchannel_check_seconds,
                "llm_generation_seconds": seg.llm_generation_seconds,
                "tts_synthesis_seconds": seg.tts_synthesis_seconds,
            }
        )

    return pd.DataFrame(rows)


def agent_segments_to_dataframe(
    segments: List[AgentSpeechSegment],
    simulation_id: str = "",
    task_id: str = "",
    llm: str = "",
    domain: str = "",
    speech_complexity: str = "",
    provider: str = "",
) -> "pd.DataFrame":
    """
    Convert agent speech segments to a pandas DataFrame.

    Each row represents one agent speech segment with all metadata.

    Args:
        segments: List of AgentSpeechSegment objects
        simulation_id: Simulation identifier
        task_id: Task identifier
        llm: LLM model name
        domain: Domain name
        speech_complexity: Speech complexity level
        provider: Provider name

    Returns:
        DataFrame with one row per segment
    """
    import pandas as pd

    rows = []
    for i, seg in enumerate(segments):
        rows.append(
            {
                # Experiment metadata
                "llm": llm,
                "domain": domain,
                "speech_complexity": speech_complexity,
                "provider": provider,
                # Simulation metadata
                "simulation_id": simulation_id,
                "task_id": task_id,
                # Segment identification
                "segment_idx": i,
                "role": seg.role,
                # Timing (tick-based)
                "start_tick": seg.start_tick,
                "end_tick": seg.end_tick,
                "tick_count": seg.tick_count,
                # Timing (seconds)
                "start_time_sec": seg.start_time_sec,
                "end_time_sec": seg.end_time_sec,
                "duration_sec": seg.duration_sec,
                # Timing (milliseconds, precise)
                "start_time_ms": seg.start_time_ms,
                "end_time_ms": seg.end_time_ms,
                "duration_ms": seg.duration_ms,
                # Content
                "transcript": seg.transcript,
                "utterance_ids": (
                    ",".join(seg.utterance_ids) if seg.utterance_ids else ""
                ),
                # Context
                "other_speaking_at_start": seg.other_speaking_at_start,
                "other_speaking_at_end": seg.other_speaking_at_end,
                # Agent-specific: Interruption
                "was_interrupted": seg.was_interrupted,
                "truncated_audio_bytes": seg.truncated_audio_bytes,
                "interruption_audio_start_ms": seg.interruption_audio_start_ms,
                # Agent-specific: VAD events
                "vad_events": ",".join(seg.vad_events) if seg.vad_events else "",
                "vad_event_count": len(seg.vad_events),
            }
        )

    return pd.DataFrame(rows)


def extract_segments_from_results(
    params: dict,
    results: "Results",
    tick_duration_sec: float = 0.2,
) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Extract all segments from a Results object and return as DataFrames.

    This is the main entry point for processing results.json files,
    compatible with the load_simulation_results() pattern from performance_analysis.py.

    Args:
        params: Experiment parameters dict (llm, domain, speech_complexity, provider, etc.)
        results: Results object containing simulations
        tick_duration_sec: Duration of each tick in seconds

    Returns:
        Tuple of (user_segments_df, agent_segments_df) with all simulations combined
    """
    import pandas as pd

    all_user_dfs = []
    all_agent_dfs = []

    llm = params.get("llm", "")
    domain = params.get("domain", "")
    speech_complexity = params.get("speech_complexity", "")
    provider = params.get("provider", "")

    for sim in results.simulations:
        if not sim.ticks:
            logger.debug(f"Simulation {sim.id} has no ticks, skipping")
            continue

        # Extract segments
        user_segs, agent_segs = extract_all_segments(sim.ticks, tick_duration_sec)

        # Convert to DataFrames with metadata
        if user_segs:
            user_df = user_segments_to_dataframe(
                user_segs,
                simulation_id=sim.id,
                task_id=sim.task_id,
                llm=llm,
                domain=domain,
                speech_complexity=speech_complexity,
                provider=provider,
            )
            all_user_dfs.append(user_df)

        if agent_segs:
            agent_df = agent_segments_to_dataframe(
                agent_segs,
                simulation_id=sim.id,
                task_id=sim.task_id,
                llm=llm,
                domain=domain,
                speech_complexity=speech_complexity,
                provider=provider,
            )
            all_agent_dfs.append(agent_df)

    # Combine all DataFrames
    user_segments_df = (
        pd.concat(all_user_dfs, ignore_index=True) if all_user_dfs else pd.DataFrame()
    )
    agent_segments_df = (
        pd.concat(all_agent_dfs, ignore_index=True) if all_agent_dfs else pd.DataFrame()
    )

    logger.info(
        f"Extracted {len(user_segments_df)} user segments and {len(agent_segments_df)} agent segments "
        f"from {len(results.simulations)} simulations"
    )

    return user_segments_df, agent_segments_df


# =============================================================================
# Debug / Validation
# =============================================================================


def print_segment_summary(
    user_segments: List[UserSpeechSegment],
    agent_segments: List[AgentSpeechSegment],
) -> None:
    """Print a summary of extracted segments for debugging."""
    print("\n" + "=" * 60)
    print("SEGMENT EXTRACTION SUMMARY")
    print("=" * 60)

    print(f"\nUser segments: {len(user_segments)}")
    for i, seg in enumerate(user_segments):
        status = ""
        if seg.is_backchannel:
            status = " [BACKCHANNEL]"
        elif seg.is_interruption:
            status = " [INTERRUPTION]"

        print(
            f"  {i}: ticks {seg.start_tick}-{seg.end_tick} "
            f"({seg.duration_sec:.2f}s) action={seg.action}{status}"
        )
        if seg.transcript:
            preview = (
                seg.transcript[:50] + "..."
                if len(seg.transcript) > 50
                else seg.transcript
            )
            print(f"      text: {preview!r}")

    print(f"\nAgent segments: {len(agent_segments)}")
    for i, seg in enumerate(agent_segments):
        status = ""
        if seg.was_interrupted:
            status = f" [INTERRUPTED, {seg.truncated_audio_bytes} bytes truncated]"

        print(
            f"  {i}: ticks {seg.start_tick}-{seg.end_tick} "
            f"({seg.duration_sec:.2f}s){status}"
        )
        if seg.transcript:
            preview = (
                seg.transcript[:50] + "..."
                if len(seg.transcript) > 50
                else seg.transcript
            )
            print(f"      text: {preview!r}")
        if seg.vad_events:
            print(f"      vad_events: {seg.vad_events}")


# =============================================================================
# VAD (Voice Activity Detection) Analysis
# =============================================================================


def extract_vad_events(
    ticks: List[Tick],
    tick_duration_sec: float = 0.2,
) -> List[VADEvent]:
    """
    Extract VAD (Voice Activity Detection) events from ticks.

    VAD events are stored in tick.agent_chunk.raw_data["vad_events"] and include:
    - "speech_started": User started speaking (detected by provider VAD)
    - "speech_stopped": User stopped speaking (detected by provider VAD)
    - "interrupted": User interrupted agent (Gemini-specific)

    Args:
        ticks: List of Tick objects from a simulation
        tick_duration_sec: Duration of each tick in seconds

    Returns:
        List of VADEvent objects with timing information
    """
    vad_events = []

    for i, tick in enumerate(ticks):
        # VAD events are stored in agent_chunk.raw_data["vad_events"]
        if tick.agent_chunk and tick.agent_chunk.raw_data:
            raw_data = tick.agent_chunk.raw_data
            if isinstance(raw_data, dict):
                raw_vad_events = raw_data.get("vad_events", [])
                for event_type in raw_vad_events:
                    vad_events.append(
                        VADEvent(
                            event_type=event_type,
                            tick_idx=i,
                            time_sec=i * tick_duration_sec,
                        )
                    )

    return vad_events


def compute_vad_latencies(
    ticks: List[Tick],
    vad_events: List[VADEvent],
    tick_duration_sec: float = 0.2,
    max_latency_ticks: int = 25,  # 5 seconds at 0.2s/tick
    simulation_id: str = "",
    task_id: str = "",
    llm: str = "",
    domain: str = "",
    speech_complexity: str = "",
    provider: str = "",
) -> Tuple[List[VADLatencyEvent], List[VADMissEvent]]:
    """
    Compute VAD detection latencies and identify missed VAD detections.

    Measures the delay between when the user actually starts speaking
    (based on contains_speech transitioning to True) and when the
    provider's VAD emits a speech_started event. Also identifies cases
    where no VAD event was received within the expected window.

    Args:
        ticks: List of Tick objects from a simulation
        vad_events: List of VAD events extracted from ticks
        tick_duration_sec: Duration of each tick in seconds
        max_latency_ticks: Maximum ticks to wait for VAD (default 25 = 5s at 0.2s/tick)
        simulation_id: Simulation identifier
        task_id: Task identifier
        llm: LLM model name
        domain: Domain name
        speech_complexity: Speech complexity level
        provider: Provider name

    Returns:
        Tuple of (latencies, misses):
        - latencies: List of VADLatencyEvent objects with latency measurements
        - misses: List of VADMissEvent objects for failed VAD detections
    """
    latencies = []
    misses = []

    # Base metadata for events
    base_metadata = {
        "llm": llm,
        "domain": domain,
        "speech_complexity": speech_complexity,
        "provider": provider,
        "simulation_id": simulation_id,
        "task_id": task_id,
    }

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
                    **base_metadata,
                    user_speech_start_tick=start_tick,
                    user_speech_start_time_sec=start_tick * tick_duration_sec,
                    user_speech_duration_ticks=duration_ticks,
                    user_speech_duration_sec=duration_ticks * tick_duration_sec,
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
            # Only match if the VAD event is reasonably close
            if vad_event.tick_idx - start_tick <= max_latency_ticks:
                latency_ticks = vad_event.tick_idx - start_tick
                latencies.append(
                    VADLatencyEvent(
                        **base_metadata,
                        user_speech_start_tick=start_tick,
                        user_speech_start_time_sec=start_tick * tick_duration_sec,
                        vad_detected_tick=vad_event.tick_idx,
                        vad_detected_time_sec=vad_event.time_sec,
                        latency_ticks=latency_ticks,
                        latency_sec=latency_ticks * tick_duration_sec,
                    )
                )
                vad_start_idx += 1  # Move to next VAD event for next user speech
                matched = True

        # If no match, record as a miss
        if not matched:
            duration_ticks = end_tick - start_tick
            misses.append(
                VADMissEvent(
                    **base_metadata,
                    user_speech_start_tick=start_tick,
                    user_speech_start_time_sec=start_tick * tick_duration_sec,
                    user_speech_duration_ticks=duration_ticks,
                    user_speech_duration_sec=duration_ticks * tick_duration_sec,
                )
            )

    logger.debug(
        f"Computed VAD latencies: {len(latencies)} detected, {len(misses)} missed"
    )
    return latencies, misses


def vad_latencies_to_dataframe(
    latencies: List[VADLatencyEvent],
) -> "pd.DataFrame":
    """Convert VAD latency events to a pandas DataFrame."""
    import pandas as pd

    rows = []
    for e in latencies:
        rows.append(
            {
                "llm": e.llm,
                "domain": e.domain,
                "speech_complexity": e.speech_complexity,
                "provider": e.provider,
                "simulation_id": e.simulation_id,
                "task_id": e.task_id,
                "user_speech_start_tick": e.user_speech_start_tick,
                "user_speech_start_time_sec": e.user_speech_start_time_sec,
                "vad_detected_tick": e.vad_detected_tick,
                "vad_detected_time_sec": e.vad_detected_time_sec,
                "latency_ticks": e.latency_ticks,
                "latency_sec": e.latency_sec,
            }
        )

    return pd.DataFrame(rows)


def vad_misses_to_dataframe(
    misses: List[VADMissEvent],
) -> "pd.DataFrame":
    """Convert VAD miss events to a pandas DataFrame."""
    import pandas as pd

    rows = []
    for e in misses:
        rows.append(
            {
                "llm": e.llm,
                "domain": e.domain,
                "speech_complexity": e.speech_complexity,
                "provider": e.provider,
                "simulation_id": e.simulation_id,
                "task_id": e.task_id,
                "user_speech_start_tick": e.user_speech_start_tick,
                "user_speech_start_time_sec": e.user_speech_start_time_sec,
                "user_speech_duration_ticks": e.user_speech_duration_ticks,
                "user_speech_duration_sec": e.user_speech_duration_sec,
            }
        )

    return pd.DataFrame(rows)


def extract_vad_from_results(
    params: dict,
    results: "Results",
    tick_duration_sec: float = 0.2,
) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Extract VAD latencies and misses from a Results object.

    Args:
        params: Experiment parameters dict
        results: Results object containing simulations
        tick_duration_sec: Duration of each tick in seconds

    Returns:
        Tuple of (latencies_df, misses_df)
    """
    import pandas as pd

    all_latency_events = []
    all_miss_events = []

    llm = params.get("llm", "")
    domain = params.get("domain", "")
    speech_complexity = params.get("speech_complexity", "")
    provider = params.get("provider", "")

    for sim in results.simulations:
        if not sim.ticks:
            continue

        # Extract VAD events
        vad_events = extract_vad_events(sim.ticks, tick_duration_sec)

        # Compute latencies and misses
        latencies, misses = compute_vad_latencies(
            sim.ticks,
            vad_events,
            tick_duration_sec,
            simulation_id=sim.id,
            task_id=sim.task_id,
            llm=llm,
            domain=domain,
            speech_complexity=speech_complexity,
            provider=provider,
        )

        all_latency_events.extend(latencies)
        all_miss_events.extend(misses)

    # Convert to DataFrames
    latencies_df = (
        vad_latencies_to_dataframe(all_latency_events)
        if all_latency_events
        else pd.DataFrame()
    )
    misses_df = (
        vad_misses_to_dataframe(all_miss_events) if all_miss_events else pd.DataFrame()
    )

    logger.info(
        f"Extracted {len(latencies_df)} VAD latency events and {len(misses_df)} VAD misses "
        f"from {len(results.simulations)} simulations"
    )

    return latencies_df, misses_df


def analyze_vad_latencies(
    latencies_df: "pd.DataFrame",
    misses_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """
    Compute summary statistics for VAD latencies.

    Aggregates by llm, domain, speech_complexity.

    Args:
        latencies_df: Raw VAD latency DataFrame
        misses_df: Raw VAD miss DataFrame

    Returns:
        Summary statistics DataFrame
    """
    import numpy as np
    import pandas as pd

    if latencies_df.empty and misses_df.empty:
        return pd.DataFrame()

    # Get all unique groups from both dataframes
    all_groups = set()
    group_cols = ["llm", "domain", "speech_complexity", "provider"]

    if not latencies_df.empty:
        for _, row in latencies_df[group_cols].drop_duplicates().iterrows():
            all_groups.add(tuple(row))

    if not misses_df.empty:
        for _, row in misses_df[group_cols].drop_duplicates().iterrows():
            all_groups.add(tuple(row))

    summary_rows = []
    for llm, domain, complexity, provider in all_groups:
        # Filter to this group
        lat_group = (
            latencies_df[
                (latencies_df["llm"] == llm)
                & (latencies_df["domain"] == domain)
                & (latencies_df["speech_complexity"] == complexity)
            ]
            if not latencies_df.empty
            else pd.DataFrame()
        )

        miss_group = (
            misses_df[
                (misses_df["llm"] == llm)
                & (misses_df["domain"] == domain)
                & (misses_df["speech_complexity"] == complexity)
            ]
            if not misses_df.empty
            else pd.DataFrame()
        )

        total_events = len(lat_group) + len(miss_group)
        detection_rate = len(lat_group) / total_events if total_events > 0 else np.nan

        row = {
            "llm": llm,
            "domain": domain,
            "speech_complexity": complexity,
            "provider": provider,
            # Counts
            "total_speech_events": total_events,
            "detected_count": len(lat_group),
            "missed_count": len(miss_group),
            "detection_rate": detection_rate,
            # Latency stats (for detected events)
            "latency_mean_sec": (
                lat_group["latency_sec"].mean() if len(lat_group) > 0 else np.nan
            ),
            "latency_std_sec": (
                lat_group["latency_sec"].std() if len(lat_group) > 0 else np.nan
            ),
            "latency_min_sec": (
                lat_group["latency_sec"].min() if len(lat_group) > 0 else np.nan
            ),
            "latency_max_sec": (
                lat_group["latency_sec"].max() if len(lat_group) > 0 else np.nan
            ),
            "latency_median_sec": (
                lat_group["latency_sec"].median() if len(lat_group) > 0 else np.nan
            ),
            "latency_p95_sec": (
                lat_group["latency_sec"].quantile(0.95)
                if len(lat_group) > 0
                else np.nan
            ),
        }
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def plot_vad_analysis(
    latencies_df: "pd.DataFrame" = None,
    analysis_df: "pd.DataFrame" = None,
    output_dir: Path = None,
    llms: List[str] = None,
    domains: List[str] = None,
    complexities: List[str] = None,
) -> List[Path]:
    """
    Create VAD analysis visualizations.

    LLM on x-axis, bars grouped by complexity (control=hatched, regular=solid).

    Creates:
    1. VAD detection rate by LLM/domain/complexity
    2. VAD latency distribution by LLM/domain/complexity

    Args:
        latencies_df: Raw VAD latency DataFrame (optional for plots-only mode)
        analysis_df: Analysis DataFrame
        output_dir: Directory to save the figures
        llms: List of LLM names (optional, extracted from analysis_df if not provided)
        domains: List of domains (optional, extracted from analysis_df if not provided)
        complexities: List of complexities (optional, extracted from analysis_df if not provided)

    Returns:
        List of paths to saved figures
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Import shared styling
    from experiments.tau_voice.exp.plot_style import (
        BAR_STYLE,
        COMPLEXITY_STYLES,
        get_legend_patch,
        get_llm_color,
        style_axis,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    if analysis_df is None or analysis_df.empty:
        logger.warning("No data for VAD analysis plots")
        return output_paths

    # Get unique values if not provided
    if llms is None:
        llms = sorted(analysis_df["llm"].unique())
    if domains is None:
        domains = sorted(analysis_df["domain"].unique())
    if complexities is None:
        complexities = sorted(analysis_df["speech_complexity"].unique())

    if not llms or not domains or not complexities:
        logger.warning("Insufficient data for VAD plots")
        return output_paths

    n_domains = len(domains)
    n_complexities = len(complexities)
    n_llms = len(llms)
    group_width = 0.8
    bar_width = group_width / n_complexities

    # =========================================================================
    # Figure 1: VAD Detection Rate - LLM on x-axis, grouped by complexity
    # =========================================================================
    fig, axes = plt.subplots(n_domains, 1, figsize=(10, 5 * n_domains), squeeze=False)
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains):
        ax = axes[d_idx]
        domain_df = analysis_df[analysis_df["domain"] == domain]

        x = np.arange(n_llms)

        for c_idx, complexity in enumerate(complexities):
            values = []
            colors = []

            for llm in llms:
                subset = domain_df[
                    (domain_df["llm"] == llm)
                    & (domain_df["speech_complexity"] == complexity)
                ]
                if len(subset) > 0:
                    rate = subset["detection_rate"].values[0]
                    values.append(rate * 100 if not np.isnan(rate) else np.nan)
                else:
                    values.append(np.nan)
                colors.append(get_llm_color(llm))

            bar_offset = (c_idx - (n_complexities - 1) / 2) * bar_width
            x_pos = x + bar_offset

            style = COMPLEXITY_STYLES.get(complexity, {"alpha": 0.8, "hatch": ""})
            ax.bar(
                x_pos,
                values,
                bar_width * 0.9,
                color=colors,
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )

        ax.set_xlabel("LLM", fontsize=11)
        ax.set_ylabel("Detection Rate (%)", fontsize=11)
        ax.set_title(f"{domain.capitalize()}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [llm.split(":")[-1][:12] if ":" in llm else llm[:12] for llm in llms],
            fontsize=9,
            rotation=45,
            ha="right",
        )
        ax.set_ylim(0, 105)
        style_axis(ax)

    # Add complexity legend
    legend_patches = [get_legend_patch(c) for c in complexities]
    axes[-1].legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=9,
        title="Complexity",
    )

    fig.suptitle("VAD Detection Rate", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "vad_detection_rate.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    output_paths.append(output_path)
    logger.info("Saved: vad_detection_rate.pdf")

    # =========================================================================
    # Figure 2: VAD Latency - LLM on x-axis, grouped by complexity
    # =========================================================================
    fig, axes = plt.subplots(n_domains, 1, figsize=(10, 5 * n_domains), squeeze=False)
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains):
        ax = axes[d_idx]
        domain_df = analysis_df[analysis_df["domain"] == domain]

        x = np.arange(n_llms)

        for c_idx, complexity in enumerate(complexities):
            values = []
            stds = []
            colors = []

            for llm in llms:
                subset = domain_df[
                    (domain_df["llm"] == llm)
                    & (domain_df["speech_complexity"] == complexity)
                ]
                if len(subset) > 0:
                    values.append(subset["latency_mean_sec"].values[0])
                    std_val = subset["latency_std_sec"].values[0]
                    stds.append(std_val if not np.isnan(std_val) else 0)
                else:
                    values.append(np.nan)
                    stds.append(0)
                colors.append(get_llm_color(llm))

            bar_offset = (c_idx - (n_complexities - 1) / 2) * bar_width
            x_pos = x + bar_offset

            style = COMPLEXITY_STYLES.get(complexity, {"alpha": 0.8, "hatch": ""})
            ax.bar(
                x_pos,
                values,
                bar_width * 0.9,
                yerr=stds,
                capsize=2,
                color=colors,
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )

        ax.set_xlabel("LLM", fontsize=11)
        ax.set_ylabel("Latency (s)", fontsize=11)
        ax.set_title(f"{domain.capitalize()}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [llm.split(":")[-1][:12] if ":" in llm else llm[:12] for llm in llms],
            fontsize=9,
            rotation=45,
            ha="right",
        )
        style_axis(ax)

    # Align y-axis scales
    max_ylim = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, max_ylim * 1.1)

    # Add complexity legend
    legend_patches = [get_legend_patch(c) for c in complexities]
    axes[-1].legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=9,
        title="Complexity",
    )

    fig.suptitle("VAD Detection Latency", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "vad_latency.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    output_paths.append(output_path)
    logger.info("Saved: vad_latency.pdf")

    return output_paths


def run_vad_analysis(
    all_results: List[Tuple[dict, "Results"]],
    output_dir: Path,
    tick_duration_sec: float = 0.2,
) -> dict:
    """
    Run the full VAD analysis pipeline.

    Args:
        all_results: List of (params, Results) tuples
        output_dir: Base output directory
        tick_duration_sec: Duration of each tick in seconds

    Returns:
        Dict with paths to output files
    """
    import pandas as pd

    # Create output directory
    analysis_dir = output_dir / "vad_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Collect all VAD data
    all_latency_dfs = []
    all_miss_dfs = []

    for params, results in all_results:
        latencies_df, misses_df = extract_vad_from_results(
            params, results, tick_duration_sec
        )
        if not latencies_df.empty:
            all_latency_dfs.append(latencies_df)
        if not misses_df.empty:
            all_miss_dfs.append(misses_df)

    result_paths = {
        "latencies_raw_path": None,
        "misses_raw_path": None,
        "analysis_path": None,
        "plot_paths": [],
    }

    # Combine all data
    latencies_df = (
        pd.concat(all_latency_dfs, ignore_index=True)
        if all_latency_dfs
        else pd.DataFrame()
    )
    misses_df = (
        pd.concat(all_miss_dfs, ignore_index=True) if all_miss_dfs else pd.DataFrame()
    )

    if latencies_df.empty and misses_df.empty:
        logger.warning("No VAD data found")
        return result_paths

    logger.info(
        f"Collected {len(latencies_df)} VAD latency events and {len(misses_df)} VAD misses"
    )

    # Save raw data
    if not latencies_df.empty:
        latencies_path = analysis_dir / "latencies_raw.csv"
        latencies_df.to_csv(latencies_path, index=False)
        result_paths["latencies_raw_path"] = latencies_path
        logger.info(f"Saved VAD latencies to {latencies_path}")

    if not misses_df.empty:
        misses_path = analysis_dir / "misses_raw.csv"
        misses_df.to_csv(misses_path, index=False)
        result_paths["misses_raw_path"] = misses_path
        logger.info(f"Saved VAD misses to {misses_path}")

    # Compute analysis
    analysis_df = analyze_vad_latencies(latencies_df, misses_df)

    if not analysis_df.empty:
        analysis_path = analysis_dir / f"{analysis_dir.name}_analysis.csv"
        analysis_df.to_csv(analysis_path, index=False)
        result_paths["analysis_path"] = analysis_path
        logger.info(f"Saved VAD analysis to {analysis_path}")

        # Generate plots
        result_paths["plot_paths"] = plot_vad_analysis(
            latencies_df, analysis_df, analysis_dir
        )

    return result_paths


# =============================================================================
# Response Latency Analysis
# =============================================================================


@dataclass
class TurnTransitionEvent:
    """
    Represents what happened after a user finished speaking.

    Outcomes:
    - "response": Agent responded to the user
    - "no_response": User spoke again before agent responded
    """

    # Experiment metadata
    llm: str = ""
    domain: str = ""
    speech_complexity: str = ""
    provider: str = ""

    # Simulation metadata
    simulation_id: str = ""
    task_id: str = ""

    # User segment info
    user_segment_idx: int = 0
    user_end_tick: int = 0
    user_end_time_sec: float = 0.0
    user_transcript: str = ""
    user_action: str = ""

    # Outcome: "response" or "no_response"
    outcome: str = ""

    # For "response" outcome: agent info
    agent_segment_idx: Optional[int] = None
    agent_start_tick: Optional[int] = None
    agent_start_time_sec: Optional[float] = None

    # For "no_response" outcome: next user info
    next_user_segment_idx: Optional[int] = None
    next_user_start_tick: Optional[int] = None
    next_user_start_time_sec: Optional[float] = None

    # Time gap (latency for response, silence for no_response)
    gap_ticks: int = 0
    gap_sec: float = 0.0


# Keep aliases for backwards compatibility
ResponseLatencyEvent = TurnTransitionEvent
NoResponseEvent = TurnTransitionEvent


def extract_turn_transitions(
    user_segments: List[UserSpeechSegment],
    agent_segments: List[AgentSpeechSegment],
    simulation_id: str = "",
    task_id: str = "",
    llm: str = "",
    domain: str = "",
    speech_complexity: str = "",
    provider: str = "",
) -> List[TurnTransitionEvent]:
    """
    Extract turn transition events from speech segments.

    For each valid user speech segment, determines what happened next:
    - "response": Agent responded to the user (measure latency)
    - "no_response": User spoke again before agent responded (measure silence)

    Filters out:
    - Backchannels (agent should not respond to these)
    - User interruptions (different dynamic, user is cutting in)
    - Cases where agent was already speaking at end of user turn

    Args:
        user_segments: List of user speech segments
        agent_segments: List of agent speech segments
        simulation_id: Simulation identifier
        task_id: Task identifier
        llm: LLM model name
        domain: Domain name
        speech_complexity: Speech complexity level
        provider: Provider name

    Returns:
        List of TurnTransitionEvent objects
    """
    events = []

    # Get valid user segments (filter out backchannels, agent speaking at end)
    # Note: We include interruptions where the agent yielded (other_speaking_at_end=False)
    # because the user has taken the floor and expects a response
    valid_user_segments = []
    for user_idx, user_seg in enumerate(user_segments):
        if user_seg.is_backchannel:
            logger.debug(
                f"Skipping backchannel at ticks {user_seg.start_tick}-{user_seg.end_tick}"
            )
            continue

        # Skip interruptions only if agent is STILL speaking at the end
        # If agent yielded (stopped), the user has taken the floor and should get a response
        if user_seg.is_interruption and user_seg.other_speaking_at_end:
            logger.debug(
                f"Skipping interruption at ticks {user_seg.start_tick}-{user_seg.end_tick}: "
                "agent still speaking at end"
            )
            continue

        if user_seg.other_speaking_at_end:
            logger.debug(
                f"Skipping segment at ticks {user_seg.start_tick}-{user_seg.end_tick}: "
                "agent was speaking at end"
            )
            continue

        valid_user_segments.append((user_idx, user_seg))

    # Sort agent segments by start_tick
    sorted_agent_segments = sorted(agent_segments, key=lambda s: s.start_tick)

    # For each valid user segment, determine if it got a response or not
    for i, (user_idx, user_seg) in enumerate(valid_user_segments):
        # Find the next agent segment that starts after this user ends
        next_agent_start = None
        next_agent_idx = None
        next_agent_seg = None
        for agent_idx, agent_seg in enumerate(sorted_agent_segments):
            if agent_seg.start_tick >= user_seg.end_tick:
                next_agent_start = agent_seg.start_tick
                next_agent_idx = agent_idx
                next_agent_seg = agent_seg
                break

        # Find the next valid user segment that starts after this user ends
        next_user_start = None
        next_user_idx = None
        next_user_seg = None
        for j in range(i + 1, len(valid_user_segments)):
            next_idx, next_seg = valid_user_segments[j]
            if next_seg.start_tick >= user_seg.end_tick:
                next_user_start = next_seg.start_tick
                next_user_idx = next_idx
                next_user_seg = next_seg
                break

        # Base event data
        base_event = {
            "llm": llm,
            "domain": domain,
            "speech_complexity": speech_complexity,
            "provider": provider,
            "simulation_id": simulation_id,
            "task_id": task_id,
            "user_segment_idx": user_idx,
            "user_end_tick": user_seg.end_tick,
            "user_end_time_sec": user_seg.end_time_sec,
            "user_transcript": user_seg.transcript,
            "user_action": user_seg.action,
        }

        # Determine outcome: response vs no-response
        if next_agent_start is not None:
            if next_user_start is None or next_agent_start <= next_user_start:
                # Agent responded before user spoke again
                gap_ticks = next_agent_seg.start_tick - user_seg.end_tick
                gap_sec = next_agent_seg.start_time_sec - user_seg.end_time_sec

                events.append(
                    TurnTransitionEvent(
                        **base_event,
                        outcome="response",
                        agent_segment_idx=next_agent_idx,
                        agent_start_tick=next_agent_seg.start_tick,
                        agent_start_time_sec=next_agent_seg.start_time_sec,
                        gap_ticks=gap_ticks,
                        gap_sec=gap_sec,
                    )
                )
            else:
                # User spoke again before agent responded
                gap_ticks = next_user_seg.start_tick - user_seg.end_tick
                gap_sec = next_user_seg.start_time_sec - user_seg.end_time_sec

                events.append(
                    TurnTransitionEvent(
                        **base_event,
                        outcome="no_response",
                        next_user_segment_idx=next_user_idx,
                        next_user_start_tick=next_user_seg.start_tick,
                        next_user_start_time_sec=next_user_seg.start_time_sec,
                        gap_ticks=gap_ticks,
                        gap_sec=gap_sec,
                    )
                )
        elif next_user_start is not None:
            # No agent response at all, but user spoke again
            gap_ticks = next_user_seg.start_tick - user_seg.end_tick
            gap_sec = next_user_seg.start_time_sec - user_seg.end_time_sec

            events.append(
                TurnTransitionEvent(
                    **base_event,
                    outcome="no_response",
                    next_user_segment_idx=next_user_idx,
                    next_user_start_tick=next_user_seg.start_tick,
                    next_user_start_time_sec=next_user_seg.start_time_sec,
                    gap_ticks=gap_ticks,
                    gap_sec=gap_sec,
                )
            )
        # else: last user segment with no following agent or user speech - skip

    response_count = sum(1 for e in events if e.outcome == "response")
    no_response_count = sum(1 for e in events if e.outcome == "no_response")
    logger.debug(
        f"Extracted {len(events)} turn transition events "
        f"({response_count} response, {no_response_count} no_response)"
    )
    return events


# Backwards compatibility alias
def extract_response_latencies(
    user_segments: List[UserSpeechSegment],
    agent_segments: List[AgentSpeechSegment],
    **kwargs,
) -> Tuple[List[TurnTransitionEvent], List[TurnTransitionEvent]]:
    """Deprecated: Use extract_turn_transitions instead."""
    events = extract_turn_transitions(user_segments, agent_segments, **kwargs)
    response_events = [e for e in events if e.outcome == "response"]
    no_response_events = [e for e in events if e.outcome == "no_response"]
    return response_events, no_response_events


def turn_transitions_to_dataframe(
    events: List[TurnTransitionEvent],
) -> "pd.DataFrame":
    """
    Convert turn transition events to a pandas DataFrame.

    Args:
        events: List of TurnTransitionEvent objects

    Returns:
        DataFrame with one row per event, with unified columns
    """
    import pandas as pd

    rows = []
    for e in events:
        rows.append(
            {
                # Experiment metadata
                "llm": e.llm,
                "domain": e.domain,
                "speech_complexity": e.speech_complexity,
                "provider": e.provider,
                # Simulation metadata
                "simulation_id": e.simulation_id,
                "task_id": e.task_id,
                # User segment info
                "user_segment_idx": e.user_segment_idx,
                "user_end_tick": e.user_end_tick,
                "user_end_time_sec": e.user_end_time_sec,
                "user_transcript": e.user_transcript,
                "user_action": e.user_action,
                # Outcome
                "outcome": e.outcome,
                # Gap (latency for response, silence for no_response)
                "gap_ticks": e.gap_ticks,
                "gap_sec": e.gap_sec,
                # Agent info (for response outcome)
                "agent_segment_idx": e.agent_segment_idx,
                "agent_start_tick": e.agent_start_tick,
                "agent_start_time_sec": e.agent_start_time_sec,
                # Next user info (for no_response outcome)
                "next_user_segment_idx": e.next_user_segment_idx,
                "next_user_start_tick": e.next_user_start_tick,
                "next_user_start_time_sec": e.next_user_start_time_sec,
            }
        )

    return pd.DataFrame(rows)


# Backwards compatibility aliases
def response_latencies_to_dataframe(
    events: List[TurnTransitionEvent],
) -> "pd.DataFrame":
    """Deprecated: Use turn_transitions_to_dataframe instead."""
    return turn_transitions_to_dataframe(events)


def no_response_events_to_dataframe(
    events: List[TurnTransitionEvent],
) -> "pd.DataFrame":
    """Deprecated: Use turn_transitions_to_dataframe instead."""
    return turn_transitions_to_dataframe(events)


def extract_turn_transitions_from_results(
    params: dict,
    results: "Results",
    tick_duration_sec: float = 0.2,
) -> "pd.DataFrame":
    """
    Extract all turn transition events from a Results object.

    This is the main entry point for processing results.json files.

    Args:
        params: Experiment parameters dict (llm, domain, speech_complexity, provider, etc.)
        results: Results object containing simulations
        tick_duration_sec: Duration of each tick in seconds

    Returns:
        DataFrame with all turn transition events (both response and no_response)
    """
    import pandas as pd

    all_dfs = []

    llm = params.get("llm", "")
    domain = params.get("domain", "")
    speech_complexity = params.get("speech_complexity", "")
    provider = params.get("provider", "")

    for sim in results.simulations:
        if not sim.ticks:
            continue

        # Extract segments
        user_segs, agent_segs = extract_all_segments(sim.ticks, tick_duration_sec)

        # Extract turn transitions
        events = extract_turn_transitions(
            user_segs,
            agent_segs,
            simulation_id=sim.id,
            task_id=sim.task_id,
            llm=llm,
            domain=domain,
            speech_complexity=speech_complexity,
            provider=provider,
        )

        if events:
            df = turn_transitions_to_dataframe(events)
            all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


# Backwards compatibility alias
def extract_response_latencies_from_results(
    params: dict,
    results: "Results",
    tick_duration_sec: float = 0.2,
) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """Deprecated: Use extract_turn_transitions_from_results instead."""
    df = extract_turn_transitions_from_results(params, results, tick_duration_sec)
    if df.empty:
        import pandas as pd

        return pd.DataFrame(), pd.DataFrame()
    response_df = df[df["outcome"] == "response"].copy()
    no_response_df = df[df["outcome"] == "no_response"].copy()
    return response_df, no_response_df


def analyze_turn_transitions(
    raw_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """
    Compute summary statistics for turn transitions.

    Aggregates by llm, domain, and speech_complexity, computing statistics
    for both response and no_response outcomes.

    Args:
        raw_df: Raw turn transition DataFrame

    Returns:
        Summary statistics DataFrame
    """
    import numpy as np
    import pandas as pd

    if raw_df.empty:
        return pd.DataFrame()

    # Group by experiment configuration
    grouped = raw_df.groupby(["llm", "domain", "speech_complexity", "provider"])

    summary_rows = []
    for (llm, domain, complexity, provider), group in grouped:
        response_df = group[group["outcome"] == "response"]
        no_response_df = group[group["outcome"] == "no_response"]

        response_gaps = response_df["gap_sec"]
        no_response_gaps = no_response_df["gap_sec"]

        summary_rows.append(
            {
                "llm": llm,
                "domain": domain,
                "speech_complexity": complexity,
                "provider": provider,
                # Overall counts
                "total_count": len(group),
                "response_count": len(response_df),
                "no_response_count": len(no_response_df),
                "response_rate": len(response_df) / len(group) if len(group) > 0 else 0,
                # Response latency stats
                "latency_mean": (
                    response_gaps.mean() if len(response_gaps) > 0 else np.nan
                ),
                "latency_std": (
                    response_gaps.std() if len(response_gaps) > 0 else np.nan
                ),
                "latency_min": (
                    response_gaps.min() if len(response_gaps) > 0 else np.nan
                ),
                "latency_max": (
                    response_gaps.max() if len(response_gaps) > 0 else np.nan
                ),
                "latency_median": (
                    response_gaps.median() if len(response_gaps) > 0 else np.nan
                ),
                "latency_p25": (
                    response_gaps.quantile(0.25) if len(response_gaps) > 0 else np.nan
                ),
                "latency_p75": (
                    response_gaps.quantile(0.75) if len(response_gaps) > 0 else np.nan
                ),
                "latency_p95": (
                    response_gaps.quantile(0.95) if len(response_gaps) > 0 else np.nan
                ),
                # No-response silence stats
                "silence_mean": (
                    no_response_gaps.mean() if len(no_response_gaps) > 0 else np.nan
                ),
                "silence_std": (
                    no_response_gaps.std() if len(no_response_gaps) > 0 else np.nan
                ),
                "silence_min": (
                    no_response_gaps.min() if len(no_response_gaps) > 0 else np.nan
                ),
                "silence_max": (
                    no_response_gaps.max() if len(no_response_gaps) > 0 else np.nan
                ),
            }
        )

    return pd.DataFrame(summary_rows)


# Backwards compatibility alias
def analyze_response_latency(raw_df: "pd.DataFrame") -> "pd.DataFrame":
    """Deprecated: Use analyze_turn_transitions instead."""
    return analyze_turn_transitions(raw_df)


def save_turn_transitions_raw(
    raw_df: "pd.DataFrame",
    output_dir: Path,
) -> Path:
    """
    Save raw turn transition data to CSV.

    Args:
        raw_df: Raw turn transition DataFrame
        output_dir: Directory to save the file

    Returns:
        Path to the saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_dir.name}_raw.csv"
    raw_df.to_csv(output_path, index=False)
    logger.info(f"Saved raw turn transition data to {output_path}")
    return output_path


def save_turn_transitions_analysis(
    analysis_df: "pd.DataFrame",
    output_dir: Path,
) -> Path:
    """
    Save turn transition analysis to CSV.

    Args:
        analysis_df: Analysis DataFrame
        output_dir: Directory to save the file

    Returns:
        Path to the saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_dir.name}_analysis.csv"
    analysis_df.to_csv(output_path, index=False)
    logger.info(f"Saved turn transition analysis to {output_path}")
    return output_path


# Backwards compatibility aliases
save_response_latency_raw = save_turn_transitions_raw
save_response_latency_analysis = save_turn_transitions_analysis


def plot_turn_transitions(
    raw_df: "pd.DataFrame" = None,
    analysis_df: "pd.DataFrame" = None,
    output_dir: Path = None,
    llms: List[str] = None,
    domains: List[str] = None,
    complexities: List[str] = None,
) -> List[Path]:
    """
    Create turn transition visualizations following performance_analysis.py conventions.

    LLM on x-axis, bars grouped by complexity (control=hatched, regular=solid).

    Creates separate figures for:
    1. Response latency per domain, LLM on x-axis, grouped by complexity
    2. No-response rate per domain, LLM on x-axis, grouped by complexity

    Args:
        raw_df: Raw turn transition DataFrame (optional for plots-only mode)
        analysis_df: Analysis DataFrame
        output_dir: Directory to save the figures
        llms: List of LLM names (optional, extracted from analysis_df if not provided)
        domains: List of domains (optional, extracted from analysis_df if not provided)
        complexities: List of complexities (optional, extracted from analysis_df if not provided)

    Returns:
        List of paths to saved figures
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Import shared styling
    from experiments.tau_voice.exp.plot_style import (
        BAR_STYLE,
        COMPLEXITY_STYLES,
        get_legend_patch,
        get_llm_color,
        style_axis,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    if analysis_df is None or analysis_df.empty:
        logger.warning("No data for turn transition plots")
        return output_paths

    # Get unique values if not provided
    if llms is None:
        llms = sorted(analysis_df["llm"].unique())
    if domains is None:
        domains = sorted(analysis_df["domain"].unique())
    if complexities is None:
        complexities = sorted(analysis_df["speech_complexity"].unique())

    if not llms or not domains or not complexities:
        logger.warning("Insufficient data for turn transition plots")
        return output_paths

    n_domains = len(domains)
    n_llms = len(llms)
    n_complexities = len(complexities)
    group_width = 0.8
    bar_width = group_width / n_complexities

    # =========================================================================
    # Figure 1: Response Latency - LLM on x-axis, grouped by complexity
    # =========================================================================
    fig, axes = plt.subplots(n_domains, 1, figsize=(10, 5 * n_domains), squeeze=False)
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains):
        ax = axes[d_idx]
        domain_df = analysis_df[analysis_df["domain"] == domain]

        x = np.arange(n_llms)

        for c_idx, complexity in enumerate(complexities):
            values = []
            stds = []
            colors = []

            for llm in llms:
                subset = domain_df[
                    (domain_df["llm"] == llm)
                    & (domain_df["speech_complexity"] == complexity)
                ]
                if len(subset) > 0 and "latency_mean" in subset.columns:
                    values.append(subset["latency_mean"].values[0])
                    std_val = (
                        subset["latency_std"].values[0]
                        if "latency_std" in subset.columns
                        else 0
                    )
                    stds.append(std_val if not np.isnan(std_val) else 0)
                else:
                    values.append(np.nan)
                    stds.append(0)
                colors.append(get_llm_color(llm))

            bar_offset = (c_idx - (n_complexities - 1) / 2) * bar_width
            x_pos = x + bar_offset

            style = COMPLEXITY_STYLES.get(complexity, {"alpha": 0.8, "hatch": ""})
            ax.bar(
                x_pos,
                values,
                bar_width * 0.9,
                yerr=stds,
                capsize=2,
                color=colors,
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )

        ax.set_xlabel("LLM", fontsize=11)
        ax.set_ylabel("Latency (s)", fontsize=11)
        ax.set_title(f"{domain.capitalize()}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [llm.split(":")[-1][:12] if ":" in llm else llm[:12] for llm in llms],
            fontsize=9,
            rotation=45,
            ha="right",
        )
        style_axis(ax)

    # Align y-axis scales
    max_ylim = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, max_ylim * 1.1)

    # Add complexity legend
    legend_patches = [get_legend_patch(c) for c in complexities]
    axes[-1].legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=9,
        title="Complexity",
    )

    fig.suptitle("Response Latency", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "response_latency.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    output_paths.append(output_path)
    logger.info("Saved: response_latency.pdf")

    # =========================================================================
    # Figure 2: No-Response Rate - LLM on x-axis, grouped by complexity
    # =========================================================================
    fig, axes = plt.subplots(n_domains, 1, figsize=(10, 5 * n_domains), squeeze=False)
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains):
        ax = axes[d_idx]
        domain_df = analysis_df[analysis_df["domain"] == domain]

        x = np.arange(n_llms)

        for c_idx, complexity in enumerate(complexities):
            values = []
            colors = []

            for llm in llms:
                subset = domain_df[
                    (domain_df["llm"] == llm)
                    & (domain_df["speech_complexity"] == complexity)
                ]
                if len(subset) > 0:
                    total = subset["total_count"].values[0]
                    no_resp = subset["no_response_count"].values[0]
                    rate = (no_resp / total * 100) if total > 0 else 0
                    values.append(rate)
                else:
                    values.append(np.nan)
                colors.append(get_llm_color(llm))

            bar_offset = (c_idx - (n_complexities - 1) / 2) * bar_width
            x_pos = x + bar_offset

            style = COMPLEXITY_STYLES.get(complexity, {"alpha": 0.8, "hatch": ""})
            ax.bar(
                x_pos,
                values,
                bar_width * 0.9,
                color=colors,
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )

        ax.set_xlabel("LLM", fontsize=11)
        ax.set_ylabel("No-Response Rate (%)", fontsize=11)
        ax.set_title(f"{domain.capitalize()}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [llm.split(":")[-1][:12] if ":" in llm else llm[:12] for llm in llms],
            fontsize=9,
            rotation=45,
            ha="right",
        )
        style_axis(ax)

    # Align y-axis scales
    max_ylim = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, max(max_ylim * 1.1, 10))

    # Add complexity legend
    legend_patches = [get_legend_patch(c) for c in complexities]
    axes[-1].legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=9,
        title="Complexity",
    )

    fig.suptitle("No-Response Rate (User Had to Retry)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "no_response_rate.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    output_paths.append(output_path)
    logger.info(f"Saved: no_response_rate.pdf")

    return output_paths


# Backwards compatibility alias
plot_response_latency = plot_turn_transitions


def run_response_latency_analysis(
    all_results: List[Tuple[dict, "Results"]],
    output_dir: Path,
    tick_duration_sec: float = 0.2,
) -> dict:
    """
    Run the full response latency (turn transition) analysis pipeline.

    This is the main entry point for the response latency analysis module.

    Args:
        all_results: List of (params, Results) tuples from load_simulation_results()
        output_dir: Base output directory
        tick_duration_sec: Duration of each tick in seconds

    Returns:
        Dict with paths to output files:
        - raw_path: Path to raw CSV (all turn transitions)
        - analysis_path: Path to analysis CSV
        - plot_path: Path to plot
    """
    import pandas as pd

    # Create output directory
    analysis_dir = output_dir / "response_latency"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Collect all turn transitions
    all_dfs = []
    for params, results in all_results:
        df = extract_turn_transitions_from_results(params, results, tick_duration_sec)
        if not df.empty:
            all_dfs.append(df)

    result_paths = {
        "raw_path": None,
        "analysis_path": None,
        "plot_path": None,
    }

    if not all_dfs:
        logger.warning("No turn transition data found")
        return result_paths

    # Combine all data
    raw_df = pd.concat(all_dfs, ignore_index=True)
    response_count = (raw_df["outcome"] == "response").sum()
    no_response_count = (raw_df["outcome"] == "no_response").sum()
    logger.info(
        f"Collected {len(raw_df)} turn transitions ({response_count} response, {no_response_count} no_response)"
    )

    # Compute analysis
    analysis_df = analyze_turn_transitions(raw_df)

    # Save outputs
    result_paths["raw_path"] = save_turn_transitions_raw(raw_df, analysis_dir)
    result_paths["analysis_path"] = save_turn_transitions_analysis(
        analysis_df, analysis_dir
    )
    result_paths["plot_paths"] = plot_turn_transitions(
        raw_df, analysis_df, analysis_dir
    )

    return result_paths


# =============================================================================
# Interruption Handling Analysis
# =============================================================================


@dataclass
class InterruptionEvent:
    """
    Represents an interruption event where one party started speaking
    while the other was already talking.
    """

    # Experiment metadata
    llm: str = ""
    domain: str = ""
    speech_complexity: str = ""
    provider: str = ""

    # Simulation metadata
    simulation_id: str = ""
    task_id: str = ""

    # Event type
    event_type: str = (
        ""  # "user_interrupts_agent", "agent_interrupts_user", "backchannel"
    )

    # Interrupting party segment info
    interrupter_segment_idx: int = 0
    interrupter_start_tick: int = 0
    interrupter_start_time_sec: float = 0.0
    interrupter_end_tick: int = 0
    interrupter_duration_sec: float = 0.0
    interrupter_transcript: str = ""

    # Interrupted party info
    interrupted_speaking_at_start: bool = True

    # Outcome: did the interrupted party yield (stop speaking)?
    interrupted_yielded: bool = False
    yield_tick: Optional[int] = None
    yield_time_sec: float = 0.0  # Time from interruption start to yield


def extract_interruption_events(
    user_segments: List[UserSpeechSegment],
    agent_segments: List[AgentSpeechSegment],
    ticks: List[Tick],
    tick_duration_sec: float = 0.2,
    no_yield_window_sec: float = 2.0,
    backchannel_yield_window_sec: float = 1.0,
    vocal_tic_yield_window_sec: float = 1.0,
    non_directed_yield_window_sec: float = 1.0,
    vocal_tic_response_window_sec: float = 2.0,
    non_directed_response_window_sec: float = 2.0,
    simulation_id: str = "",
    task_id: str = "",
    llm: str = "",
    domain: str = "",
    speech_complexity: str = "",
    provider: str = "",
    out_of_turn_effects: Optional[List[AudioEffectSegment]] = None,
) -> List[InterruptionEvent]:
    """
    Extract interruption events from speech segments.

    Identifies nine types of events:
    1. user_interrupts_agent: User starts speaking while agent is talking
    2. agent_interrupts_user: Agent starts speaking while user is talking
    3. backchannel: User gives a backchannel while agent is talking
    4. vocal_tic: User vocal tic occurs while agent is talking (agent should NOT yield)
    5. non_directed_speech: User talks to someone else while agent is talking (agent should NOT yield)
    6. agent_responds_to_vocal_tic: Agent starts speaking after vocal tic (agent should NOT respond) - ERROR
    7. agent_responds_to_non_directed: Agent starts speaking after non-directed speech (agent should NOT respond) - ERROR
    8. vocal_tic_silent_correct: Vocal tic when agent silent, agent correctly did NOT respond - CORRECT
    9. non_directed_silent_correct: Non-directed speech when agent silent, agent correctly did NOT respond - CORRECT

    For user_interrupts_agent/agent_interrupts_user: calculates whether the interrupted party yielded.
    For backchannel/vocal_tic/non_directed_speech: yielding is considered incorrect behavior.
    For agent_responds_to_*: the agent incorrectly started speaking in response to non-speech audio.
    For *_silent_correct: the agent correctly stayed silent when receiving non-speech audio.

    Args:
        user_segments: List of user speech segments
        agent_segments: List of agent speech segments
        ticks: List of Tick objects (needed to check speaking status at each tick)
        tick_duration_sec: Duration of each tick in seconds
        no_yield_window_sec: Time window for user interruption yield detection (default: 2.0)
        backchannel_yield_window_sec: Time window for backchannel yield detection (default: 1.0)
        vocal_tic_yield_window_sec: Time window for vocal tic yield detection (default: 1.0)
        non_directed_yield_window_sec: Time window for non-directed yield detection (default: 1.0)
        vocal_tic_response_window_sec: Time window for vocal tic response detection (default: 2.0)
        non_directed_response_window_sec: Time window for non-directed response detection (default: 2.0)
        simulation_id: Simulation identifier
        task_id: Task identifier
        llm: LLM model name
        domain: Domain name
        speech_complexity: Speech complexity level
        provider: Provider name

    Returns:
        List of InterruptionEvent objects
    """
    events = []

    # Calculate yield windows in ticks for each event type
    no_yield_window_ticks = int(no_yield_window_sec / tick_duration_sec)
    backchannel_yield_window_ticks = int(
        backchannel_yield_window_sec / tick_duration_sec
    )
    vocal_tic_yield_window_ticks = int(vocal_tic_yield_window_sec / tick_duration_sec)
    non_directed_yield_window_ticks = int(
        non_directed_yield_window_sec / tick_duration_sec
    )
    vocal_tic_response_window_ticks = int(
        vocal_tic_response_window_sec / tick_duration_sec
    )
    non_directed_response_window_ticks = int(
        non_directed_response_window_sec / tick_duration_sec
    )

    # Base metadata for all events
    base_metadata = {
        "llm": llm,
        "domain": domain,
        "speech_complexity": speech_complexity,
        "provider": provider,
        "simulation_id": simulation_id,
        "task_id": task_id,
    }

    # Process user segments that interrupt the agent (or are backchannels)
    for user_idx, user_seg in enumerate(user_segments):
        if not user_seg.other_speaking_at_start:
            # User didn't start while agent was speaking
            continue

        # Determine event type and corresponding yield window
        # Priority: backchannel > vocal_tic > non_directed_speech > user_interrupts_agent
        # Vocal tics and non-directed speech should NOT cause agent to yield
        if user_seg.is_backchannel:
            event_type = "backchannel"
            yield_window_ticks = backchannel_yield_window_ticks
        elif user_seg.has_vocal_tic:
            event_type = "vocal_tic"
            yield_window_ticks = vocal_tic_yield_window_ticks
        elif user_seg.has_non_directed_speech:
            event_type = "non_directed_speech"
            yield_window_ticks = non_directed_yield_window_ticks
        elif user_seg.is_interruption:
            event_type = "user_interrupts_agent"
            yield_window_ticks = no_yield_window_ticks
        else:
            # User started while agent was speaking but not marked as interruption
            # This could be a transition case - still count it
            event_type = "user_interrupts_agent"
            yield_window_ticks = no_yield_window_ticks

        # Find when agent stopped speaking after user started
        start_tick = user_seg.start_tick
        yield_tick = None
        agent_yielded = False

        # Look ahead up to yield_window_ticks to find when agent stopped
        for i in range(start_tick, min(start_tick + yield_window_ticks, len(ticks))):
            agent_speaking = (
                ticks[i].agent_chunk.contains_speech if ticks[i].agent_chunk else False
            )
            if not agent_speaking:
                agent_yielded = True
                yield_tick = i
                break

        yield_time_sec = (
            (yield_tick - start_tick) * tick_duration_sec if yield_tick else 0.0
        )

        events.append(
            InterruptionEvent(
                **base_metadata,
                event_type=event_type,
                interrupter_segment_idx=user_idx,
                interrupter_start_tick=user_seg.start_tick,
                interrupter_start_time_sec=user_seg.start_time_sec,
                interrupter_end_tick=user_seg.end_tick,
                interrupter_duration_sec=user_seg.duration_sec,
                interrupter_transcript=user_seg.transcript,
                interrupted_speaking_at_start=True,
                interrupted_yielded=agent_yielded,
                yield_tick=yield_tick,
                yield_time_sec=yield_time_sec,
            )
        )

    # Process agent segments that interrupt the user
    for agent_idx, agent_seg in enumerate(agent_segments):
        if not agent_seg.other_speaking_at_start:
            # Agent didn't start while user was speaking
            continue

        event_type = "agent_interrupts_user"

        # Find when user stopped speaking after agent started
        start_tick = agent_seg.start_tick
        yield_tick = None
        user_yielded = False

        # Agent interruption uses no_yield_window for consistency
        for i in range(start_tick, min(start_tick + no_yield_window_ticks, len(ticks))):
            user_speaking = (
                ticks[i].user_chunk.contains_speech if ticks[i].user_chunk else False
            )
            if not user_speaking:
                user_yielded = True
                yield_tick = i
                break

        yield_time_sec = (
            (yield_tick - start_tick) * tick_duration_sec if yield_tick else 0.0
        )

        events.append(
            InterruptionEvent(
                **base_metadata,
                event_type=event_type,
                interrupter_segment_idx=agent_idx,
                interrupter_start_tick=agent_seg.start_tick,
                interrupter_start_time_sec=agent_seg.start_time_sec,
                interrupter_end_tick=agent_seg.end_tick,
                interrupter_duration_sec=agent_seg.duration_sec,
                interrupter_transcript=agent_seg.transcript,
                interrupted_speaking_at_start=True,
                interrupted_yielded=user_yielded,
                yield_tick=yield_tick,
                yield_time_sec=yield_time_sec,
            )
        )

    # Process agent segments that incorrectly respond to vocal tics or non-directed speech
    # These are cases where agent was NOT speaking, vocal tic/non-directed occurred,
    # and agent started speaking in response (incorrect behavior)

    # Find user segments with vocal tics or non-directed speech where agent was NOT speaking
    for user_idx, user_seg in enumerate(user_segments):
        # Skip if agent was speaking at start (already handled above)
        if user_seg.other_speaking_at_start:
            continue

        # Check if this segment has vocal tic or non-directed speech
        if not (user_seg.has_vocal_tic or user_seg.has_non_directed_speech):
            continue

        # Select appropriate response window based on effect type
        if user_seg.has_vocal_tic:
            response_window_ticks = vocal_tic_response_window_ticks
        else:
            response_window_ticks = non_directed_response_window_ticks

        # Look for agent segment that starts within response_window after this segment
        agent_responded = False
        for agent_idx, agent_seg in enumerate(agent_segments):
            # Agent must start after user segment started and within window of user segment end
            if agent_seg.start_tick < user_seg.start_tick:
                continue
            if agent_seg.start_tick > user_seg.end_tick + response_window_ticks:
                continue

            # Determine event type based on the effect
            if user_seg.has_vocal_tic:
                event_type = "agent_responds_to_vocal_tic"
            else:
                event_type = "agent_responds_to_non_directed"

            events.append(
                InterruptionEvent(
                    **base_metadata,
                    event_type=event_type,
                    interrupter_segment_idx=agent_idx,
                    interrupter_start_tick=agent_seg.start_tick,
                    interrupter_start_time_sec=agent_seg.start_time_sec,
                    interrupter_end_tick=agent_seg.end_tick,
                    interrupter_duration_sec=agent_seg.duration_sec,
                    interrupter_transcript=agent_seg.transcript,
                    interrupted_speaking_at_start=False,  # Agent was not speaking
                    interrupted_yielded=True,  # Agent incorrectly responded
                    yield_tick=None,
                    yield_time_sec=0.0,
                )
            )
            agent_responded = True
            # Only count first agent response to avoid duplicates
            break

        # If agent did NOT respond, record a "correct" event (agent correctly stayed silent)
        if not agent_responded:
            if user_seg.has_vocal_tic:
                event_type = "vocal_tic_silent_correct"
            else:
                event_type = "non_directed_silent_correct"

            events.append(
                InterruptionEvent(
                    **base_metadata,
                    event_type=event_type,
                    interrupter_segment_idx=user_idx,
                    interrupter_start_tick=user_seg.start_tick,
                    interrupter_start_time_sec=user_seg.start_time_sec,
                    interrupter_end_tick=user_seg.end_tick,
                    interrupter_duration_sec=user_seg.duration_sec,
                    interrupter_transcript=user_seg.transcript,
                    interrupted_speaking_at_start=False,  # Agent was not speaking
                    interrupted_yielded=False,  # Agent correctly did NOT respond
                    yield_tick=None,
                    yield_time_sec=0.0,
                )
            )

    # Process out-of-turn effects (non-directed speech / vocal tics that occur during silence)
    # These are effects that happen when the user is NOT in a speech segment
    if out_of_turn_effects:
        for effect in out_of_turn_effects:
            if effect.effect_type not in ("non_directed_speech", "vocal_tic"):
                continue

            effect_tick = effect.start_tick

            # Select appropriate windows based on effect type
            if effect.effect_type == "vocal_tic":
                yield_window_sec = vocal_tic_yield_window_sec
                response_window_ticks = vocal_tic_response_window_ticks
            else:
                yield_window_sec = non_directed_yield_window_sec
                response_window_ticks = non_directed_response_window_ticks

            # Check if agent was speaking during this effect
            agent_speaking_during = False
            speaking_agent_seg = None
            for agent_seg in agent_segments:
                if agent_seg.start_tick <= effect_tick < agent_seg.end_tick:
                    agent_speaking_during = True
                    speaking_agent_seg = agent_seg
                    break

            if agent_speaking_during and speaking_agent_seg:
                # Agent was speaking - check if they yielded (stopped) within yield window
                agent_stopped_tick = speaking_agent_seg.end_tick
                time_after_effect = (
                    agent_stopped_tick - effect_tick
                ) * tick_duration_sec

                # If agent stopped within yield window of the effect, they incorrectly yielded
                if time_after_effect <= yield_window_sec:
                    if effect.effect_type == "vocal_tic":
                        event_type = "vocal_tic"
                    else:
                        event_type = "non_directed_speech"

                    events.append(
                        InterruptionEvent(
                            **base_metadata,
                            event_type=event_type,
                            interrupter_segment_idx=-1,  # Not a user segment
                            interrupter_start_tick=effect_tick,
                            interrupter_start_time_sec=effect.start_time_sec,
                            interrupter_end_tick=effect.end_tick,
                            interrupter_duration_sec=effect.duration_sec,
                            interrupter_transcript=effect.text or "",
                            interrupted_speaking_at_start=True,  # Agent was speaking
                            interrupted_yielded=True,  # Agent incorrectly stopped
                            yield_tick=agent_stopped_tick,
                            yield_time_sec=time_after_effect,
                        )
                    )
            else:
                # Agent was NOT speaking - check if they responded (started) within window
                agent_responded = False
                for agent_seg in agent_segments:
                    # Agent must start after effect and within response window
                    if agent_seg.start_tick < effect_tick:
                        continue
                    if agent_seg.start_tick > effect.end_tick + response_window_ticks:
                        continue

                    if effect.effect_type == "vocal_tic":
                        event_type = "agent_responds_to_vocal_tic"
                    else:
                        event_type = "agent_responds_to_non_directed"

                    events.append(
                        InterruptionEvent(
                            **base_metadata,
                            event_type=event_type,
                            interrupter_segment_idx=-1,  # Not a user segment
                            interrupter_start_tick=agent_seg.start_tick,
                            interrupter_start_time_sec=agent_seg.start_time_sec,
                            interrupter_end_tick=agent_seg.end_tick,
                            interrupter_duration_sec=agent_seg.duration_sec,
                            interrupter_transcript=agent_seg.transcript,
                            interrupted_speaking_at_start=False,
                            interrupted_yielded=True,  # Agent incorrectly responded
                            yield_tick=None,
                            yield_time_sec=0.0,
                        )
                    )
                    agent_responded = True
                    # Only count first response
                    break

                # If agent did NOT respond, record a "correct" event
                if not agent_responded:
                    if effect.effect_type == "vocal_tic":
                        event_type = "vocal_tic_silent_correct"
                    else:
                        event_type = "non_directed_silent_correct"

                    events.append(
                        InterruptionEvent(
                            **base_metadata,
                            event_type=event_type,
                            interrupter_segment_idx=-1,  # Not a user segment
                            interrupter_start_tick=effect_tick,
                            interrupter_start_time_sec=effect.start_time_sec,
                            interrupter_end_tick=effect.end_tick,
                            interrupter_duration_sec=effect.duration_sec,
                            interrupter_transcript=effect.text or "",
                            interrupted_speaking_at_start=False,
                            interrupted_yielded=False,  # Agent correctly did NOT respond
                            yield_tick=None,
                            yield_time_sec=0.0,
                        )
                    )

    # Sort by start time
    events.sort(key=lambda e: e.interrupter_start_tick)

    user_interrupts = sum(1 for e in events if e.event_type == "user_interrupts_agent")
    backchannels = sum(1 for e in events if e.event_type == "backchannel")
    vocal_tics = sum(1 for e in events if e.event_type == "vocal_tic")
    non_directed = sum(1 for e in events if e.event_type == "non_directed_speech")
    agent_interrupts = sum(1 for e in events if e.event_type == "agent_interrupts_user")
    agent_responds_vocal = sum(
        1 for e in events if e.event_type == "agent_responds_to_vocal_tic"
    )
    agent_responds_non_dir = sum(
        1 for e in events if e.event_type == "agent_responds_to_non_directed"
    )
    vocal_tic_silent_correct = sum(
        1 for e in events if e.event_type == "vocal_tic_silent_correct"
    )
    non_directed_silent_correct = sum(
        1 for e in events if e.event_type == "non_directed_silent_correct"
    )
    logger.debug(
        f"Extracted {len(events)} interruption events "
        f"({user_interrupts} user_int, {backchannels} backchannel, "
        f"{vocal_tics} vocal_tic, {non_directed} non_directed, {agent_interrupts} agent_int, "
        f"{agent_responds_vocal} resp_vocal, {agent_responds_non_dir} resp_non_dir, "
        f"{vocal_tic_silent_correct} tic_silent_ok, {non_directed_silent_correct} non_dir_silent_ok)"
    )

    return events


def interruption_events_to_dataframe(
    events: List[InterruptionEvent],
) -> "pd.DataFrame":
    """
    Convert interruption events to a pandas DataFrame.

    Args:
        events: List of InterruptionEvent objects

    Returns:
        DataFrame with one row per event
    """
    import pandas as pd

    rows = []
    for e in events:
        rows.append(
            {
                # Experiment metadata
                "llm": e.llm,
                "domain": e.domain,
                "speech_complexity": e.speech_complexity,
                "provider": e.provider,
                # Simulation metadata
                "simulation_id": e.simulation_id,
                "task_id": e.task_id,
                # Event type
                "event_type": e.event_type,
                # Interrupter info
                "interrupter_segment_idx": e.interrupter_segment_idx,
                "interrupter_start_tick": e.interrupter_start_tick,
                "interrupter_start_time_sec": e.interrupter_start_time_sec,
                "interrupter_end_tick": e.interrupter_end_tick,
                "interrupter_duration_sec": e.interrupter_duration_sec,
                "interrupter_transcript": e.interrupter_transcript,
                # Outcome
                "interrupted_yielded": e.interrupted_yielded,
                "yield_tick": e.yield_tick,
                "yield_time_sec": e.yield_time_sec,
            }
        )

    return pd.DataFrame(rows)


def extract_interruptions_from_results(
    params: dict,
    results: "Results",
    tick_duration_sec: float = 0.2,
    no_yield_window_sec: float = 2.0,
    backchannel_yield_window_sec: float = 1.0,
    vocal_tic_yield_window_sec: float = 1.0,
    non_directed_yield_window_sec: float = 1.0,
    vocal_tic_response_window_sec: float = 2.0,
    non_directed_response_window_sec: float = 2.0,
) -> "pd.DataFrame":
    """
    Extract all interruption events from a Results object.

    Args:
        params: Experiment parameters dict
        results: Results object containing simulations
        tick_duration_sec: Duration of each tick in seconds
        no_yield_window_sec: Time window for user interruption yield detection (default: 2.0)
        backchannel_yield_window_sec: Time window for backchannel yield detection (default: 1.0)
        vocal_tic_yield_window_sec: Time window for vocal tic yield detection (default: 1.0)
        non_directed_yield_window_sec: Time window for non-directed yield detection (default: 1.0)
        vocal_tic_response_window_sec: Time window for vocal tic response detection (default: 2.0)
        non_directed_response_window_sec: Time window for non-directed response detection (default: 2.0)

    Returns:
        DataFrame with all interruption events
    """
    import pandas as pd

    all_dfs = []

    llm = params.get("llm", "")
    domain = params.get("domain", "")
    speech_complexity = params.get("speech_complexity", "")
    provider = params.get("provider", "")

    for sim in results.simulations:
        if not sim.ticks:
            continue

        # Extract segments
        user_segs, agent_segs = extract_all_segments(sim.ticks, tick_duration_sec)

        # Extract interruption events
        events = extract_interruption_events(
            user_segs,
            agent_segs,
            sim.ticks,
            tick_duration_sec=tick_duration_sec,
            no_yield_window_sec=no_yield_window_sec,
            backchannel_yield_window_sec=backchannel_yield_window_sec,
            vocal_tic_yield_window_sec=vocal_tic_yield_window_sec,
            non_directed_yield_window_sec=non_directed_yield_window_sec,
            vocal_tic_response_window_sec=vocal_tic_response_window_sec,
            non_directed_response_window_sec=non_directed_response_window_sec,
            simulation_id=sim.id,
            task_id=sim.task_id,
            llm=llm,
            domain=domain,
            speech_complexity=speech_complexity,
            provider=provider,
        )

        if events:
            df = interruption_events_to_dataframe(events)
            all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


def analyze_interruptions(
    raw_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """
    Compute summary statistics for interruption events.

    Aggregates by llm, domain, and speech_complexity.

    Args:
        raw_df: Raw interruption DataFrame

    Returns:
        Summary statistics DataFrame
    """
    import numpy as np
    import pandas as pd

    if raw_df.empty:
        return pd.DataFrame()

    # Group by experiment configuration
    grouped = raw_df.groupby(["llm", "domain", "speech_complexity", "provider"])

    summary_rows = []
    for (llm, domain, complexity, provider), group in grouped:
        # User interrupts agent
        user_int = group[group["event_type"] == "user_interrupts_agent"]
        user_int_yielded = user_int[user_int["interrupted_yielded"]]

        # Backchannels (agent should NOT yield)
        backchannels = group[group["event_type"] == "backchannel"]
        backchannel_agent_continued = backchannels[~backchannels["interrupted_yielded"]]
        backchannel_agent_stopped = backchannels[backchannels["interrupted_yielded"]]

        # Vocal tics (agent should NOT yield)
        vocal_tics = group[group["event_type"] == "vocal_tic"]
        vocal_tic_agent_continued = vocal_tics[~vocal_tics["interrupted_yielded"]]
        vocal_tic_agent_stopped = vocal_tics[vocal_tics["interrupted_yielded"]]

        # Non-directed speech (agent should NOT yield)
        non_directed = group[group["event_type"] == "non_directed_speech"]
        non_directed_agent_continued = non_directed[
            ~non_directed["interrupted_yielded"]
        ]
        non_directed_agent_stopped = non_directed[non_directed["interrupted_yielded"]]

        # Agent responds to vocal tic (agent should NOT respond)
        agent_resp_vocal = group[group["event_type"] == "agent_responds_to_vocal_tic"]

        # Agent responds to non-directed speech (agent should NOT respond)
        agent_resp_non_dir = group[
            group["event_type"] == "agent_responds_to_non_directed"
        ]

        # Vocal tic when agent silent - agent correctly did NOT respond
        vocal_tic_silent_correct = group[
            group["event_type"] == "vocal_tic_silent_correct"
        ]

        # Non-directed when agent silent - agent correctly did NOT respond
        non_directed_silent_correct = group[
            group["event_type"] == "non_directed_silent_correct"
        ]

        # Agent interrupts user
        agent_int = group[group["event_type"] == "agent_interrupts_user"]
        agent_int_yielded = agent_int[agent_int["interrupted_yielded"]]

        # Calculate no-yield counts
        user_int_no_yield = user_int[~user_int["interrupted_yielded"]]
        agent_int_no_yield = agent_int[~agent_int["interrupted_yielded"]]

        summary_rows.append(
            {
                "llm": llm,
                "domain": domain,
                "speech_complexity": complexity,
                "provider": provider,
                # User interrupts agent
                "user_interrupts_count": len(user_int),
                "user_interrupts_agent_yielded": len(user_int_yielded),
                "user_interrupts_agent_no_yield": len(user_int_no_yield),
                "user_interrupts_yield_rate": (
                    len(user_int_yielded) / len(user_int)
                    if len(user_int) > 0
                    else np.nan
                ),
                "user_interrupts_yield_time_mean": (
                    user_int_yielded["yield_time_sec"].mean()
                    if len(user_int_yielded) > 0
                    else np.nan
                ),
                "user_interrupts_yield_time_std": (
                    user_int_yielded["yield_time_sec"].std()
                    if len(user_int_yielded) > 0
                    else np.nan
                ),
                # Backchannels (agent should NOT yield)
                "backchannel_count": len(backchannels),
                "backchannel_agent_continued": len(backchannel_agent_continued),
                "backchannel_agent_stopped": len(backchannel_agent_stopped),
                "backchannel_correct_rate": (
                    len(backchannel_agent_continued) / len(backchannels)
                    if len(backchannels) > 0
                    else np.nan
                ),
                # Vocal tics (agent should NOT yield)
                "vocal_tic_count": len(vocal_tics),
                "vocal_tic_agent_continued": len(vocal_tic_agent_continued),
                "vocal_tic_agent_stopped": len(vocal_tic_agent_stopped),
                "vocal_tic_correct_rate": (
                    len(vocal_tic_agent_continued) / len(vocal_tics)
                    if len(vocal_tics) > 0
                    else np.nan
                ),
                # Non-directed speech (agent should NOT yield)
                "non_directed_count": len(non_directed),
                "non_directed_agent_continued": len(non_directed_agent_continued),
                "non_directed_agent_stopped": len(non_directed_agent_stopped),
                "non_directed_correct_rate": (
                    len(non_directed_agent_continued) / len(non_directed)
                    if len(non_directed) > 0
                    else np.nan
                ),
                # Agent responds to vocal tic (agent should NOT respond)
                "agent_responds_vocal_tic_count": len(agent_resp_vocal),
                # Agent responds to non-directed (agent should NOT respond)
                "agent_responds_non_directed_count": len(agent_resp_non_dir),
                # Vocal tic when agent silent - totals and rates
                "vocal_tic_silent_total": len(vocal_tic_silent_correct)
                + len(agent_resp_vocal),
                "vocal_tic_silent_correct": len(vocal_tic_silent_correct),
                "vocal_tic_silent_incorrect": len(agent_resp_vocal),
                "vocal_tic_silent_correct_rate": (
                    len(vocal_tic_silent_correct)
                    / (len(vocal_tic_silent_correct) + len(agent_resp_vocal))
                    if (len(vocal_tic_silent_correct) + len(agent_resp_vocal)) > 0
                    else np.nan
                ),
                # Non-directed when agent silent - totals and rates
                "non_directed_silent_total": len(non_directed_silent_correct)
                + len(agent_resp_non_dir),
                "non_directed_silent_correct": len(non_directed_silent_correct),
                "non_directed_silent_incorrect": len(agent_resp_non_dir),
                "non_directed_silent_correct_rate": (
                    len(non_directed_silent_correct)
                    / (len(non_directed_silent_correct) + len(agent_resp_non_dir))
                    if (len(non_directed_silent_correct) + len(agent_resp_non_dir)) > 0
                    else np.nan
                ),
                # Agent interrupts user
                "agent_interrupts_count": len(agent_int),
                "agent_interrupts_user_yielded": len(agent_int_yielded),
                "agent_interrupts_user_no_yield": len(agent_int_no_yield),
                "agent_interrupts_yield_rate": (
                    len(agent_int_yielded) / len(agent_int)
                    if len(agent_int) > 0
                    else np.nan
                ),
                "agent_interrupts_yield_time_mean": (
                    agent_int_yielded["yield_time_sec"].mean()
                    if len(agent_int_yielded) > 0
                    else np.nan
                ),
                "agent_interrupts_yield_time_std": (
                    agent_int_yielded["yield_time_sec"].std()
                    if len(agent_int_yielded) > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(summary_rows)


def save_interruptions_raw(
    raw_df: "pd.DataFrame",
    output_dir: Path,
) -> Path:
    """Save raw interruption data to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_dir.name}_raw.csv"
    raw_df.to_csv(output_path, index=False)
    logger.info(f"Saved raw interruption data to {output_path}")
    return output_path


def save_interruptions_analysis(
    analysis_df: "pd.DataFrame",
    output_dir: Path,
) -> Path:
    """Save interruption analysis to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_dir.name}_analysis.csv"
    analysis_df.to_csv(output_path, index=False)
    logger.info(f"Saved interruption analysis to {output_path}")
    return output_path


def plot_interruptions(
    raw_df: "pd.DataFrame" = None,
    analysis_df: "pd.DataFrame" = None,
    output_dir: Path = None,
    yield_window_sec: float = 2.0,
    llms: List[str] = None,
    domains: List[str] = None,
    complexities: List[str] = None,
) -> List[Path]:
    """
    Create interruption handling visualizations.

    LLM on x-axis, bars grouped by complexity (control=hatched, regular=solid).

    Creates separate figures for:
    1. Agent yield rate (when user interrupts)
    2. Agent yield time (for cases where agent yielded)
    3. Backchannel handling (agent correctly continues vs incorrectly stops)

    Args:
        raw_df: Raw interruption DataFrame (optional for plots-only mode)
        analysis_df: Analysis DataFrame
        output_dir: Directory to save the figures
        yield_window_sec: Yield detection window (for display in titles)
        llms: List of LLM names (optional, extracted from analysis_df if not provided)
        domains: List of domains (optional, extracted from analysis_df if not provided)
        complexities: List of complexities (optional, extracted from analysis_df if not provided)

    Returns:
        List of paths to saved figures
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Import shared styling
    from experiments.tau_voice.exp.plot_style import (
        BAR_STYLE,
        COMPLEXITY_STYLES,
        get_legend_patch,
        get_llm_color,
        style_axis,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    if analysis_df is None or analysis_df.empty:
        logger.warning("No data for interruption plots")
        return output_paths

    # Get unique values if not provided
    if llms is None:
        llms = sorted(analysis_df["llm"].unique())
    if domains is None:
        domains = sorted(analysis_df["domain"].unique())
    if complexities is None:
        complexities = sorted(analysis_df["speech_complexity"].unique())

    if not llms or not domains or not complexities:
        logger.warning("Insufficient data for interruption plots")
        return output_paths

    n_domains = len(domains)
    n_complexities = len(complexities)
    n_llms = len(llms)
    group_width = 0.8
    bar_width = group_width / n_complexities

    # =========================================================================
    # Figure 1: Agent Yield Rate - LLM on x-axis, grouped by complexity
    # =========================================================================
    fig, axes = plt.subplots(n_domains, 1, figsize=(10, 5 * n_domains), squeeze=False)
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains):
        ax = axes[d_idx]
        domain_df = analysis_df[analysis_df["domain"] == domain]

        x = np.arange(n_llms)

        for c_idx, complexity in enumerate(complexities):
            values = []
            colors = []

            for llm in llms:
                subset = domain_df[
                    (domain_df["llm"] == llm)
                    & (domain_df["speech_complexity"] == complexity)
                ]
                if len(subset) > 0:
                    rate = subset["user_interrupts_yield_rate"].values[0]
                    values.append(rate * 100 if not np.isnan(rate) else np.nan)
                else:
                    values.append(np.nan)
                colors.append(get_llm_color(llm))

            bar_offset = (c_idx - (n_complexities - 1) / 2) * bar_width
            x_pos = x + bar_offset

            style = COMPLEXITY_STYLES.get(complexity, {"alpha": 0.8, "hatch": ""})
            ax.bar(
                x_pos,
                values,
                bar_width * 0.9,
                color=colors,
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )

        ax.set_xlabel("LLM", fontsize=11)
        ax.set_ylabel("Yield Rate (%)", fontsize=11)
        ax.set_title(f"{domain.capitalize()}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [llm.split(":")[-1][:12] if ":" in llm else llm[:12] for llm in llms],
            fontsize=9,
            rotation=45,
            ha="right",
        )
        ax.set_ylim(0, 105)
        style_axis(ax)

    # Add complexity legend
    legend_patches = [get_legend_patch(c) for c in complexities]
    axes[-1].legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=9,
        title="Complexity",
    )

    fig.suptitle(
        f"Agent Yield Rate (User Interrupts Agent)\n[No yield = agent still speaking after {yield_window_sec:.0f}s]",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = output_dir / "agent_yield_rate.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    output_paths.append(output_path)
    logger.info("Saved: agent_yield_rate.pdf")

    # =========================================================================
    # Figure 2: Agent Yield Time - LLM on x-axis, grouped by complexity
    # =========================================================================
    fig, axes = plt.subplots(n_domains, 1, figsize=(10, 5 * n_domains), squeeze=False)
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains):
        ax = axes[d_idx]
        domain_df = analysis_df[analysis_df["domain"] == domain]

        x = np.arange(n_llms)

        for c_idx, complexity in enumerate(complexities):
            values = []
            stds = []
            colors = []

            for llm in llms:
                subset = domain_df[
                    (domain_df["llm"] == llm)
                    & (domain_df["speech_complexity"] == complexity)
                ]
                if len(subset) > 0:
                    values.append(subset["user_interrupts_yield_time_mean"].values[0])
                    std_val = (
                        subset["user_interrupts_yield_time_std"].values[0]
                        if "user_interrupts_yield_time_std" in subset.columns
                        else 0
                    )
                    stds.append(std_val if not np.isnan(std_val) else 0)
                else:
                    values.append(np.nan)
                    stds.append(0)
                colors.append(get_llm_color(llm))

            bar_offset = (c_idx - (n_complexities - 1) / 2) * bar_width
            x_pos = x + bar_offset

            style = COMPLEXITY_STYLES.get(complexity, {"alpha": 0.8, "hatch": ""})
            ax.bar(
                x_pos,
                values,
                bar_width * 0.9,
                yerr=stds,
                capsize=2,
                color=colors,
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )

        ax.set_xlabel("LLM", fontsize=11)
        ax.set_ylabel("Yield Time (s)", fontsize=11)
        ax.set_title(f"{domain.capitalize()}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [llm.split(":")[-1][:12] if ":" in llm else llm[:12] for llm in llms],
            fontsize=9,
            rotation=45,
            ha="right",
        )
        style_axis(ax)

    # Align y-axis scales
    max_ylim = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, max_ylim * 1.1)

    # Add complexity legend
    legend_patches = [get_legend_patch(c) for c in complexities]
    axes[-1].legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=9,
        title="Complexity",
    )

    fig.suptitle(
        f"Agent Yield Time (When Agent Yielded Within {yield_window_sec:.0f}s)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = output_dir / "agent_yield_time.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    output_paths.append(output_path)
    logger.info("Saved: agent_yield_time.pdf")

    # =========================================================================
    # Figure 3: Backchannel Handling - LLM on x-axis, grouped by complexity
    # =========================================================================
    fig, axes = plt.subplots(n_domains, 1, figsize=(10, 5 * n_domains), squeeze=False)
    axes = axes[:, 0]

    for d_idx, domain in enumerate(domains):
        ax = axes[d_idx]
        domain_df = analysis_df[analysis_df["domain"] == domain]

        x = np.arange(n_llms)

        for c_idx, complexity in enumerate(complexities):
            values = []
            colors = []

            for llm in llms:
                subset = domain_df[
                    (domain_df["llm"] == llm)
                    & (domain_df["speech_complexity"] == complexity)
                ]
                if len(subset) > 0:
                    rate = subset["backchannel_correct_rate"].values[0]
                    values.append(rate * 100 if not np.isnan(rate) else np.nan)
                else:
                    values.append(np.nan)
                colors.append(get_llm_color(llm))

            bar_offset = (c_idx - (n_complexities - 1) / 2) * bar_width
            x_pos = x + bar_offset

            style = COMPLEXITY_STYLES.get(complexity, {"alpha": 0.8, "hatch": ""})
            ax.bar(
                x_pos,
                values,
                bar_width * 0.9,
                color=colors,
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor=BAR_STYLE["edgecolor"],
                linewidth=BAR_STYLE["linewidth"],
            )

        ax.set_xlabel("LLM", fontsize=11)
        ax.set_ylabel("Correct Rate (%)", fontsize=11)
        ax.set_title(f"{domain.capitalize()}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [llm.split(":")[-1][:12] if ":" in llm else llm[:12] for llm in llms],
            fontsize=9,
            rotation=45,
            ha="right",
        )
        ax.set_ylim(0, 105)
        style_axis(ax)

    # Add complexity legend
    legend_patches = [get_legend_patch(c) for c in complexities]
    axes[-1].legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=9,
        title="Complexity",
    )

    fig.suptitle(
        "Backchannel Handling (Agent Correctly Continues)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = output_dir / "backchannel_handling.pdf"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    output_paths.append(output_path)
    logger.info("Saved: backchannel_handling.pdf")

    return output_paths


# =============================================================================
# Unified Voice Quality Analysis
# =============================================================================
# This section provides a unified approach to voice quality metrics that ensures
# consistency between the speech timeline visualization and the computed metrics.
# All events shown on the timeline will match exactly with the analysis tables.


@dataclass
class VoiceQualityEvent:
    """
    Unified event for voice quality analysis.

    This dataclass captures all event types that are shown on the speech timeline
    and used for computing voice quality metrics. Using a single event type ensures
    consistency between visualization and analysis.

    Event categories:
    1. Response events: Agent's response to user speech
       - "response": Agent responded (success)
       - "no_response": Agent failed to respond (error)

    2. Yield events: Agent yielding when user interrupts
       - "yield": Agent yielded to user interruption (success)
       - "no_yield": Agent failed to yield (error)

    3. Backchannel events: Agent handling of user backchannels
       - "backchannel_correct": Agent correctly continued speaking (success)
       - "backchannel_error": Agent incorrectly stopped (error)

    4. Vocal tic events: Agent handling of vocal tics ("um", "uh")
       - "vocal_tic_correct": Agent correctly ignored (success)
       - "vocal_tic_error": Agent incorrectly responded/yielded (error)

    5. Non-directed speech events: Agent handling of speech not directed at it
       - "non_directed_correct": Agent correctly ignored (success)
       - "non_directed_error": Agent incorrectly responded/yielded (error)
    """

    # Event identification
    event_category: (
        str  # "response", "yield", "backchannel", "vocal_tic", "non_directed"
    )
    event_type: str  # Specific type within category
    is_error: bool  # True if this is an error (agent behaved incorrectly)

    # Timing (for latency calculations)
    latency_sec: Optional[float] = None  # Response latency or yield latency

    # Experiment metadata
    llm: str = ""
    domain: str = ""
    speech_complexity: str = ""
    provider: str = ""

    # Simulation metadata
    simulation_id: str = ""
    task_id: str = ""

    # Event timing
    event_time_sec: float = 0.0
    event_tick: int = 0

    # Additional context
    transcript: str = ""


def extract_voice_quality_events_from_simulation(
    ticks: List[Tick],
    tick_duration_sec: float = 0.2,
    no_yield_window_sec: float = 2.0,
    backchannel_yield_window_sec: float = 1.0,
    vocal_tic_yield_window_sec: float = 1.0,
    non_directed_yield_window_sec: float = 1.0,
    vocal_tic_response_window_sec: float = 2.0,
    non_directed_response_window_sec: float = 2.0,
    simulation_id: str = "",
    task_id: str = "",
    llm: str = "",
    domain: str = "",
    speech_complexity: str = "",
    provider: str = "",
) -> List[VoiceQualityEvent]:
    """
    Extract all voice quality events from a single simulation.

    This function applies the same processing as the speech timeline visualization:
    1. Filters end-of-conversation artifacts
    2. Extracts speech segments
    3. Extracts out-of-turn effects
    4. Converts all events to unified VoiceQualityEvent format

    Args:
        ticks: List of Tick objects from a simulation
        tick_duration_sec: Duration of each tick in seconds
        no_yield_window_sec: Time window for user interruption yield detection (default: 2.0)
        backchannel_yield_window_sec: Time window for backchannel yield detection (default: 1.0)
        vocal_tic_yield_window_sec: Time window for vocal tic yield detection (default: 1.0)
        non_directed_yield_window_sec: Time window for non-directed yield detection (default: 1.0)
        vocal_tic_response_window_sec: Time window for vocal tic response detection (default: 2.0)
        non_directed_response_window_sec: Time window for non-directed response detection (default: 2.0)
        simulation_id: Simulation identifier
        task_id: Task identifier
        llm: LLM model name
        domain: Domain name
        speech_complexity: Speech complexity level
        provider: Provider name

    Returns:
        List of VoiceQualityEvent objects
    """
    events: List[VoiceQualityEvent] = []

    # CRITICAL: Apply the same tick filtering as the timeline visualization
    filtered_ticks = filter_end_of_conversation_ticks(ticks)

    # Extract speech segments from filtered ticks
    user_segs, agent_segs = extract_all_segments(filtered_ticks, tick_duration_sec)

    # Extract out-of-turn effects (effects during gaps between speech)
    out_of_turn_effects = extract_out_of_turn_effects(filtered_ticks, tick_duration_sec)

    # Base metadata for all events
    base_metadata = {
        "llm": llm,
        "domain": domain,
        "speech_complexity": speech_complexity,
        "provider": provider,
        "simulation_id": simulation_id,
        "task_id": task_id,
    }

    # =========================================================================
    # 1. Response Events (from turn transitions)
    # =========================================================================
    turn_transitions = extract_turn_transitions(
        user_segs,
        agent_segs,
        simulation_id=simulation_id,
        task_id=task_id,
        llm=llm,
        domain=domain,
        speech_complexity=speech_complexity,
        provider=provider,
    )

    for tt in turn_transitions:
        if tt.outcome == "response":
            events.append(
                VoiceQualityEvent(
                    event_category="response",
                    event_type="response",
                    is_error=False,
                    latency_sec=tt.gap_sec,
                    event_time_sec=tt.user_end_time_sec,
                    event_tick=tt.user_end_tick,
                    transcript=tt.user_transcript,
                    **base_metadata,
                )
            )
        elif tt.outcome == "no_response":
            events.append(
                VoiceQualityEvent(
                    event_category="response",
                    event_type="no_response",
                    is_error=True,
                    latency_sec=None,
                    event_time_sec=tt.user_end_time_sec,
                    event_tick=tt.user_end_tick,
                    transcript=tt.user_transcript,
                    **base_metadata,
                )
            )

    # =========================================================================
    # 2. Interruption Events (yield, backchannel, vocal_tic, non_directed)
    # =========================================================================
    interruption_events = extract_interruption_events(
        user_segs,
        agent_segs,
        filtered_ticks,  # Use filtered ticks!
        tick_duration_sec=tick_duration_sec,
        no_yield_window_sec=no_yield_window_sec,
        backchannel_yield_window_sec=backchannel_yield_window_sec,
        vocal_tic_yield_window_sec=vocal_tic_yield_window_sec,
        non_directed_yield_window_sec=non_directed_yield_window_sec,
        vocal_tic_response_window_sec=vocal_tic_response_window_sec,
        non_directed_response_window_sec=non_directed_response_window_sec,
        simulation_id=simulation_id,
        task_id=task_id,
        llm=llm,
        domain=domain,
        speech_complexity=speech_complexity,
        provider=provider,
        out_of_turn_effects=out_of_turn_effects,  # Pass out-of-turn effects!
    )

    for ie in interruption_events:
        if ie.event_type == "user_interrupts_agent":
            # User interrupted agent - did agent yield?
            if ie.interrupted_yielded:
                events.append(
                    VoiceQualityEvent(
                        event_category="yield",
                        event_type="yield",
                        is_error=False,
                        latency_sec=ie.yield_time_sec,
                        event_time_sec=ie.interrupter_start_time_sec,
                        event_tick=ie.interrupter_start_tick,
                        transcript=ie.interrupter_transcript,
                        **base_metadata,
                    )
                )
            else:
                events.append(
                    VoiceQualityEvent(
                        event_category="yield",
                        event_type="no_yield",
                        is_error=True,
                        latency_sec=None,
                        event_time_sec=ie.interrupter_start_time_sec,
                        event_tick=ie.interrupter_start_tick,
                        transcript=ie.interrupter_transcript,
                        **base_metadata,
                    )
                )

        elif ie.event_type == "backchannel":
            # Backchannel - agent should NOT yield
            if ie.interrupted_yielded:
                events.append(
                    VoiceQualityEvent(
                        event_category="backchannel",
                        event_type="backchannel_error",
                        is_error=True,
                        latency_sec=None,
                        event_time_sec=ie.interrupter_start_time_sec,
                        event_tick=ie.interrupter_start_tick,
                        transcript=ie.interrupter_transcript,
                        **base_metadata,
                    )
                )
            else:
                events.append(
                    VoiceQualityEvent(
                        event_category="backchannel",
                        event_type="backchannel_correct",
                        is_error=False,
                        latency_sec=None,
                        event_time_sec=ie.interrupter_start_time_sec,
                        event_tick=ie.interrupter_start_tick,
                        transcript=ie.interrupter_transcript,
                        **base_metadata,
                    )
                )

        elif ie.event_type == "vocal_tic":
            # Vocal tic while agent speaking - agent should NOT yield
            if ie.interrupted_yielded:
                events.append(
                    VoiceQualityEvent(
                        event_category="vocal_tic",
                        event_type="vocal_tic_error",
                        is_error=True,
                        latency_sec=None,
                        event_time_sec=ie.interrupter_start_time_sec,
                        event_tick=ie.interrupter_start_tick,
                        transcript=ie.interrupter_transcript,
                        **base_metadata,
                    )
                )
            else:
                events.append(
                    VoiceQualityEvent(
                        event_category="vocal_tic",
                        event_type="vocal_tic_correct",
                        is_error=False,
                        latency_sec=None,
                        event_time_sec=ie.interrupter_start_time_sec,
                        event_tick=ie.interrupter_start_tick,
                        transcript=ie.interrupter_transcript,
                        **base_metadata,
                    )
                )

        elif ie.event_type == "non_directed_speech":
            # Non-directed speech while agent speaking - agent should NOT yield
            if ie.interrupted_yielded:
                events.append(
                    VoiceQualityEvent(
                        event_category="non_directed",
                        event_type="non_directed_error",
                        is_error=True,
                        latency_sec=None,
                        event_time_sec=ie.interrupter_start_time_sec,
                        event_tick=ie.interrupter_start_tick,
                        transcript=ie.interrupter_transcript,
                        **base_metadata,
                    )
                )
            else:
                events.append(
                    VoiceQualityEvent(
                        event_category="non_directed",
                        event_type="non_directed_correct",
                        is_error=False,
                        latency_sec=None,
                        event_time_sec=ie.interrupter_start_time_sec,
                        event_tick=ie.interrupter_start_tick,
                        transcript=ie.interrupter_transcript,
                        **base_metadata,
                    )
                )

        elif ie.event_type == "agent_responds_to_vocal_tic":
            # Agent incorrectly responded to vocal tic (agent was silent)
            events.append(
                VoiceQualityEvent(
                    event_category="vocal_tic",
                    event_type="vocal_tic_error",
                    is_error=True,
                    latency_sec=None,
                    event_time_sec=ie.interrupter_start_time_sec,
                    event_tick=ie.interrupter_start_tick,
                    transcript=ie.interrupter_transcript,
                    **base_metadata,
                )
            )

        elif ie.event_type == "vocal_tic_silent_correct":
            # Agent correctly ignored vocal tic (agent was silent)
            events.append(
                VoiceQualityEvent(
                    event_category="vocal_tic",
                    event_type="vocal_tic_correct",
                    is_error=False,
                    latency_sec=None,
                    event_time_sec=ie.interrupter_start_time_sec,
                    event_tick=ie.interrupter_start_tick,
                    transcript=ie.interrupter_transcript,
                    **base_metadata,
                )
            )

        elif ie.event_type == "agent_responds_to_non_directed":
            # Agent incorrectly responded to non-directed speech (agent was silent)
            events.append(
                VoiceQualityEvent(
                    event_category="non_directed",
                    event_type="non_directed_error",
                    is_error=True,
                    latency_sec=None,
                    event_time_sec=ie.interrupter_start_time_sec,
                    event_tick=ie.interrupter_start_tick,
                    transcript=ie.interrupter_transcript,
                    **base_metadata,
                )
            )

        elif ie.event_type == "non_directed_silent_correct":
            # Agent correctly ignored non-directed speech (agent was silent)
            events.append(
                VoiceQualityEvent(
                    event_category="non_directed",
                    event_type="non_directed_correct",
                    is_error=False,
                    latency_sec=None,
                    event_time_sec=ie.interrupter_start_time_sec,
                    event_tick=ie.interrupter_start_tick,
                    transcript=ie.interrupter_transcript,
                    **base_metadata,
                )
            )

        # Note: "agent_interrupts_user" events are tracked but not used in metrics
        # since we focus on agent errors, not user behavior

    # Sort by event time
    events.sort(key=lambda e: e.event_time_sec)

    return events


def voice_quality_events_to_dataframe(
    events: List[VoiceQualityEvent],
) -> "pd.DataFrame":
    """
    Convert voice quality events to a pandas DataFrame.

    Args:
        events: List of VoiceQualityEvent objects

    Returns:
        DataFrame with one row per event
    """
    import pandas as pd

    rows = []
    for e in events:
        rows.append(
            {
                "event_category": e.event_category,
                "event_type": e.event_type,
                "is_error": e.is_error,
                "latency_sec": e.latency_sec,
                "llm": e.llm,
                "domain": e.domain,
                "speech_complexity": e.speech_complexity,
                "provider": e.provider,
                "simulation_id": e.simulation_id,
                "task_id": e.task_id,
                "event_time_sec": e.event_time_sec,
                "event_tick": e.event_tick,
                "transcript": e.transcript,
            }
        )

    return pd.DataFrame(rows)


def compute_voice_quality_metrics(
    raw_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """
    Compute voice quality metrics from raw event data.

    Computes the following metrics (aggregated by llm, domain, speech_complexity, provider):
    1. response_rate: % of user turns that got an agent response
    2. response_latency_mean: Mean response latency (seconds)
    3. yield_rate: % of user interruptions where agent yielded
    4. yield_latency_mean: Mean yield latency (seconds)
    5. backchannel_error_rate: % of backchannels that incorrectly stopped agent
    6. vocal_tic_error_rate: % of vocal tics that incorrectly triggered agent
    7. non_directed_error_rate: % of non-directed speech that incorrectly triggered agent

    Args:
        raw_df: Raw event DataFrame from voice_quality_events_to_dataframe()

    Returns:
        Summary metrics DataFrame
    """
    import numpy as np
    import pandas as pd

    if raw_df.empty:
        return pd.DataFrame()

    # Group by experiment configuration
    grouped = raw_df.groupby(["llm", "domain", "speech_complexity", "provider"])

    summary_rows = []
    for (llm, domain, complexity, provider), group in grouped:
        # Response events
        response_events = group[group["event_category"] == "response"]
        n_response = (response_events["event_type"] == "response").sum()
        n_no_response = (response_events["event_type"] == "no_response").sum()
        n_response_total = n_response + n_no_response
        response_rate = (
            n_response / n_response_total if n_response_total > 0 else np.nan
        )

        response_latencies = response_events[
            response_events["event_type"] == "response"
        ]["latency_sec"].dropna()
        response_latency_mean = (
            response_latencies.mean() if len(response_latencies) > 0 else np.nan
        )
        response_latency_std = (
            response_latencies.std() if len(response_latencies) > 0 else np.nan
        )

        # Yield events
        yield_events = group[group["event_category"] == "yield"]
        n_yield = (yield_events["event_type"] == "yield").sum()
        n_no_yield = (yield_events["event_type"] == "no_yield").sum()
        n_yield_total = n_yield + n_no_yield
        yield_rate = n_yield / n_yield_total if n_yield_total > 0 else np.nan

        yield_latencies = yield_events[yield_events["event_type"] == "yield"][
            "latency_sec"
        ].dropna()
        yield_latency_mean = (
            yield_latencies.mean() if len(yield_latencies) > 0 else np.nan
        )
        yield_latency_std = (
            yield_latencies.std() if len(yield_latencies) > 0 else np.nan
        )

        # Backchannel events
        backchannel_events = group[group["event_category"] == "backchannel"]
        n_backchannel_correct = (
            backchannel_events["event_type"] == "backchannel_correct"
        ).sum()
        n_backchannel_error = (
            backchannel_events["event_type"] == "backchannel_error"
        ).sum()
        n_backchannel_total = n_backchannel_correct + n_backchannel_error
        backchannel_error_rate = (
            n_backchannel_error / n_backchannel_total
            if n_backchannel_total > 0
            else np.nan
        )

        # Vocal tic events
        vocal_tic_events = group[group["event_category"] == "vocal_tic"]
        n_vocal_tic_correct = (
            vocal_tic_events["event_type"] == "vocal_tic_correct"
        ).sum()
        n_vocal_tic_error = (vocal_tic_events["event_type"] == "vocal_tic_error").sum()
        n_vocal_tic_total = n_vocal_tic_correct + n_vocal_tic_error
        vocal_tic_error_rate = (
            n_vocal_tic_error / n_vocal_tic_total if n_vocal_tic_total > 0 else np.nan
        )

        # Non-directed speech events
        non_directed_events = group[group["event_category"] == "non_directed"]
        n_non_directed_correct = (
            non_directed_events["event_type"] == "non_directed_correct"
        ).sum()
        n_non_directed_error = (
            non_directed_events["event_type"] == "non_directed_error"
        ).sum()
        n_non_directed_total = n_non_directed_correct + n_non_directed_error
        non_directed_error_rate = (
            n_non_directed_error / n_non_directed_total
            if n_non_directed_total > 0
            else np.nan
        )

        summary_rows.append(
            {
                "llm": llm,
                "domain": domain,
                "speech_complexity": complexity,
                "provider": provider,
                # Response metrics
                "response_count": n_response,
                "no_response_count": n_no_response,
                "response_total": n_response_total,
                "response_rate": response_rate,
                "response_latency_mean": response_latency_mean,
                "response_latency_std": response_latency_std,
                # Yield metrics
                "yield_count": n_yield,
                "no_yield_count": n_no_yield,
                "yield_total": n_yield_total,
                "yield_rate": yield_rate,
                "yield_latency_mean": yield_latency_mean,
                "yield_latency_std": yield_latency_std,
                # Backchannel metrics
                "backchannel_correct_count": n_backchannel_correct,
                "backchannel_error_count": n_backchannel_error,
                "backchannel_total": n_backchannel_total,
                "backchannel_error_rate": backchannel_error_rate,
                # Vocal tic metrics
                "vocal_tic_correct_count": n_vocal_tic_correct,
                "vocal_tic_error_count": n_vocal_tic_error,
                "vocal_tic_total": n_vocal_tic_total,
                "vocal_tic_error_rate": vocal_tic_error_rate,
                # Non-directed metrics
                "non_directed_correct_count": n_non_directed_correct,
                "non_directed_error_count": n_non_directed_error,
                "non_directed_total": n_non_directed_total,
                "non_directed_error_rate": non_directed_error_rate,
            }
        )

    return pd.DataFrame(summary_rows)


def plot_voice_quality_summary(
    analysis_df: "pd.DataFrame",
    output_path: Optional[Path] = None,
) -> "plt.Figure":
    """
    Create a single summary plot showing all voice quality metrics.

    Shows a grouped bar chart with:
    - Response rate (higher is better)
    - Yield rate (higher is better)
    - Error rates for backchannel, vocal_tic, non_directed (lower is better)

    Args:
        analysis_df: Analysis DataFrame from compute_voice_quality_metrics()
        output_path: Optional path to save the figure

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from experiments.tau_voice.exp.plot_style import get_short_llm_name, style_axis

    if analysis_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig

    # Get unique LLMs
    llms = analysis_df["llm"].unique()
    n_llms = len(llms)

    # Create figure with two subplots: success rates and error rates
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Colors for different metrics
    colors_success = ["#2563eb", "#10b981"]  # Blue, Green
    colors_error = ["#dc2626", "#f59e0b", "#8b5cf6"]  # Red, Orange, Violet

    x = np.arange(n_llms)
    width = 0.35

    # Subplot 1: Success rates (Response Rate, Yield Rate)
    response_rates = []
    yield_rates = []
    for llm in llms:
        llm_data = analysis_df[analysis_df["llm"] == llm]
        response_rates.append(llm_data["response_rate"].mean() * 100)
        yield_rates.append(llm_data["yield_rate"].mean() * 100)

    bars1 = ax1.bar(
        x - width / 2,
        response_rates,
        width,
        label="Response Rate",
        color=colors_success[0],
    )
    bars2 = ax1.bar(
        x + width / 2, yield_rates, width, label="Yield Rate", color=colors_success[1]
    )

    ax1.set_ylabel("Rate (%)", fontsize=11)
    ax1.set_title("Success Rates (Higher is Better)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [get_short_llm_name(llm) for llm in llms], rotation=45, ha="right"
    )
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 105)
    ax1.axhline(y=100, color="gray", linestyle="--", alpha=0.3)
    style_axis(ax1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.annotate(
                    f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Subplot 2: Error rates (lower is better)
    backchannel_errors = []
    vocal_tic_errors = []
    non_directed_errors = []
    for llm in llms:
        llm_data = analysis_df[analysis_df["llm"] == llm]
        backchannel_errors.append(llm_data["backchannel_error_rate"].mean() * 100)
        vocal_tic_errors.append(llm_data["vocal_tic_error_rate"].mean() * 100)
        non_directed_errors.append(llm_data["non_directed_error_rate"].mean() * 100)

    width_error = 0.25
    bars3 = ax2.bar(
        x - width_error,
        backchannel_errors,
        width_error,
        label="Backchannel Error",
        color=colors_error[0],
    )
    bars4 = ax2.bar(
        x,
        vocal_tic_errors,
        width_error,
        label="Vocal Tic Error",
        color=colors_error[1],
    )
    bars5 = ax2.bar(
        x + width_error,
        non_directed_errors,
        width_error,
        label="Non-Directed Error",
        color=colors_error[2],
    )

    ax2.set_ylabel("Error Rate (%)", fontsize=11)
    ax2.set_title("Error Rates (Lower is Better)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [get_short_llm_name(llm) for llm in llms], rotation=45, ha="right"
    )
    ax2.legend(loc="upper right")
    # Handle NaN values in error rates
    all_errors = [
        e
        for e in backchannel_errors + vocal_tic_errors + non_directed_errors
        if not np.isnan(e)
    ]
    max_error = max(all_errors) if all_errors else 0
    ax2.set_ylim(0, max(100, max_error * 1.1))
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    style_axis(ax2)

    # Add value labels
    for bars in [bars3, bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax2.annotate(
                    f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        logger.info(f"Saved voice quality summary plot to {output_path}")

    return fig


def run_voice_quality_analysis(
    all_results: List[Tuple[dict, "Results"]],
    output_dir: Path,
    tick_duration_sec: float = 0.2,
    no_yield_window_sec: float = 2.0,
    backchannel_yield_window_sec: float = 1.0,
    vocal_tic_yield_window_sec: float = 1.0,
    non_directed_yield_window_sec: float = 1.0,
    vocal_tic_response_window_sec: float = 2.0,
    non_directed_response_window_sec: float = 2.0,
) -> dict:
    """
    Run the unified voice quality analysis pipeline.

    This produces:
    - voice_quality_raw.csv: All events (one row per event)
    - voice_quality_analysis.csv: Aggregated metrics
    - voice_quality_summary.pdf: Single summary plot

    Args:
        all_results: List of (params, Results) tuples from load_simulation_results()
        output_dir: Base output directory
        tick_duration_sec: Duration of each tick in seconds
        no_yield_window_sec: Time window for user interruption yield detection (default: 2.0)
        backchannel_yield_window_sec: Time window for backchannel yield detection (default: 1.0)
        vocal_tic_yield_window_sec: Time window for vocal tic yield detection (default: 1.0)
        non_directed_yield_window_sec: Time window for non-directed yield detection (default: 1.0)
        vocal_tic_response_window_sec: Time window for vocal tic response detection (default: 2.0)
        non_directed_response_window_sec: Time window for non-directed response detection (default: 2.0)

    Returns:
        Dict with paths to output files
    """

    # Create output directory
    analysis_dir = output_dir / "voice_quality"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Collect all events
    all_events: List[VoiceQualityEvent] = []

    for params, results in all_results:
        llm = params.get("llm", "")
        domain = params.get("domain", "")
        speech_complexity = params.get("speech_complexity", "")
        provider = params.get("provider", "")

        for sim in results.simulations:
            if not sim.ticks:
                continue

            events = extract_voice_quality_events_from_simulation(
                sim.ticks,
                tick_duration_sec=tick_duration_sec,
                no_yield_window_sec=no_yield_window_sec,
                backchannel_yield_window_sec=backchannel_yield_window_sec,
                vocal_tic_yield_window_sec=vocal_tic_yield_window_sec,
                non_directed_yield_window_sec=non_directed_yield_window_sec,
                vocal_tic_response_window_sec=vocal_tic_response_window_sec,
                non_directed_response_window_sec=non_directed_response_window_sec,
                simulation_id=sim.id,
                task_id=sim.task_id,
                llm=llm,
                domain=domain,
                speech_complexity=speech_complexity,
                provider=provider,
            )
            all_events.extend(events)

    result_paths = {
        "raw_path": None,
        "analysis_path": None,
        "plot_path": None,
    }

    if not all_events:
        logger.warning("No voice quality events found")
        return result_paths

    # Convert to DataFrame
    raw_df = voice_quality_events_to_dataframe(all_events)
    logger.info(f"Collected {len(raw_df)} voice quality events")

    # Log event counts by category
    for category in ["response", "yield", "backchannel", "vocal_tic", "non_directed"]:
        cat_events = raw_df[raw_df["event_category"] == category]
        n_success = (~cat_events["is_error"]).sum()
        n_error = cat_events["is_error"].sum()
        logger.info(
            f"  {category}: {len(cat_events)} events ({n_success} success, {n_error} error)"
        )

    # Compute analysis
    analysis_df = compute_voice_quality_metrics(raw_df)

    # Save raw data
    raw_path = analysis_dir / "voice_quality_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    result_paths["raw_path"] = raw_path
    logger.info(f"Saved raw voice quality data to {raw_path}")

    # Save analysis
    analysis_path = analysis_dir / "voice_quality_analysis.csv"
    analysis_df.to_csv(analysis_path, index=False)
    result_paths["analysis_path"] = analysis_path
    logger.info(f"Saved voice quality analysis to {analysis_path}")

    # Create summary plot
    plot_path = analysis_dir / "voice_quality_summary.pdf"
    plot_voice_quality_summary(analysis_df, plot_path)
    result_paths["plot_path"] = plot_path

    return result_paths


# =============================================================================
# Speech Activity Timeline Visualization
# =============================================================================


def load_stereo_waveform(
    audio_path: Path,
    target_points: int = 2000,
):
    """
    Load a stereo WAV file and return downsampled waveforms for visualization.

    Uses tau2 audio utilities for proper audio loading and conversion.

    Args:
        audio_path: Path to the stereo WAV file (user=left, agent=right)
        target_points: Number of points to downsample to for visualization

    Returns:
        Tuple of (user_waveform, agent_waveform, duration_sec) or None if load fails
        Waveforms are normalized to [-1, 1] range
    """
    import numpy as np

    from tau2.voice.utils.audio_io import load_wav_file
    from tau2.voice.utils.audio_preprocessing import (
        audio_data_to_numpy,
        convert_to_pcm16,
    )

    try:
        # Load using tau2 utilities
        audio = load_wav_file(audio_path)

        if audio.format.channels != 2:
            logger.warning(
                f"Expected stereo audio, got {audio.format.channels} channels"
            )
            return None

        # Convert to PCM16 if needed (required by audio_data_to_numpy)
        if not audio.format.is_pcm16:
            audio = convert_to_pcm16(audio)

        # Convert to numpy array (interleaved stereo: L, R, L, R, ...)
        samples = audio_data_to_numpy(audio, dtype=np.int16)

        # Reshape to (n_frames, 2) - stereo interleaved format
        samples = samples.reshape(-1, 2)
        n_frames = len(samples)
        sample_rate = audio.format.sample_rate

        # Separate channels: user=left (ch0), agent=right (ch1)
        user_audio = samples[:, 0].astype(np.float32)
        agent_audio = samples[:, 1].astype(np.float32)

        # Normalize to [-1, 1]
        max_val = 32767.0  # int16 max
        user_audio /= max_val
        agent_audio /= max_val

        # Downsample for visualization using max-abs envelope in windows
        duration_sec = n_frames / sample_rate
        window_size = max(1, n_frames // target_points)

        def downsample_envelope(signal: np.ndarray, window: int) -> np.ndarray:
            """Get envelope (max abs) for each window."""
            n_windows = len(signal) // window
            if n_windows == 0:
                return np.abs(signal)
            signal = signal[: n_windows * window]
            reshaped = signal.reshape(n_windows, window)
            return np.max(np.abs(reshaped), axis=1)

        user_envelope = downsample_envelope(user_audio, window_size)
        agent_envelope = downsample_envelope(agent_audio, window_size)

        logger.debug(
            f"Loaded waveform: {n_frames} samples, {duration_sec:.1f}s, "
            f"user_max={user_envelope.max():.3f}, agent_max={agent_envelope.max():.3f}"
        )

        return user_envelope, agent_envelope, duration_sec

    except Exception as e:
        logger.warning(f"Failed to load audio from {audio_path}: {e}")
        return None


def plot_speech_timeline(
    user_segments: List[UserSpeechSegment],
    agent_segments: List[AgentSpeechSegment],
    total_duration_sec: Optional[float] = None,
    simulation_id: str = "",
    task_id: str = "",
    domain: str = "",
    agent_llm: str = "",
    background_noise: str = "",
    figsize: Tuple[float, float] = (14, 4),
    turn_transitions: Optional[List["TurnTransitionEvent"]] = None,
    interruption_events: Optional[List["InterruptionEvent"]] = None,
    out_of_turn_effects: Optional[List[AudioEffectSegment]] = None,
    frame_drops: Optional[List[FrameDropEvent]] = None,
    audio_path: Optional[Path] = None,
) -> "plt.Figure":
    """
    Create a timeline visualization showing when user and agent are speaking.

    Shows:
    - User speech segments as blue bars
    - Agent speech segments as red bars
    - Overlap regions highlighted in purple
    - Interruption markers (▼ user interrupts, ▲ agent interrupts)
    - Backchannel markers (○)
    - Audio effects during speech (below bars)
    - Out-of-turn effects in gap regions (when provided)
    - Optional waveform overlay from stereo audio file

    Optional diagnostic markers (when events are provided):
    - No-response markers (✗) - User had to retry, agent didn't respond
    - No-yield markers (⊘) - Interrupted party didn't yield
    - Backchannel-as-interruption markers (!) - Backchannel incorrectly treated as interruption
    - Incorrect response markers (∼✗ or …✗) - Agent responded to vocal tic or non-directed speech

    Args:
        user_segments: List of UserSpeechSegment objects
        agent_segments: List of AgentSpeechSegment objects
        total_duration_sec: Total duration in seconds (auto-calculated if None)
        simulation_id: Simulation ID for title
        task_id: Task ID for title
        domain: Domain name for title (e.g., "retail", "airline")
        agent_llm: Agent LLM model name for title
        figsize: Figure size as (width, height)
        turn_transitions: Optional list of TurnTransitionEvent for no-response markers
        interruption_events: Optional list of InterruptionEvent for no-yield markers
        out_of_turn_effects: Optional list of AudioEffectSegment for effects during gaps
        audio_path: Optional path to stereo WAV file for waveform overlay

    Returns:
        matplotlib Figure
    """
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    import numpy as np

    # Calculate total duration if not provided
    if total_duration_sec is None:
        max_user = max((s.end_time_sec for s in user_segments), default=0)
        max_agent = max((s.end_time_sec for s in agent_segments), default=0)
        total_duration_sec = max(max_user, max_agent) + 1

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor="#fafafa")
    ax.set_facecolor("#fafafa")

    # Color palette
    colors = {
        "user": "#2563eb",  # Blue
        "agent": "#dc2626",  # Red
        "overlap": "#7c3aed",  # Purple
        "grid": "#e5e7eb",  # Light gray
        "text": "#1f2937",  # Dark text
        "interruption": "#f59e0b",  # Orange
        "backchannel": "#10b981",  # Green
        "no_response": "#ef4444",  # Red for no-response
        "no_yield": "#991b1b",  # Dark red for no-yield
        "backchannel_issue": "#dc2626",  # Red for backchannel issues
        # Audio effects colors
        "burst_noise": "#8b5cf6",  # Violet for burst noise
        "vocal_tic": "#06b6d4",  # Cyan for vocal tics
        "non_directed": "#ec4899",  # Pink for non-directed speech
        "muffling": "#64748b",  # Slate for muffling
        "frame_drop": "#f97316",  # Orange for frame drops
    }

    # Y positions
    y_user = 0.6
    y_agent = 0.0
    bar_height = 0.35

    # === WAVEFORM OVERLAY ===
    # Plot waveform behind the speech bars if audio file is provided
    if audio_path is not None and Path(audio_path).exists():
        waveform_data = load_stereo_waveform(Path(audio_path))
        if waveform_data is not None:
            user_envelope, agent_envelope, audio_duration = waveform_data

            # Create time axis for waveform
            t_wave = np.linspace(0, audio_duration, len(user_envelope))

            # Clamp waveform to total_duration_sec (exclude end-of-conversation artifacts)
            clamp_mask = t_wave <= total_duration_sec
            t_wave = t_wave[clamp_mask]
            user_envelope = user_envelope[clamp_mask]
            agent_envelope = agent_envelope[clamp_mask]

            # Scale envelope to fit within the bar region
            wave_height = bar_height * 0.8  # Slightly smaller than bar

            # Plot waveforms as filled areas with contrasting colors
            # Use darker shades that contrast with the speech bar colors
            user_wave_color = "#1e3a5f"  # Dark navy (contrasts with blue bars)
            agent_wave_color = "#7f1d1d"  # Dark maroon (contrasts with red bars)

            user_wave_scaled = user_envelope * wave_height / 2
            ax.fill_between(
                t_wave,
                y_user - user_wave_scaled,
                y_user + user_wave_scaled,
                color=user_wave_color,
                alpha=0.6,
                zorder=2,  # Above bars
                linewidth=0,
            )

            agent_wave_scaled = agent_envelope * wave_height / 2
            ax.fill_between(
                t_wave,
                y_agent - agent_wave_scaled,
                y_agent + agent_wave_scaled,
                color=agent_wave_color,
                alpha=0.6,
                zorder=2,  # Above bars
                linewidth=0,
            )

            logger.debug(
                f"Waveform plotted: t_wave=[{t_wave[0]:.2f}, {t_wave[-1]:.2f}], "
                f"user y=[{(y_user - user_wave_scaled.max()):.3f}, {(y_user + user_wave_scaled.max()):.3f}], "
                f"agent y=[{(y_agent - agent_wave_scaled.max()):.3f}, {(y_agent + agent_wave_scaled.max()):.3f}]"
            )

    # Convert segments to broken_barh format: list of (start, duration)
    user_intervals = [(s.start_time_sec, s.duration_sec) for s in user_segments]
    agent_intervals = [(s.start_time_sec, s.duration_sec) for s in agent_segments]

    # Plot speaking regions (on top of waveform)
    if user_intervals:
        ax.broken_barh(
            user_intervals,
            (y_user - bar_height / 2, bar_height),
            facecolors=colors["user"],
            edgecolors="none",
            alpha=0.85,
            label="User",
        )

    if agent_intervals:
        ax.broken_barh(
            agent_intervals,
            (y_agent - bar_height / 2, bar_height),
            facecolors=colors["agent"],
            edgecolors="none",
            alpha=0.85,
            label="Agent",
        )

    # Find and highlight overlaps
    for user_seg in user_segments:
        for agent_seg in agent_segments:
            # Check for overlap
            overlap_start = max(user_seg.start_time_sec, agent_seg.start_time_sec)
            overlap_end = min(user_seg.end_time_sec, agent_seg.end_time_sec)
            if overlap_start < overlap_end:
                ax.axvspan(
                    overlap_start,
                    overlap_end,
                    alpha=0.25,
                    color=colors["overlap"],
                    zorder=0,
                )

    # Add event markers
    marker_y_offset = bar_height / 2 + 0.08
    marker_y_offset_high = bar_height / 2 + 0.18  # Higher offset for diagnostic markers

    for user_seg in user_segments:
        if user_seg.is_interruption:
            # User interruption marker (triangle pointing down)
            ax.plot(
                user_seg.start_time_sec,
                y_user + marker_y_offset,
                marker="v",
                color=colors["interruption"],
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=10,
            )
        elif user_seg.is_backchannel:
            # Backchannel marker (circle)
            ax.plot(
                user_seg.start_time_sec,
                y_user + marker_y_offset,
                marker="o",
                color=colors["backchannel"],
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=10,
            )

    for agent_seg in agent_segments:
        if agent_seg.other_speaking_at_start:
            # Agent started while user was speaking (agent interrupts - error)
            # Placed on error track since agent interrupting user is an error
            ax.plot(
                agent_seg.start_time_sec,
                y_agent - marker_y_offset_high,
                marker="^",
                color=colors["no_yield"],  # Red - agent error
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=15,
            )

    # === DIAGNOSTIC MARKERS ===

    # No-response markers (agent failed to respond - shown on agent track)
    if turn_transitions:
        for event in turn_transitions:
            if event.outcome == "no_response":
                # Mark at the end of the user segment that got no response
                # Placed on agent track since it's an agent failure
                ax.plot(
                    event.user_end_time_sec,
                    y_agent - marker_y_offset_high,
                    marker="X",
                    color=colors["no_response"],
                    markersize=9,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    zorder=15,
                )

    # No-yield and backchannel-as-interruption markers
    if interruption_events:
        for event in interruption_events:
            # No-yield: agent didn't stop when user interrupted
            if (
                event.event_type == "user_interrupts_agent"
                and not event.interrupted_yielded
            ):
                ax.plot(
                    event.interrupter_start_time_sec,
                    y_agent - marker_y_offset_high,
                    marker="$⊘$",  # "No" symbol
                    color=colors["no_yield"],
                    markersize=10,
                    zorder=15,
                )

            # Agent incorrectly yielded to backchannel (was speaking, stopped when shouldn't)
            # Backchannels only occur when agent is speaking, so we check if agent yielded
            if event.event_type == "backchannel" and event.interrupted_yielded:
                ax.plot(
                    event.interrupter_start_time_sec,
                    y_agent - marker_y_offset_high,
                    marker="$!$",
                    color=colors["backchannel_issue"],
                    markersize=10,
                    zorder=15,
                )

            # Agent incorrectly yielded to vocal tic (was speaking, stopped when shouldn't)
            if event.event_type == "vocal_tic" and event.interrupted_yielded:
                ax.plot(
                    event.interrupter_start_time_sec,
                    y_agent - marker_y_offset_high,
                    marker="$\u223c$",  # Wave symbol
                    color=colors["no_yield"],  # Red to indicate error
                    markersize=12,
                    zorder=15,
                )
                ax.plot(
                    event.interrupter_start_time_sec,
                    y_agent - marker_y_offset_high,
                    marker="x",
                    color=colors["no_yield"],
                    markersize=8,
                    markeredgewidth=2,
                    zorder=16,
                )

            # Agent incorrectly yielded to non-directed speech (was speaking, stopped when shouldn't)
            if event.event_type == "non_directed_speech" and event.interrupted_yielded:
                ax.plot(
                    event.interrupter_start_time_sec,
                    y_agent - marker_y_offset_high,
                    marker="$\u2026$",  # Ellipsis
                    color=colors["no_yield"],  # Red to indicate error
                    markersize=12,
                    zorder=15,
                )
                ax.plot(
                    event.interrupter_start_time_sec,
                    y_agent - marker_y_offset_high,
                    marker="x",
                    color=colors["no_yield"],
                    markersize=8,
                    markeredgewidth=2,
                    zorder=16,
                )

            # Agent incorrectly responded to vocal tic (wasn't speaking, started when shouldn't)
            if event.event_type == "agent_responds_to_vocal_tic":
                ax.plot(
                    event.interrupter_start_time_sec,
                    y_agent - marker_y_offset_high,
                    marker="$\u223c$",  # Wave symbol
                    color=colors["no_yield"],  # Red to indicate error
                    markersize=12,
                    zorder=15,
                )
                ax.plot(
                    event.interrupter_start_time_sec,
                    y_agent - marker_y_offset_high,
                    marker="x",
                    color=colors["no_yield"],
                    markersize=8,
                    markeredgewidth=2,
                    zorder=16,
                )

            # Agent incorrectly responded to non-directed speech (wasn't speaking, started when shouldn't)
            if event.event_type == "agent_responds_to_non_directed":
                ax.plot(
                    event.interrupter_start_time_sec,
                    y_agent - marker_y_offset_high,
                    marker="$\u2026$",  # Ellipsis
                    color=colors["no_yield"],  # Red to indicate error
                    markersize=12,
                    zorder=15,
                )
                ax.plot(
                    event.interrupter_start_time_sec,
                    y_agent - marker_y_offset_high,
                    marker="x",
                    color=colors["no_yield"],
                    markersize=8,
                    markeredgewidth=2,
                    zorder=16,
                )

    # === AUDIO EFFECTS MARKERS ===
    # These markers show audio effects applied during speech segments
    # Effects are now segments with start/end times - we place a marker at the start
    marker_y_effects_user = y_user - bar_height / 2 - 0.06  # Below user bar
    marker_y_effects_agent = y_agent + bar_height / 2 + 0.06  # Above agent bar

    def _plot_effect_marker(ax, effect, y_pos, colors):
        """Plot a marker for an audio effect segment at its start time."""
        if effect.effect_type == "burst_noise":
            ax.plot(
                effect.start_time_sec,
                y_pos,
                marker="$\u26a1$",  # Lightning bolt
                color=colors["burst_noise"],
                markersize=8,
                zorder=12,
            )
        elif effect.effect_type == "vocal_tic":
            ax.plot(
                effect.start_time_sec,
                y_pos,
                marker="$\u223c$",  # Tilde/wave
                color=colors["vocal_tic"],
                markersize=7,
                zorder=12,
            )
        elif effect.effect_type == "non_directed_speech":
            ax.plot(
                effect.start_time_sec,
                y_pos,
                marker="$\u2026$",  # Ellipsis
                color=colors["non_directed"],
                markersize=8,
                zorder=12,
            )
        elif effect.effect_type == "muffling":
            ax.plot(
                effect.start_time_sec,
                y_pos,
                marker="$\u2592$",  # Medium shade block
                color=colors["muffling"],
                markersize=7,
                zorder=12,
            )

    # Plot audio effects for user segments
    for user_seg in user_segments:
        for effect in user_seg.audio_effects:
            _plot_effect_marker(ax, effect, marker_y_effects_user, colors)

    # Plot audio effects for agent segments (less common but possible)
    for agent_seg in agent_segments:
        for effect in agent_seg.audio_effects:
            _plot_effect_marker(ax, effect, marker_y_effects_agent, colors)

    # === OUT-OF-TURN EFFECTS ===
    # Effects that occur in gaps between speech segments - all placed at user track level
    if out_of_turn_effects:
        for effect in out_of_turn_effects:
            _plot_effect_marker(ax, effect, marker_y_effects_user, colors)

    # === FRAME DROP MARKERS ===
    # Frame drops simulate network packet loss - shown on user track
    if frame_drops:
        for fd in frame_drops:
            # Use a different marker style - vertical bar/pipe to indicate dropped frames
            ax.plot(
                fd.time_sec,
                marker_y_effects_user,
                marker="|",
                color=colors["frame_drop"],
                markersize=10,
                markeredgewidth=2,
                zorder=12,
            )

    # Grid lines every 5 seconds
    ax.set_xlim(0, total_duration_sec)
    ax.set_ylim(-0.4, 1.0)

    for t in np.arange(0, total_duration_sec, 5):
        ax.axvline(t, color=colors["grid"], linewidth=0.5, zorder=0)

    # Styling
    ax.set_yticks([y_agent, y_user])
    ax.set_yticklabels(
        ["Agent", "User"], fontsize=11, fontweight="medium", color=colors["text"]
    )
    ax.set_xlabel("Time (seconds)", fontsize=10, color=colors["text"], labelpad=8)

    # Add background noise annotation below "User" label
    if background_noise:
        # Use display name if available, otherwise format the filename
        from tau2.user_simulation_voice_presets import BACKGROUND_NOISE_DISPLAY_NAMES

        noise_display = BACKGROUND_NOISE_DISPLAY_NAMES.get(
            background_noise, background_noise.replace("_", " ").replace(".wav", "")
        )
        ax.text(
            -0.01,  # Just to the left of the plot area
            y_user - 0.08,  # Below the User label
            f"({noise_display})",
            transform=ax.get_yaxis_transform(),
            fontsize=7,
            color=colors["text"],
            alpha=0.6,
            ha="right",
            va="top",
        )

    # Reference lines
    ax.axhline(y_user, color=colors["grid"], linewidth=0.8, alpha=0.4)
    ax.axhline(y_agent, color=colors["grid"], linewidth=0.8, alpha=0.4)

    # Title
    title = "Speech Activity Timeline"
    title_parts = []
    if domain:
        title_parts.append(domain.capitalize())
    if agent_llm:
        # Shorten model name for display (e.g., "gemini-2.0-flash-exp" -> "Gemini 2.0 Flash")
        llm_display = agent_llm.replace("-exp", "").replace("-", " ").title()
        title_parts.append(llm_display)
    if title_parts:
        title += f" — {', '.join(title_parts)}"
    ax.set_title(title, fontsize=13, fontweight="bold", color=colors["text"], pad=10)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend - positioned below the plot to avoid covering the timeline
    # Organized into: Observations (what happened) and Evaluations (agent behavior issues)

    # === OBSERVATIONS ===
    observation_elements = [
        mlines.Line2D(
            [],
            [],
            color=colors["user"],
            marker="s",
            linestyle="None",
            markersize=10,
            label="User",
        ),
        mlines.Line2D(
            [],
            [],
            color=colors["agent"],
            marker="s",
            linestyle="None",
            markersize=10,
            label="Agent",
        ),
        mlines.Line2D(
            [],
            [],
            color=colors["interruption"],
            marker="v",
            linestyle="None",
            markersize=8,
            label="User Int.",
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
    ]

    # === EVALUATIONS (agent behavior issues) ===
    # Always show all evaluation items in legend
    evaluation_elements = [
        mlines.Line2D(
            [],
            [],
            color=colors["no_yield"],  # Red - agent error
            marker="^",
            linestyle="None",
            markersize=8,
            label="Agent Int.",
        ),
        mlines.Line2D(
            [],
            [],
            color=colors["no_response"],
            marker="X",
            linestyle="None",
            markersize=8,
            label="No Response",
        ),
        mlines.Line2D(
            [],
            [],
            color=colors["no_yield"],
            marker="$⊘$",
            linestyle="None",
            markersize=8,
            label="No Yield",
        ),
        mlines.Line2D(
            [],
            [],
            color=colors["backchannel_issue"],
            marker="$!$",
            linestyle="None",
            markersize=8,
            label="BC Issue",
        ),
        mlines.Line2D(
            [],
            [],
            color=colors["no_yield"],
            marker="$\u223c\u2717$",  # Wave + X
            linestyle="None",
            markersize=10,
            label="Voc. Tic Error",
        ),
        mlines.Line2D(
            [],
            [],
            color=colors["no_yield"],
            marker="$\u2026\u2717$",  # Ellipsis + X
            linestyle="None",
            markersize=10,
            label="Non-Agent Dir. Error",
        ),
    ]

    # Always add audio effects to observations
    observation_elements.extend(
        [
            mlines.Line2D(
                [],
                [],
                color=colors["burst_noise"],
                marker="$\u26a1$",
                linestyle="None",
                markersize=8,
                label="Burst",
            ),
            mlines.Line2D(
                [],
                [],
                color=colors["vocal_tic"],
                marker="$\u223c$",
                linestyle="None",
                markersize=7,
                label="Vocal Tic",
            ),
            mlines.Line2D(
                [],
                [],
                color=colors["non_directed"],
                marker="$\u2026$",
                linestyle="None",
                markersize=8,
                label="Non-Agent Dir.",
            ),
            mlines.Line2D(
                [],
                [],
                color=colors["muffling"],
                marker="$\u2592$",
                linestyle="None",
                markersize=7,
                label="Muffling",
            ),
            mlines.Line2D(
                [],
                [],
                color=colors["frame_drop"],
                marker="|",
                linestyle="None",
                markersize=8,
                markeredgewidth=2,
                label="Frame Drop",
            ),
        ]
    )

    # Combine observations and evaluations
    legend_elements = observation_elements + evaluation_elements

    # Adjust ncol to balance rows - prefer fewer columns to avoid sparse rows
    n_items = len(legend_elements)
    if n_items <= 5:
        ncol = n_items  # Single row
    elif n_items <= 10:
        ncol = (n_items + 1) // 2  # Two balanced rows
    else:
        ncol = (n_items + 2) // 3  # Three balanced rows

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        framealpha=0.95,
        fontsize=8,
        ncol=ncol,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at bottom for legend
    return fig


def save_speech_timeline(
    user_segments: List[UserSpeechSegment],
    agent_segments: List[AgentSpeechSegment],
    output_path: Path,
    total_duration_sec: Optional[float] = None,
    simulation_id: str = "",
    task_id: str = "",
    domain: str = "",
    agent_llm: str = "",
    background_noise: str = "",
    turn_transitions: Optional[List["TurnTransitionEvent"]] = None,
    interruption_events: Optional[List["InterruptionEvent"]] = None,
    out_of_turn_effects: Optional[List[AudioEffectSegment]] = None,
    frame_drops: Optional[List[FrameDropEvent]] = None,
    audio_path: Optional[Path] = None,
) -> Path:
    """
    Save a speech timeline visualization to a file.

    Args:
        user_segments: List of UserSpeechSegment objects
        agent_segments: List of AgentSpeechSegment objects
        output_path: Path to save the figure
        total_duration_sec: Total duration in seconds
        simulation_id: Simulation ID for title
        task_id: Task ID for title
        domain: Domain name for title (e.g., "retail", "airline")
        agent_llm: Agent LLM model name for title
        background_noise: Background noise file name for annotation
        turn_transitions: Optional list of TurnTransitionEvent for no-response markers
        interruption_events: Optional list of InterruptionEvent for no-yield markers
        out_of_turn_effects: Optional list of AudioEffectSegment for effects during gaps
        frame_drops: Optional list of FrameDropEvent for frame drop markers
        audio_path: Optional path to stereo WAV file for waveform overlay

    Returns:
        Path to the saved file
    """
    import matplotlib.pyplot as plt

    fig = plot_speech_timeline(
        user_segments,
        agent_segments,
        total_duration_sec,
        simulation_id,
        task_id,
        domain=domain,
        agent_llm=agent_llm,
        background_noise=background_noise,
        turn_transitions=turn_transitions,
        interruption_events=interruption_events,
        out_of_turn_effects=out_of_turn_effects,
        frame_drops=frame_drops,
        audio_path=audio_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved speech timeline to {output_path}")
    return output_path


def run_interruption_analysis(
    all_results: List[Tuple[dict, "Results"]],
    output_dir: Path,
    tick_duration_sec: float = 0.2,
    no_yield_window_sec: float = 2.0,
    backchannel_yield_window_sec: float = 1.0,
    vocal_tic_yield_window_sec: float = 1.0,
    non_directed_yield_window_sec: float = 1.0,
    vocal_tic_response_window_sec: float = 2.0,
    non_directed_response_window_sec: float = 2.0,
) -> dict:
    """
    Run the full interruption handling analysis pipeline.

    Args:
        all_results: List of (params, Results) tuples from load_simulation_results()
        output_dir: Base output directory
        tick_duration_sec: Duration of each tick in seconds
        no_yield_window_sec: Time window for user interruption yield detection (default: 2.0)
        backchannel_yield_window_sec: Time window for backchannel yield detection (default: 1.0)
        vocal_tic_yield_window_sec: Time window for vocal tic yield detection (default: 1.0)
        non_directed_yield_window_sec: Time window for non-directed yield detection (default: 1.0)
        vocal_tic_response_window_sec: Time window for vocal tic response detection (default: 2.0)
        non_directed_response_window_sec: Time window for non-directed response detection (default: 2.0)

    Returns:
        Dict with paths to output files
    """
    import pandas as pd

    # Create output directory
    analysis_dir = output_dir / "interruption_handling"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Collect all interruption events
    all_dfs = []
    for params, results in all_results:
        df = extract_interruptions_from_results(
            params,
            results,
            tick_duration_sec,
            no_yield_window_sec,
            backchannel_yield_window_sec,
            vocal_tic_yield_window_sec,
            non_directed_yield_window_sec,
            vocal_tic_response_window_sec,
            non_directed_response_window_sec,
        )
        if not df.empty:
            all_dfs.append(df)

    result_paths = {
        "raw_path": None,
        "analysis_path": None,
        "plot_paths": [],
    }

    if not all_dfs:
        logger.warning("No interruption data found")
        return result_paths

    # Combine all data
    raw_df = pd.concat(all_dfs, ignore_index=True)
    user_int = (raw_df["event_type"] == "user_interrupts_agent").sum()
    backchannels = (raw_df["event_type"] == "backchannel").sum()
    vocal_tics = (raw_df["event_type"] == "vocal_tic").sum()
    non_directed = (raw_df["event_type"] == "non_directed_speech").sum()
    agent_int = (raw_df["event_type"] == "agent_interrupts_user").sum()
    agent_resp_vocal = (raw_df["event_type"] == "agent_responds_to_vocal_tic").sum()
    agent_resp_non_dir = (
        raw_df["event_type"] == "agent_responds_to_non_directed"
    ).sum()
    vocal_tic_silent_ok = (raw_df["event_type"] == "vocal_tic_silent_correct").sum()
    non_dir_silent_ok = (raw_df["event_type"] == "non_directed_silent_correct").sum()
    logger.info(
        f"Collected {len(raw_df)} interruption events "
        f"({user_int} user_int, {backchannels} backchannel, "
        f"{vocal_tics} vocal_tic, {non_directed} non_directed, {agent_int} agent_int, "
        f"{agent_resp_vocal} resp_vocal, {agent_resp_non_dir} resp_non_dir, "
        f"{vocal_tic_silent_ok} tic_silent_ok, {non_dir_silent_ok} non_dir_silent_ok)"
    )

    # Compute analysis
    analysis_df = analyze_interruptions(raw_df)

    # Save outputs
    result_paths["raw_path"] = save_interruptions_raw(raw_df, analysis_dir)
    result_paths["analysis_path"] = save_interruptions_analysis(
        analysis_df, analysis_dir
    )
    result_paths["plot_paths"] = plot_interruptions(
        raw_df, analysis_df, analysis_dir, no_yield_window_sec
    )

    return result_paths


# =============================================================================
# CLI for Testing
# =============================================================================


def get_tick_duration(sim: "SimulationRun", default: float = 0.2) -> float:
    """Get tick duration from simulation metadata, falling back to default."""
    if sim.speech_environment and hasattr(
        sim.speech_environment, "tick_duration_seconds"
    ):
        return sim.speech_environment.tick_duration_seconds
    return default


# =============================================================================
# Main Analysis Orchestration (for collections of results.json files)
# =============================================================================


def analyze_voice_results(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    filter_domains: Optional[List[str]] = None,
    tick_duration_sec: float = 0.2,
    no_yield_window_sec: float = 2.0,
    backchannel_yield_window_sec: float = 1.0,
    vocal_tic_yield_window_sec: float = 1.0,
    non_directed_yield_window_sec: float = 1.0,
    vocal_tic_response_window_sec: float = 2.0,
    non_directed_response_window_sec: float = 2.0,
    results: Optional[list] = None,
) -> dict:
    """
    Main analysis function for voice/conversation dynamics in tau_voice experiments.

    Recursively searches for results.json files in data_dir and runs all voice
    analysis pipelines, generating raw tables, analysis tables, and plots.

    This follows the same architectural pattern as performance_analysis.py:
    - Each analysis outputs to a dedicated subdirectory
    - Consistent pattern: raw.csv, analysis.csv, *.pdf

    Args:
        data_dir: Directory containing simulation folders (searched recursively)
        output_dir: Directory for output figures (default: data_dir/analysis/voice_analysis)
        filter_domains: Optional list of domains to include
        tick_duration_sec: Duration of each tick in seconds (default: 0.2)
        no_yield_window_sec: Time window for user interruption yield detection (default: 2.0)
        backchannel_yield_window_sec: Time window for backchannel yield detection (default: 1.0)
        vocal_tic_yield_window_sec: Time window for vocal tic yield detection (default: 1.0)
        non_directed_yield_window_sec: Time window for non-directed yield detection (default: 1.0)
        vocal_tic_response_window_sec: Time window for vocal tic response detection (default: 2.0)
        non_directed_response_window_sec: Time window for non-directed response detection (default: 2.0)
        results: Optional pre-loaded results to avoid reloading data

    Returns:
        Dict with paths to all generated output files
    """
    # Import load_simulation_results from data_loader
    from experiments.tau_voice.exp.data_loader import load_simulation_results

    logger.info(f"Analyzing voice/conversation dynamics in {data_dir}...")

    # Load results if not provided
    if results is None:
        all_results = load_simulation_results(data_dir, filter_domains)
    else:
        all_results = results

    if not all_results:
        logger.warning("No results found. Exiting.")
        return {}

    logger.info(f"Analyzing {len(all_results)} simulation results.")

    # Set up output directory
    if output_dir is None:
        output_dir = data_dir / "analysis/voice_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving voice analysis to: {output_dir}")

    # Track all output paths
    all_output_paths = {}

    # ==========================================================================
    # Unified Voice Quality Analysis (PRIMARY - matches speech timeline exactly)
    # ==========================================================================
    try:
        logger.info("Running unified voice quality analysis...")
        result_paths = run_voice_quality_analysis(
            all_results,
            output_dir,
            tick_duration_sec,
            no_yield_window_sec,
            backchannel_yield_window_sec,
            vocal_tic_yield_window_sec,
            non_directed_yield_window_sec,
            vocal_tic_response_window_sec,
            non_directed_response_window_sec,
        )
        all_output_paths["voice_quality"] = result_paths
        logger.info("  ✓ Voice quality analysis complete")
    except Exception as e:
        logger.error(f"Failed to generate voice quality analysis: {e}")

    # ==========================================================================
    # Legacy: Response Latency / Turn Transitions Analysis
    # (kept for backwards compatibility, may be removed in future)
    # ==========================================================================
    try:
        logger.info("Running response latency analysis...")
        result_paths = run_response_latency_analysis(
            all_results, output_dir, tick_duration_sec
        )
        all_output_paths["response_latency"] = result_paths
        logger.info("  ✓ Response latency analysis complete")
    except Exception as e:
        logger.error(f"Failed to generate response latency analysis: {e}")

    # ==========================================================================
    # Interruption Handling Analysis
    # ==========================================================================
    try:
        logger.info("Running interruption handling analysis...")
        result_paths = run_interruption_analysis(
            all_results,
            output_dir,
            tick_duration_sec,
            no_yield_window_sec,
            backchannel_yield_window_sec,
            vocal_tic_yield_window_sec,
            non_directed_yield_window_sec,
            vocal_tic_response_window_sec,
            non_directed_response_window_sec,
        )
        all_output_paths["interruption_handling"] = result_paths
        logger.info("  ✓ Interruption handling analysis complete")
    except Exception as e:
        logger.error(f"Failed to generate interruption analysis: {e}")

    # ==========================================================================
    # Speech Segments Analysis (raw segment data)
    # ==========================================================================
    try:
        logger.info("Running speech segments analysis...")
        result_paths = run_speech_segments_analysis(
            all_results, output_dir, tick_duration_sec
        )
        all_output_paths["speech_segments"] = result_paths
        logger.info("  ✓ Speech segments analysis complete")
    except Exception as e:
        logger.error(f"Failed to generate speech segments analysis: {e}")

    # ==========================================================================
    # VAD (Voice Activity Detection) Analysis
    # ==========================================================================
    try:
        logger.info("Running VAD analysis...")
        result_paths = run_vad_analysis(all_results, output_dir, tick_duration_sec)
        all_output_paths["vad_analysis"] = result_paths
        logger.info("  ✓ VAD analysis complete")
    except Exception as e:
        logger.error(f"Failed to generate VAD analysis: {e}")

    # Summary
    # Note: Paper outputs are now generated by paper_outputs.py via run_all_analysis.py

    logger.info("=" * 70)
    logger.info("VOICE ANALYSIS COMPLETE")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)

    return all_output_paths


def run_speech_segments_analysis(
    all_results: List[Tuple[dict, "Results"]],
    output_dir: Path,
    tick_duration_sec: float = 0.2,
) -> dict:
    """
    Run speech segment extraction and save raw data.

    Args:
        all_results: List of (params, Results) tuples
        output_dir: Base output directory
        tick_duration_sec: Duration of each tick in seconds

    Returns:
        Dict with paths to output files
    """
    import pandas as pd

    # Create output directory
    analysis_dir = output_dir / "speech_segments"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Collect all segment data
    all_user_dfs = []
    all_agent_dfs = []

    for params, results in all_results:
        user_df, agent_df = extract_segments_from_results(
            params, results, tick_duration_sec
        )
        if not user_df.empty:
            all_user_dfs.append(user_df)
        if not agent_df.empty:
            all_agent_dfs.append(agent_df)

    result_paths = {
        "user_segments_path": None,
        "agent_segments_path": None,
        "analysis_path": None,
    }

    # Combine and save user segments
    if all_user_dfs:
        user_segments_df = pd.concat(all_user_dfs, ignore_index=True)
        user_path = analysis_dir / "user_segments_raw.csv"
        user_segments_df.to_csv(user_path, index=False)
        result_paths["user_segments_path"] = user_path
        logger.info(f"Saved {len(user_segments_df)} user segments to {user_path}")

    # Combine and save agent segments
    if all_agent_dfs:
        agent_segments_df = pd.concat(all_agent_dfs, ignore_index=True)
        agent_path = analysis_dir / "agent_segments_raw.csv"
        agent_segments_df.to_csv(agent_path, index=False)
        result_paths["agent_segments_path"] = agent_path
        logger.info(f"Saved {len(agent_segments_df)} agent segments to {agent_path}")

    # Generate summary analysis
    if all_user_dfs or all_agent_dfs:
        analysis_df = _compute_speech_segments_analysis(
            (
                pd.concat(all_user_dfs, ignore_index=True)
                if all_user_dfs
                else pd.DataFrame()
            ),
            (
                pd.concat(all_agent_dfs, ignore_index=True)
                if all_agent_dfs
                else pd.DataFrame()
            ),
        )
        if not analysis_df.empty:
            analysis_path = analysis_dir / f"{analysis_dir.name}_analysis.csv"
            analysis_df.to_csv(analysis_path, index=False)
            result_paths["analysis_path"] = analysis_path
            logger.info(f"Saved speech segments analysis to {analysis_path}")

    return result_paths


def _compute_speech_segments_analysis(
    user_df: "pd.DataFrame",
    agent_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """
    Compute summary statistics for speech segments.

    Aggregates by llm, domain, speech_complexity.
    """
    import numpy as np
    import pandas as pd

    if user_df.empty and agent_df.empty:
        return pd.DataFrame()

    summary_rows = []

    # Get grouping columns from whichever dataframe is non-empty
    ref_df = user_df if not user_df.empty else agent_df
    group_cols = ["llm", "domain", "speech_complexity", "provider"]

    for group_vals, _ in ref_df.groupby(group_cols):
        llm, domain, complexity, provider = group_vals

        # Filter to this group
        user_group = (
            user_df[
                (user_df["llm"] == llm)
                & (user_df["domain"] == domain)
                & (user_df["speech_complexity"] == complexity)
            ]
            if not user_df.empty
            else pd.DataFrame()
        )

        agent_group = (
            agent_df[
                (agent_df["llm"] == llm)
                & (agent_df["domain"] == domain)
                & (agent_df["speech_complexity"] == complexity)
            ]
            if not agent_df.empty
            else pd.DataFrame()
        )

        row = {
            "llm": llm,
            "domain": domain,
            "speech_complexity": complexity,
            "provider": provider,
            # User segment stats
            "user_segment_count": len(user_group),
            "user_total_duration_sec": (
                user_group["duration_sec"].sum() if len(user_group) > 0 else 0
            ),
            "user_mean_duration_sec": (
                user_group["duration_sec"].mean() if len(user_group) > 0 else np.nan
            ),
            "user_interruption_count": (
                user_group["is_interruption"].sum() if len(user_group) > 0 else 0
            ),
            "user_backchannel_count": (
                user_group["is_backchannel"].sum() if len(user_group) > 0 else 0
            ),
            # Agent segment stats
            "agent_segment_count": len(agent_group),
            "agent_total_duration_sec": (
                agent_group["duration_sec"].sum() if len(agent_group) > 0 else 0
            ),
            "agent_mean_duration_sec": (
                agent_group["duration_sec"].mean() if len(agent_group) > 0 else np.nan
            ),
            "agent_interrupted_count": (
                agent_group["was_interrupted"].sum() if len(agent_group) > 0 else 0
            ),
        }
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


# =============================================================================
# CLI
# =============================================================================


def get_cli_parser() -> "argparse.ArgumentParser":
    """Create the argument parser for voice analysis CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Voice/Conversation Dynamics Analysis for tau_voice experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all results in a directory (batch mode)
  python -m experiments.tau_voice.exp.voice_analysis --data-dir data/exp/my_results

  # Generate a single timeline from a results.json file
  python -m experiments.tau_voice.exp.voice_analysis results.json --timeline timeline.pdf

  # Generate timeline with diagnostic markers
  python -m experiments.tau_voice.exp.voice_analysis results.json --timeline timeline.pdf --diagnostics
""",
    )

    # Batch mode arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing simulation folders (searched recursively for results.json files)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for output files (default: data_dir/analysis/voice_analysis)",
    )
    parser.add_argument(
        "--filter-domains",
        type=str,
        nargs="+",
        help="Only include specific domains (e.g., retail airline)",
    )

    # Single file mode arguments
    parser.add_argument(
        "results_path",
        type=str,
        nargs="?",
        help="Path to a single results.json or simulation.json file (for timeline mode)",
    )
    parser.add_argument(
        "--timeline",
        type=str,
        metavar="OUTPUT_PATH",
        help="Generate speech timeline and save to specified path (e.g., timeline.pdf)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Include diagnostic markers (no-response, no-yield, backchannel issues) on timeline",
    )
    parser.add_argument(
        "--audio",
        type=str,
        metavar="AUDIO_PATH",
        help="Path to stereo WAV file (both.wav) for waveform overlay on timeline",
    )
    parser.add_argument(
        "--sim-index",
        type=int,
        default=0,
        help="Simulation index to use when processing results.json (default: 0)",
    )

    # Common arguments
    parser.add_argument(
        "--tick-duration",
        type=float,
        default=0.2,
        help="Tick duration in seconds (default: 0.2)",
    )
    # Individual threshold arguments for each error type
    parser.add_argument(
        "--no-yield-window",
        type=float,
        default=2.0,
        help="Time window (seconds) to detect yield after user interruption (default: 2.0)",
    )
    parser.add_argument(
        "--backchannel-yield-window",
        type=float,
        default=1.0,
        help="Time window (seconds) to detect yield after backchannel (default: 1.0)",
    )
    parser.add_argument(
        "--vocal-tic-yield-window",
        type=float,
        default=1.0,
        help="Time window (seconds) to detect yield after vocal tic (default: 1.0)",
    )
    parser.add_argument(
        "--non-directed-yield-window",
        type=float,
        default=1.0,
        help="Time window (seconds) to detect yield after non-directed speech (default: 1.0)",
    )
    parser.add_argument(
        "--vocal-tic-response-window",
        type=float,
        default=2.0,
        help="Time window (seconds) to detect agent response to vocal tic (default: 2.0)",
    )
    parser.add_argument(
        "--non-directed-response-window",
        type=float,
        default=2.0,
        help="Time window (seconds) to detect agent response to non-directed speech (default: 2.0)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directory contents before running analysis.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate plots from existing CSV files without reloading data.",
    )

    return parser


# =============================================================================
# Plots-Only Mode (regenerate from existing CSVs)
# =============================================================================


def regenerate_plots_from_csv(output_dir: Path) -> None:
    """
    Regenerate all plots from existing CSV files without reloading data.

    This is useful for iterating on plot styling without recomputing data.

    Args:
        output_dir: Directory containing the analysis subdirectories with CSV files.
    """
    import pandas as pd
    from loguru import logger

    logger.info(f"Regenerating plots from existing CSVs in {output_dir}...")

    # Import plotting functions
    from experiments.tau_voice.exp.plot_style import DOMAINS, SPEECH_COMPLEXITIES

    # =========================================================================
    # VAD Analysis
    # =========================================================================
    try:
        vad_dir = output_dir / "vad_analysis"
        analysis_csv = vad_dir / f"{vad_dir.name}_analysis.csv"
        if analysis_csv.exists():
            df_analysis = pd.read_csv(analysis_csv)
            # Get unique values from data
            llms = sorted(df_analysis["llm"].unique().tolist())
            domains = [d for d in DOMAINS if d in df_analysis["domain"].unique()]
            complexities = [
                c
                for c in SPEECH_COMPLEXITIES
                if c in df_analysis["speech_complexity"].unique()
            ]

            plot_vad_analysis(
                analysis_df=df_analysis,
                output_dir=vad_dir,
                llms=llms,
                domains=domains,
                complexities=complexities,
            )
            logger.info("Regenerated VAD analysis plots")
        else:
            logger.warning(f"Skipping VAD analysis: {analysis_csv} not found")
    except Exception as e:
        logger.error(f"Failed to regenerate VAD plots: {e}")

    # =========================================================================
    # Response Latency Analysis
    # =========================================================================
    try:
        latency_dir = output_dir / "response_latency"
        analysis_csv = latency_dir / f"{latency_dir.name}_analysis.csv"
        if analysis_csv.exists():
            df_analysis = pd.read_csv(analysis_csv)
            llms = sorted(df_analysis["llm"].unique().tolist())
            domains = [d for d in DOMAINS if d in df_analysis["domain"].unique()]
            complexities = [
                c
                for c in SPEECH_COMPLEXITIES
                if c in df_analysis["speech_complexity"].unique()
            ]

            plot_turn_transitions(
                analysis_df=df_analysis,
                output_dir=latency_dir,
                llms=llms,
                domains=domains,
                complexities=complexities,
            )
            logger.info("Regenerated response latency plots")
        else:
            logger.warning(f"Skipping response latency: {analysis_csv} not found")
    except Exception as e:
        logger.error(f"Failed to regenerate response latency plots: {e}")

    # =========================================================================
    # Interruption Analysis
    # =========================================================================
    try:
        interruption_dir = output_dir / "interruption_handling"
        analysis_csv = interruption_dir / f"{interruption_dir.name}_analysis.csv"
        if analysis_csv.exists():
            df_analysis = pd.read_csv(analysis_csv)
            llms = sorted(df_analysis["llm"].unique().tolist())
            domains = [d for d in DOMAINS if d in df_analysis["domain"].unique()]
            complexities = [
                c
                for c in SPEECH_COMPLEXITIES
                if c in df_analysis["speech_complexity"].unique()
            ]

            plot_interruptions(
                analysis_df=df_analysis,
                output_dir=interruption_dir,
                llms=llms,
                domains=domains,
                complexities=complexities,
            )
            logger.info("Regenerated interruption analysis plots")
        else:
            logger.warning(f"Skipping interruption analysis: {analysis_csv} not found")
    except Exception as e:
        logger.error(f"Failed to regenerate interruption plots: {e}")

    # Note: Speech timeline plots require full simulation data and cannot be
    # regenerated from CSV alone.
    logger.info(
        "Skipping speech timeline plots (requires full simulation data, "
        "not available in plots-only mode)"
    )

    # =========================================================================
    # Paper Outputs (from CSVs)
    # Note: Paper outputs are now generated by paper_outputs.py via run_all_analysis.py

    logger.info("Plot regeneration complete!")


def _regenerate_voice_paper_outputs_from_csv(output_dir: Path) -> None:
    """
    Regenerate voice paper outputs from existing CSV files.
    """
    import pandas as pd

    # Paper outputs go to shared paper directory at analysis level
    paper_dir = output_dir.parent / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Load response latency analysis
    latency_csv = output_dir / "response_latency" / "response_latency_analysis.csv"
    interruption_csv = (
        output_dir / "interruption_handling" / "interruption_handling_analysis.csv"
    )

    if not latency_csv.exists() or not interruption_csv.exists():
        logger.warning("Required CSVs not found for paper outputs.")
        return

    df_latency = pd.read_csv(latency_csv)
    df_interruption = pd.read_csv(interruption_csv)

    # Generate voice quality table from CSVs
    _generate_voice_quality_table_from_csv(paper_dir, df_latency, df_interruption)


def _generate_voice_quality_table_from_csv(
    output_dir: Path,
    df_latency: "pd.DataFrame",
    df_interruption: "pd.DataFrame",
) -> None:
    """
    Generate LaTeX table for voice quality metrics from CSV data.
    """
    import numpy as np
    import pandas as pd

    from experiments.tau_voice.exp.plot_style import (
        DOMAINS,
        get_model_sort_key,
        get_short_llm_name,
    )

    # Filter to regular complexity only
    df_latency_reg = df_latency[df_latency["speech_complexity"] == "regular"].copy()
    df_int_reg = df_interruption[
        df_interruption["speech_complexity"] == "regular"
    ].copy()

    if df_latency_reg.empty and df_int_reg.empty:
        logger.warning("No regular complexity data for voice quality table.")
        return

    # Merge the dataframes
    merge_cols = ["llm", "domain", "speech_complexity", "provider"]
    if not df_latency_reg.empty and not df_int_reg.empty:
        df = pd.merge(df_latency_reg, df_int_reg, on=merge_cols, how="outer")
    elif not df_latency_reg.empty:
        df = df_latency_reg
    else:
        df = df_int_reg

    # Get domains
    domains = [d for d in DOMAINS if d in df["domain"].unique()]

    # Generate LaTeX table
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(
        r"\caption{Voice interaction quality metrics (Regular condition). \textbf{Bold} indicates best per domain. $\uparrow$ = higher is better, $\downarrow$ = lower is better.}"
    )
    lines.append(r"\label{tab:voice-quality}")
    lines.append(r"\centering")
    lines.append(r"\begin{small}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Domain} & \textbf{Model} & \makecell{\textbf{Resp.}\\\textbf{Rate}$\uparrow$} & \makecell{\textbf{Resp.}\\\textbf{Latency (s)}$\downarrow$} & \makecell{\textbf{Yield}\\\textbf{Rate}$\uparrow$} & \makecell{\textbf{Yield}\\\textbf{Time (s)}$\downarrow$} & \makecell{\textbf{Backchannel}\\\textbf{Correct}$\uparrow$} \\"
    )
    lines.append(r"\midrule")

    for domain in domains:
        domain_data = df[df["domain"] == domain]

        if domain_data.empty:
            continue

        # Find best values for each metric
        best_resp_rate = (
            domain_data["response_rate"].max()
            if "response_rate" in domain_data
            else None
        )
        best_resp_latency = (
            domain_data["latency_mean"].min() if "latency_mean" in domain_data else None
        )
        best_yield_rate = (
            domain_data["user_interrupts_yield_rate"].max()
            if "user_interrupts_yield_rate" in domain_data
            else None
        )
        best_yield_time = (
            domain_data["user_interrupts_yield_time_mean"].min()
            if "user_interrupts_yield_time_mean" in domain_data
            else None
        )
        best_backchannel = (
            domain_data["backchannel_correct_rate"].max()
            if "backchannel_correct_rate" in domain_data
            else None
        )

        domain_llms = sorted(
            domain_data["llm"].unique().tolist(), key=get_model_sort_key
        )

        for i, llm in enumerate(domain_llms):
            row = domain_data[domain_data["llm"] == llm].iloc[0]
            model_name = get_short_llm_name(llm, max_len=25)

            # Domain label (only on first row)
            if i == 0:
                domain_label = (
                    rf"\multirow{{{len(domain_llms)}}}{{*}}{{{domain.capitalize()}}}"
                )
            else:
                domain_label = ""

            # Format values with bold for best
            def fmt_pct(val, best_val, higher_better=True):
                if pd.isna(val):
                    return "--"
                is_best = (val == best_val) if higher_better else (val == best_val)
                pct = f"{int(val * 100)}\\%"
                return rf"\textbf{{{pct}}}" if is_best else pct

            def fmt_sec(val, best_val):
                if pd.isna(val):
                    return "--"
                is_best = val == best_val
                sec = f"{val:.2f}"
                return rf"\textbf{{{sec}}}" if is_best else sec

            resp_rate = row.get("response_rate", np.nan)
            resp_latency = row.get("latency_mean", np.nan)
            yield_rate = row.get("user_interrupts_yield_rate", np.nan)
            yield_time = row.get("user_interrupts_yield_time_mean", np.nan)
            backchannel = row.get("backchannel_correct_rate", np.nan)

            resp_rate_str = fmt_pct(resp_rate, best_resp_rate)
            resp_latency_str = fmt_sec(resp_latency, best_resp_latency)
            yield_rate_str = fmt_pct(yield_rate, best_yield_rate)
            yield_time_str = fmt_sec(yield_time, best_yield_time)
            backchannel_str = fmt_pct(backchannel, best_backchannel)

            lines.append(
                f"{domain_label} & {model_name} & {resp_rate_str} & {resp_latency_str} & "
                f"{yield_rate_str} & {yield_time_str} & {backchannel_str} \\\\"
            )

        # Add midrule between domains (except after last)
        if domain != domains[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{small}")
    lines.append(r"\end{table}")

    # Write to file
    tex_path = output_dir / "voice_quality_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved: {tex_path}")


def main():
    """Main entry point for voice analysis CLI."""
    parser = get_cli_parser()
    args = parser.parse_args()

    # Determine mode: batch (--data-dir) or single file (results_path)
    if args.data_dir:
        # Batch mode: analyze collection of results.json files
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = Path.cwd() / data_dir

        output_dir = None
        if args.output_dir:
            output_dir = Path(args.output_dir)
            if not output_dir.is_absolute():
                output_dir = Path.cwd() / output_dir

        # Handle --clean flag: delete output directory contents before running
        if args.clean and not args.plots_only:
            import shutil

            from loguru import logger

            clean_dir = (
                output_dir if output_dir else data_dir / "analysis/voice_analysis"
            )
            if clean_dir.exists():
                logger.warning(f"Cleaning output directory: {clean_dir}")
                shutil.rmtree(clean_dir)
                logger.info(f"Deleted: {clean_dir}")

        # Handle --plots-only mode
        if args.plots_only:
            from loguru import logger

            # In plots-only mode, output_dir must exist
            if output_dir is None:
                output_dir = data_dir / "analysis/voice_analysis"
            if not output_dir.exists():
                logger.error(
                    f"Output directory {output_dir} does not exist. Cannot use --plots-only."
                )
                return
            regenerate_plots_from_csv(output_dir)
            return

        analyze_voice_results(
            data_dir=data_dir,
            output_dir=output_dir,
            filter_domains=args.filter_domains,
            tick_duration_sec=args.tick_duration,
            no_yield_window_sec=args.no_yield_window,
            backchannel_yield_window_sec=args.backchannel_yield_window,
            vocal_tic_yield_window_sec=args.vocal_tic_yield_window,
            non_directed_yield_window_sec=args.non_directed_yield_window,
            vocal_tic_response_window_sec=args.vocal_tic_response_window,
            non_directed_response_window_sec=args.non_directed_response_window,
        )

    elif args.results_path:
        # Single file mode: process one results.json file
        results_path = Path(args.results_path)

        # Import simulation classes here to avoid loading full tau2 package at module level
        from tau2.data_model.simulation import Results, SimulationRun

        if results_path.name == "simulation.json":
            # Load single simulation
            with open(results_path, "r") as f:
                sim = SimulationRun.model_validate_json(f.read())
            sims = [sim]
        else:
            # Load results.json
            results = Results.load(results_path)
            print(f"Loaded {len(results.simulations)} simulations")
            sims = results.simulations

        # Timeline mode
        if args.timeline:
            sim = sims[args.sim_index]
            if not sim.ticks:
                print(f"No ticks found in simulation {args.sim_index}")
            else:
                tick_duration = get_tick_duration(sim, args.tick_duration)
                print(f"Using tick duration: {tick_duration}s")
                # Filter out end-of-conversation artifacts before any processing
                filtered_ticks = filter_end_of_conversation_ticks(sim.ticks)
                user_segs, agent_segs = extract_all_segments(
                    filtered_ticks, tick_duration
                )
                total_duration = len(filtered_ticks) * tick_duration
                output_path = Path(args.timeline)

                # Extract out-of-turn effects (effects during gaps between speech)
                # Must be extracted before interruption_events so we can pass it
                out_of_turn = extract_out_of_turn_effects(filtered_ticks, tick_duration)
                if out_of_turn:
                    print(f"  Out-of-turn effects: {len(out_of_turn)}")

                # Extract frame drops
                frame_drops = extract_frame_drops(filtered_ticks, tick_duration)
                if frame_drops:
                    print(f"  Frame drops: {len(frame_drops)}")

                # Extract diagnostic events if requested
                turn_transitions = None
                interruption_events = None
                if args.diagnostics:
                    print("Extracting diagnostic events...")
                    turn_transitions = extract_turn_transitions(
                        user_segs,
                        agent_segs,
                        simulation_id=sim.id,
                        task_id=sim.task_id,
                    )
                    # Pass out_of_turn_effects to detect non-directed/vocal-tic events in gaps
                    interruption_events = extract_interruption_events(
                        user_segs,
                        agent_segs,
                        filtered_ticks,
                        tick_duration,
                        out_of_turn_effects=out_of_turn,
                    )
                    n_no_response = sum(
                        1 for t in turn_transitions if t.outcome == "no_response"
                    )
                    n_no_yield = sum(
                        1
                        for e in interruption_events
                        if e.event_type == "user_interrupts_agent"
                        and not e.interrupted_yielded
                    )
                    print(f"  No-response events: {n_no_response}")
                    print(f"  No-yield events: {n_no_yield}")

                # Get audio path if provided
                audio_path = Path(args.audio) if args.audio else None
                if audio_path and audio_path.exists():
                    print(f"  Loading waveform from: {audio_path}")

                save_speech_timeline(
                    user_segs,
                    agent_segs,
                    output_path,
                    total_duration_sec=total_duration,
                    simulation_id=sim.id,
                    task_id=sim.task_id,
                    turn_transitions=turn_transitions,
                    interruption_events=interruption_events,
                    out_of_turn_effects=out_of_turn,
                    frame_drops=frame_drops,
                    audio_path=audio_path,
                )
                print(f"Saved timeline to: {output_path}")
        else:
            # Default: print segment summary
            for i, sim in enumerate(sims[:3]):  # First 3 for demo
                print(f"\n{'=' * 60}")
                print(f"SIMULATION {i}: {sim.task_id}")
                print("=" * 60)
                if sim.ticks:
                    tick_duration = get_tick_duration(sim, args.tick_duration)
                    user_segs, agent_segs = extract_all_segments(
                        sim.ticks, tick_duration
                    )
                    print_segment_summary(user_segs, agent_segs)
                else:
                    print("No ticks found")
    else:
        # No arguments provided, show help
        parser.print_help()


if __name__ == "__main__":
    main()
