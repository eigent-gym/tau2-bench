# Speech Timeline Analysis Methodology

This document describes how the speech timeline visualization and its underlying events are computed in `voice_analysis.py`. The timeline is central to analyzing voice agent behavior, including turn-taking, interruption handling, and response latency.

## Table of Contents

1. [Overview](#overview)
2. [Agent Error Definitions](#agent-error-definitions)
3. [Data Model: Ticks](#data-model-ticks)
4. [Preprocessing](#preprocessing)
5. [Speech Segment Extraction](#speech-segment-extraction)
6. [Turn Transition Events](#turn-transition-events)
7. [Interruption Events](#interruption-events)
8. [Audio Effects](#audio-effects)
9. [Frame Drops](#frame-drops)
10. [Timeline Visualization](#timeline-visualization)
11. [Configuration Parameters](#configuration-parameters)

---

## Overview

The analysis pipeline transforms raw simulation data (tick-by-tick state) into meaningful conversational events. The process flows as follows:

```
Raw Ticks → Preprocessing → Segment Extraction → Event Detection → Visualization
```

Each simulation consists of discrete **ticks** (default: 200ms each). At each tick, we have information about:
- Whether the user is speaking (`user_chunk.contains_speech`)
- Whether the agent is speaking (`agent_chunk.contains_speech`)
- Audio effects applied (vocal tics, non-directed speech, burst noise, muffling)
- Turn-taking decisions made by the user simulator
- VAD (Voice Activity Detection) events from the agent

---

## Agent Error Definitions

The following agent errors are tracked and reported in metrics.

**Key timing parameters (configurable via CLI):**
- `--no-yield-window`: 2.0 seconds (default) - time for agent to yield after user interruption
- `--backchannel-yield-window`: 1.0 seconds (default) - time to detect incorrect yield after backchannel
- `--vocal-tic-yield-window`: 1.0 seconds (default) - time to detect incorrect yield after vocal tic
- `--non-directed-yield-window`: 1.0 seconds (default) - time to detect incorrect yield after non-directed speech
- `--vocal-tic-response-window`: 2.0 seconds (default) - time to detect incorrect response to vocal tic
- `--non-directed-response-window`: 2.0 seconds (default) - time to detect incorrect response to non-directed speech

### Turn-Taking Errors

| Error | Condition | Description |
|-------|-----------|-------------|
| **No-Response** | User finishes speaking → User speaks again before agent responds | Agent failed to respond to user's turn. User had to retry or repeat themselves. |
| **No-Yield** | User interrupts agent → Agent does not stop within 2.0s | Agent failed to yield the floor when user interrupted. Agent kept talking over the user for more than the yield window. |
| **Agent Interruption** | User speaking → Agent starts speaking | Agent started speaking while user was still talking. Agent should wait for user to finish before responding. |

### Backchannel Handling Errors

| Error | Condition | Description |
|-------|-----------|-------------|
| **Backchannel Yield** | Agent speaking → User gives backchannel → Agent stops within 1.0s | Agent incorrectly interpreted a backchannel ("mm-hmm", "uh-huh") as an interruption and stopped speaking. Agent should have continued. |

### Audio Robustness Errors

These errors test whether the agent can distinguish real user speech from noise or non-directed audio.

| Error | Condition | Description |
|-------|-----------|-------------|
| **Vocal Tic Yield** | Agent speaking → Vocal tic occurs → Agent stops within 1.0s | Agent incorrectly stopped speaking when user had a vocal tic (cough, sneeze, "um"). Agent should have continued. |
| **Non-Directed Yield** | Agent speaking → Non-directed speech occurs → Agent stops within 1.0s | Agent incorrectly stopped when user spoke to someone else ("Hold on", "One sec"). Agent should have continued. |
| **Responds to Vocal Tic** | Agent silent → Vocal tic occurs → Agent starts speaking within 2.0s | Agent incorrectly treated a vocal tic as a user turn and responded. Agent should have stayed silent. |
| **Responds to Non-Directed** | Agent silent → Non-directed speech occurs → Agent starts speaking within 2.0s | Agent incorrectly treated non-directed speech as a user turn and responded. Agent should have stayed silent. |

### Summary Table

| Error Type | Agent State | Trigger | Incorrect Behavior | Time Window |
|------------|-------------|---------|-------------------|-------------|
| No-Response | Silent | User turn ends | No response before user retries | — |
| No-Yield | Speaking | User interrupts | Keep speaking | 2.0s |
| Agent Interruption | Any | User speaking | Start speaking over user | — |
| Backchannel Yield | Speaking | User backchannel | Stop speaking | 1.0s |
| Vocal Tic Yield | Speaking | Vocal tic | Stop speaking | 1.0s |
| Non-Directed Yield | Speaking | Non-directed speech | Stop speaking | 1.0s |
| Responds to Vocal Tic | Silent | Vocal tic | Start speaking | 2.0s |
| Responds to Non-Directed | Silent | Non-directed speech | Start speaking | 2.0s |

---

## Data Model: Ticks

A **Tick** is the fundamental unit of the discrete-time simulation. Each tick contains:

| Field | Description |
|-------|-------------|
| `user_chunk.contains_speech` | Boolean: is the user speaking? |
| `user_chunk.content` | Transcript text for this tick |
| `user_chunk.turn_taking_action` | User's decision (keep_talking, stop_talking, backchannel, etc.) |
| `user_chunk.audio_effects` | Active audio effects (vocal_tic, non_directed_speech, etc.) |
| `agent_chunk.contains_speech` | Boolean: is the agent speaking? |
| `agent_chunk.content` | Agent's transcript for this tick |
| `agent_chunk.raw_data.vad_events` | VAD events (speech_started, speech_stopped, interrupted) |
| `agent_chunk.raw_data.cumulative_user_audio_at_tick_start_ms` | Precise timing in milliseconds |

---

## Preprocessing

### End-of-Conversation Artifact Removal

**Function:** `filter_end_of_conversation_ticks()`

Conversations end when the user outputs `###STOP###`. This creates a spurious 1-tick user speech segment at the very end that would trigger false positive "No Yield" events.

**Logic:**
1. Check if the last tick has `user_chunk.contains_speech = True`
2. Check if the previous tick did NOT have user speech
3. If both conditions are met → remove the last tick

This prevents artificial no-yield events from the conversation termination signal.

---

## Speech Segment Extraction

### User Speech Segments

**Function:** `extract_user_segments()`

Groups contiguous ticks where `user_chunk.contains_speech = True` into `UserSpeechSegment` objects.

**For each segment, we extract:**

| Field | How Computed |
|-------|--------------|
| `start_tick`, `end_tick` | First and last+1 tick indices where speech is present |
| `start_time_sec`, `end_time_sec` | `tick_idx × tick_duration_sec` |
| `start_time_ms` | Precise timing from `cumulative_user_audio_at_tick_start_ms` |
| `transcript` | Concatenation of `user_chunk.content` across all ticks in segment |
| `action` | Turn-taking action from first tick (keep_talking, stop_talking, backchannel, etc.) |
| `is_interruption` | `True` if agent was speaking at segment start AND not a backchannel |
| `is_backchannel` | `True` if `action == "backchannel"` |
| `other_speaking_at_start` | `True` if `agent_chunk.contains_speech` at first tick |
| `other_speaking_at_end` | `True` if agent was speaking at segment end (computed during finalization) |
| `audio_effects` | List of `AudioEffectSegment` objects during this segment |
| `has_vocal_tic` | `True` if any audio effect is of type `vocal_tic` |
| `has_non_directed_speech` | `True` if any effect is `non_directed_speech` |

### Agent Speech Segments

**Function:** `extract_agent_segments()`

Same approach as user segments, grouping contiguous ticks where `agent_chunk.contains_speech = True`.

**Additional fields for agent segments:**

| Field | How Computed |
|-------|--------------|
| `was_interrupted` | `True` if user speech started during this agent segment |
| `vad_events` | List of VAD events that occurred during this segment |
| `truncated_audio_bytes` | Amount of audio truncated due to interruption |

---

## Turn Transition Events

**Function:** `extract_turn_transitions()`

Analyzes what happens after each valid user speech segment ends.

### Filtering: Which User Segments Are Analyzed?

Not all user segments trigger turn transition analysis:

| Condition | Action |
|-----------|--------|
| `is_backchannel = True` | **Skip** - Agent should not respond to backchannels |
| `is_interruption = True` AND `other_speaking_at_end = True` | **Skip** - User interrupted but didn't take the floor |
| `other_speaking_at_end = True` | **Skip** - Agent was already speaking |

### Outcome Classification

For each valid user segment, we determine what happened next:

**"response"** - Agent started speaking before user spoke again
```
User ends speaking → Agent starts → ... (Agent responded)
```
- `gap_sec` = time from user end to agent start

**"no_response"** - User spoke again before agent responded
```
User ends speaking → User starts again → ... (Agent failed to respond)
```
- `gap_sec` = time from user end to next user start

### Algorithm

```python
for each valid_user_segment:
    next_agent = first agent segment starting >= user.end_tick
    next_user = first valid user segment starting >= user.end_tick
    
    if next_agent starts before next_user:
        outcome = "response"
        gap = next_agent.start - user.end
    else:
        outcome = "no_response"  
        gap = next_user.start - user.end
```

---

## Interruption Events

**Function:** `extract_interruption_events()`

Identifies and classifies all overlap events where one party starts speaking while the other is already speaking.

### Event Types

| Event Type | Description | Expected Agent Behavior |
|------------|-------------|------------------------|
| `user_interrupts_agent` | User starts speaking while agent is talking | Agent should yield |
| `agent_interrupts_user` | Agent starts speaking while user is talking | User should yield |
| `backchannel` | User gives a backchannel while agent is talking | Agent should NOT yield |
| `vocal_tic` | User vocal tic while agent is talking | Agent should NOT yield |
| `non_directed_speech` | User talks to someone else while agent talks | Agent should NOT yield |
| `agent_responds_to_vocal_tic` | Agent starts speaking after vocal tic (error) | Should NOT have responded |
| `agent_responds_to_non_directed` | Agent starts speaking after non-directed (error) | Should NOT have responded |
| `vocal_tic_silent_correct` | Vocal tic when agent silent, agent stayed silent | Correct behavior |
| `non_directed_silent_correct` | Non-directed when agent silent, agent stayed silent | Correct behavior |

### Priority for Classification

When user starts while agent is speaking, we classify with this priority:

```
backchannel > vocal_tic > non_directed_speech > user_interrupts_agent
```

### Yield Detection

For interruptions, we check whether the interrupted party **yielded** (stopped speaking):

```python
yield_window_ticks = yield_window_sec / tick_duration_sec  # default: 3.0s / 0.2s = 15 ticks

for tick in range(start_tick, start_tick + yield_window_ticks):
    if interrupted_party.contains_speech == False:
        yielded = True
        yield_tick = tick
        break
```

**Key metrics:**
- `interrupted_yielded`: Boolean - did the interrupted party stop?
- `yield_time_sec`: How long until they stopped (if they did)

### Agent Response Detection (for vocal tics / non-directed)

For vocal tics and non-directed speech when agent was silent:

```python
response_window_ticks = 2.0 / tick_duration_sec  # 2 second window

# If agent starts speaking within 2 seconds of effect end:
#   → "agent_responds_to_vocal_tic" or "agent_responds_to_non_directed" (ERROR)
# If agent does NOT start:
#   → "vocal_tic_silent_correct" or "non_directed_silent_correct" (CORRECT)
```

---

## Audio Effects

**Function:** `extract_out_of_turn_effects()`

Extracts audio effects that occur during conversation gaps (when `contains_speech = False`).

### Effect Types

| Effect Type | Description |
|-------------|-------------|
| `burst_noise` | Transient noise (cough, door slam, keyboard) |
| `vocal_tic` | Brief vocalization (um, uh, throat clear) |
| `non_directed_speech` | User talking to someone else ("Hold on", "One sec") |
| `muffling` | Audio quality degradation |

### In-Turn vs Out-of-Turn

- **In-turn effects**: Occur during a speech segment, stored in segment's `audio_effects` list
- **Out-of-turn effects**: Occur during silence, extracted separately for diagnostic analysis

### Detection Algorithm

For each tick where `contains_speech = False`:
1. Check `user_chunk.audio_effects` for active effects
2. Track effect start/end to form `AudioEffectSegment` objects
3. Record timing and effect text

---

## Frame Drops

**Function:** `extract_frame_drops()`

Extracts simulated network packet loss events from `channel_effects`.

### Detection

```python
for each tick:
    if user_chunk.channel_effects.frame_drops_enabled:
        if user_chunk.channel_effects.frame_drop_ms > 0:
            record FrameDropEvent(
                tick_idx=i,
                time_sec=i * tick_duration_sec,
                duration_ms=frame_drop_ms,
                during_speech=user_chunk.contains_speech
            )
```

---

## Timeline Visualization

**Function:** `plot_speech_timeline()`

Creates a visual representation of the conversation dynamics.

### Visual Elements

| Element | Appearance | Meaning |
|---------|------------|---------|
| Blue bars | Upper row | User speech segments |
| Red bars | Lower row | Agent speech segments |
| Purple regions | Overlap of blue/red | Simultaneous speech |
| ▼ (orange) | On user bar | User interruption |
| ▲ (orange) | On agent bar | Agent interruption |
| ○ (green) | On user bar | Backchannel |
| ✗ (red) | Below timeline | No-response event (user had to retry) |
| ⊘ (dark red) | Below agent bar | No-yield event (agent didn't stop when interrupted) |
| ! (red) | Below agent bar | Backchannel yield error (agent incorrectly stopped) |
| ∼✗ (red) | Below agent bar | Vocal tic yield error (agent incorrectly stopped) |
| …✗ (red) | Below agent bar | Non-directed yield error (agent incorrectly stopped) |
| Waveform | Behind bars | Audio amplitude (if audio file provided) |

### Diagnostic Markers

When `--diagnostics` is enabled:

1. **No-Response Markers (✗)**: Placed at user segment end when `outcome == "no_response"`
2. **No-Yield Markers (⊘)**: Placed when `interrupted_yielded == False` for user interruptions
3. **Backchannel Yield Error (!)**: When agent incorrectly stopped after user backchannel (`event_type == "backchannel"` and `interrupted_yielded == True`)
4. **Vocal Tic Yield Error (∼✗)**: Agent incorrectly stopped after vocal tic
5. **Non-Directed Yield Error (…✗)**: Agent incorrectly stopped after non-directed speech
6. **Incorrect Response (∼✗ / …✗)**: Agent responded to vocal tic or non-directed speech when silent

---

## Configuration Parameters

### Tick Duration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tick_duration_sec` | 0.2 | Duration of each tick in seconds |

The tick duration is automatically extracted from simulation metadata if available:
```python
tick_duration = sim.params.tick_duration_sec or default
```

### Yield Window

| Parameter | Default | Description |
|-----------|---------|-------------|
| `yield_window_sec` | 3.0 | Time window to detect if interrupted party yields |

A shorter window (e.g., 2.0s) is stricter; a longer window (e.g., 5.0s) is more lenient.

### Response Window

| Parameter | Default | Description |
|-----------|---------|-------------|
| `response_window_ticks` | 10 (2.0s) | Window to detect incorrect agent response to tic/non-directed |

---

## Summary: Key Metrics

### Turn-Taking Quality

| Metric | What It Measures |
|--------|------------------|
| Response rate | % of user turns that get agent response |
| Response latency | Time from user end to agent start |
| No-response count | How often user had to repeat themselves |

### Interruption Handling

| Metric | What It Measures |
|--------|------------------|
| Yield rate | % of interruptions where interrupted party yields |
| Yield latency | Time until interrupted party stops |
| Backchannel preservation | Does agent continue after user backchannel? |

### Robustness to Noise

| Metric | What It Measures |
|--------|------------------|
| Vocal tic robustness | Does agent correctly ignore vocal tics? |
| Non-directed speech robustness | Does agent correctly ignore non-directed speech? |
| False response rate | How often agent incorrectly responds to noise |

---

## Code Reference

Main functions in `voice_analysis.py`:

```python
# Preprocessing
filter_end_of_conversation_ticks(ticks) -> List[Tick]

# Segment extraction
extract_user_segments(ticks, tick_duration_sec) -> List[UserSpeechSegment]
extract_agent_segments(ticks, tick_duration_sec) -> List[AgentSpeechSegment]
extract_all_segments(ticks, tick_duration_sec) -> Tuple[List, List]

# Event detection
extract_turn_transitions(user_segments, agent_segments, ...) -> List[TurnTransitionEvent]
extract_interruption_events(user_segments, agent_segments, ticks, ...) -> List[InterruptionEvent]
extract_out_of_turn_effects(ticks, tick_duration_sec) -> List[AudioEffectSegment]
extract_frame_drops(ticks, tick_duration_sec) -> List[FrameDropEvent]

# Visualization
plot_speech_timeline(user_segments, agent_segments, ...) -> Figure
save_speech_timeline(user_segments, agent_segments, output_path, ...) -> Path
```
