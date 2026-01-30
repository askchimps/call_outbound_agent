from __future__ import annotations

import asyncio
import re
from dotenv import load_dotenv
import json
import os
import time
from datetime import datetime
from typing import AsyncIterable

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    cli,
    WorkerOptions,
    RoomInputOptions,
    ModelSettings,
)
from livekit.plugins import (
    deepgram,
    openai,
    elevenlabs,
    silero,
    noise_cancellation,  # noqa: F401
)
from livekit.plugins.turn_detector.english import EnglishModel


# load environment variables, this is optional, only used for local development
load_dotenv(dotenv_path=".env.local")


def perf_log(service: str, message: str, duration_ms: float = None):
    """Log performance metrics with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if duration_ms is not None:
        print(f"[PERF_LOG - {timestamp}] [{service}] {message} - {duration_ms:.2f}ms", flush=True)
    else:
        print(f"[PERF_LOG - {timestamp}] [{service}] {message}", flush=True)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")


class OutboundCaller(Agent):
    def __init__(self):
        # Read base prompt from external file
        with open("baseprompt.txt", "r") as f:
            instructions = f.read()

        super().__init__(instructions=instructions)
        # keep reference to the participant
        self.participant: rtc.RemoteParticipant | None = None

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def transcription_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[str]:
        """Filter out <thinking> tags from LLM output before TTS"""
        # Buffer to handle thinking tags that may span multiple chunks
        buffer = ""
        thinking_pattern = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL)

        async for chunk in text:
            buffer += chunk

            # Process complete thinking blocks
            while True:
                match = thinking_pattern.search(buffer)
                if match:
                    # Yield text before the thinking block
                    before = buffer[: match.start()]
                    if before:
                        yield before
                    # Skip the thinking content, continue after it
                    buffer = buffer[match.end() :]
                else:
                    break

            # Check if we're inside an incomplete thinking tag
            open_tag_match = re.search(r"<think(?:ing)?>[^<]*$", buffer)
            if open_tag_match:
                # Yield everything before the opening tag
                before = buffer[: open_tag_match.start()]
                if before:
                    yield before
                buffer = buffer[open_tag_match.start() :]
            else:
                # No incomplete thinking tag, yield everything except potential partial tag
                # Keep potential partial opening tags in buffer
                partial_match = re.search(r"<(?:t(?:h(?:i(?:n(?:k(?:i(?:n(?:g)?)?)?)?)?)?)?)?$", buffer)
                if partial_match:
                    before = buffer[: partial_match.start()]
                    if before:
                        yield before
                    buffer = buffer[partial_match.start() :]
                else:
                    if buffer:
                        yield buffer
                    buffer = ""

        # Yield any remaining content (filtered)
        if buffer:
            # Final cleanup of any remaining thinking tags
            cleaned = thinking_pattern.sub("", buffer)
            if cleaned:
                yield cleaned


async def entrypoint(ctx: JobContext):
    print(f"[INFO] connecting to room {ctx.room.name}", flush=True)
    await ctx.connect()

    # when dispatching the agent, we'll pass it the phone number to dial
    dial_info = json.loads(ctx.job.metadata)
    participant_identity = phone_number = dial_info["phone_number"]

    # create a new agent instance (fresh history for each call)
    agent = OutboundCaller()

    # Configure LLM - supports custom vLLM endpoint via LLM_BASE_URL
    llm_base_url = os.getenv("LLM_BASE_URL")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o")

    if llm_base_url:
        # Custom LLM (e.g., vLLM on vast.ai)
        llm = openai.LLM(
            model=llm_model,
            base_url=llm_base_url,
        )
    else:
        # Default to OpenAI
        llm = openai.LLM(model=llm_model)

    session = AgentSession(
        turn_detection=EnglishModel(),
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        tts=elevenlabs.TTS(
            voice_id=os.getenv("TTS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            model=os.getenv("TTS_MODEL_ID", "eleven_flash_v2_5"),
        ),
        llm=llm,
    )

    # Performance tracking state
    perf_state = {
        "user_speech_start": None,
        "user_speech_end": None,
        "stt_commit_time": None,
        "llm_start": None,
        "llm_first_token": None,
        "llm_complete": None,
        "tts_start": None,
        "turn_id": 0,
    }

    # VAD events - Voice Activity Detection
    @session.on("user_started_speaking")
    def on_user_started_speaking():
        perf_state["turn_id"] += 1
        perf_state["user_speech_start"] = time.time()
        perf_log("VAD", f"[Turn {perf_state['turn_id']}] User started speaking")

    @session.on("user_stopped_speaking")
    def on_user_stopped_speaking():
        perf_state["user_speech_end"] = time.time()
        if perf_state["user_speech_start"]:
            duration = (perf_state["user_speech_end"] - perf_state["user_speech_start"]) * 1000
            perf_log("VAD", f"[Turn {perf_state['turn_id']}] User stopped speaking (speech duration)", duration)

    # STT events - Speech to Text
    @session.on("user_speech_committed")
    def on_user_speech_committed(msg):
        perf_state["stt_commit_time"] = time.time()
        text_preview = msg.content[:80] if len(msg.content) > 80 else msg.content

        # Time from speech end to STT commit (STT processing time)
        if perf_state["user_speech_end"]:
            stt_latency = (perf_state["stt_commit_time"] - perf_state["user_speech_end"]) * 1000
            perf_log("STT", f"[Turn {perf_state['turn_id']}] Processing latency (speech end -> text ready)", stt_latency)

        # Total time from speech start to STT commit
        if perf_state["user_speech_start"]:
            total_stt = (perf_state["stt_commit_time"] - perf_state["user_speech_start"]) * 1000
            perf_log("STT", f"[Turn {perf_state['turn_id']}] Total STT time (speech start -> text ready)", total_stt)

        perf_log("STT", f"[Turn {perf_state['turn_id']}] Transcribed: '{text_preview}'")
        perf_state["llm_start"] = time.time()
        perf_log("LLM", f"[Turn {perf_state['turn_id']}] Request sent to LLM")

    # Agent/LLM events
    @session.on("agent_started_speaking")
    def on_agent_started_speaking():
        now = time.time()

        # LLM Time to First Token (includes TTS buffering)
        if perf_state["llm_start"] and not perf_state["llm_first_token"]:
            perf_state["llm_first_token"] = now
            ttft = (now - perf_state["llm_start"]) * 1000
            perf_log("LLM+TTS", f"[Turn {perf_state['turn_id']}] Time to first audio (LLM TTFT + TTS buffer)", ttft)

        # End-to-end latency: user stops speaking -> agent starts speaking
        if perf_state["user_speech_end"]:
            e2e_latency = (now - perf_state["user_speech_end"]) * 1000
            perf_log("E2E", f"[Turn {perf_state['turn_id']}] User stop -> Agent start (perceived latency)", e2e_latency)

        perf_state["tts_start"] = now
        perf_log("TTS", f"[Turn {perf_state['turn_id']}] Agent started speaking (audio playing)")

    @session.on("agent_stopped_speaking")
    def on_agent_stopped_speaking():
        if perf_state["tts_start"]:
            duration = (time.time() - perf_state["tts_start"]) * 1000
            perf_log("TTS", f"[Turn {perf_state['turn_id']}] Agent stopped speaking (playback duration)", duration)

    @session.on("agent_speech_committed")
    def on_agent_speech_committed(msg):
        now = time.time()
        text_preview = msg.content[:80] if len(msg.content) > 80 else msg.content

        # Full pipeline time
        if perf_state["stt_commit_time"]:
            pipeline_duration = (now - perf_state["stt_commit_time"]) * 1000
            perf_log("PIPELINE", f"[Turn {perf_state['turn_id']}] LLM + TTS total (text ready -> speech done)", pipeline_duration)

        # Total turn time
        if perf_state["user_speech_start"]:
            total_turn = (now - perf_state["user_speech_start"]) * 1000
            perf_log("TOTAL", f"[Turn {perf_state['turn_id']}] Full turn (user start -> agent done)", total_turn)

        perf_log("LLM", f"[Turn {perf_state['turn_id']}] Response: '{text_preview}'")

        # Reset for next turn
        perf_state["llm_first_token"] = None
        perf_state["llm_start"] = None
        perf_state["tts_start"] = None
        perf_state["user_speech_start"] = None
        perf_state["user_speech_end"] = None
        perf_state["stt_commit_time"] = None

    # start the session first before dialing, to ensure that when the user picks up
    # the agent does not miss anything the user says
    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                # enable Krisp background voice and noise removal
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        )
    )

    # `create_sip_participant` starts dialing the user
    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                # function blocks until user answers the call, or if the call fails
                wait_until_answered=True,
            )
        )

        # wait for the agent session start and participant join
        await session_started
        participant = await ctx.wait_for_participant(identity=participant_identity)
        print(f"[INFO] participant joined: {participant.identity}", flush=True)

        agent.set_participant(participant)

    except api.TwirpError as e:
        print(
            f"[ERROR] error creating SIP participant: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}",
            flush=True
        )
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller",
        )
    )
