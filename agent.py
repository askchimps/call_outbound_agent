from __future__ import annotations

import asyncio
import logging
import re
from dotenv import load_dotenv
import json
import os
from typing import Any, AsyncIterable

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    function_tool,
    RunContext,
    get_job_context,
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
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

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

    async def hangup(self):
        """Helper function to hang up the call by deleting the room"""

        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(
                room=job_ctx.room.name,
            )
        )

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Called when the user wants to end the call"""
        logger.info(f"ending the call for {self.participant.identity}")

        # let the agent finish speaking
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.wait_for_playout()

        await self.hangup()

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
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect()

    # when dispatching the agent, we'll pass it the phone number to dial
    dial_info = json.loads(ctx.job.metadata)
    participant_identity = phone_number = dial_info["phone_number"]

    # create a new agent instance (fresh history for each call)
    agent = OutboundCaller()

    # the following uses GPT-4o, Deepgram and ElevenLabs
    session = AgentSession(
        turn_detection=EnglishModel(),
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        tts=elevenlabs.TTS(
            voice_id=os.getenv("TTS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            model=os.getenv("TTS_MODEL_ID", "eleven_flash_v2_5"),
        ),
        llm=openai.LLM(model=os.getenv("LLM_MODEL", "gpt-4o")),
    )

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
        logger.info(f"participant joined: {participant.identity}")

        agent.set_participant(participant)

    except api.TwirpError as e:
        logger.error(
            f"error creating SIP participant: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}"
        )
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller",
        )
    )
