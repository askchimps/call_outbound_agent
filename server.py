from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from livekit import api
import os
import uuid
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.local")

app = FastAPI(title="Outbound Caller API", version="1.0.0")


class DispatchCallRequest(BaseModel):
    phone_number: str
    transfer_to: str | None = None


class DispatchCallResponse(BaseModel):
    dispatch_id: str
    room_name: str
    phone_number: str
    status: str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/dispatch", response_model=DispatchCallResponse)
async def dispatch_call(request: DispatchCallRequest):
    """
    Dispatch an outbound call to the specified phone number.
    
    - **phone_number**: The phone number to call (e.g., "+919088572757")
    - **transfer_to**: Optional phone number to transfer the call to when requested
    """
    livekit_url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not all([livekit_url, api_key, api_secret]):
        raise HTTPException(
            status_code=500,
            detail="LiveKit credentials not configured"
        )

    try:
        lk_api = api.LiveKitAPI(
            url=livekit_url,
            api_key=api_key,
            api_secret=api_secret,
        )

        room_name = f"call-{uuid.uuid4().hex[:12]}"
        
        metadata = {
            "phone_number": request.phone_number,
            "transfer_to": request.transfer_to or "",
        }

        dispatch = await lk_api.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name="outbound-caller",
                room=room_name,
                metadata=str(metadata).replace("'", '"'),
            )
        )

        await lk_api.aclose()

        return DispatchCallResponse(
            dispatch_id=dispatch.dispatch_id,
            room_name=room_name,
            phone_number=request.phone_number,
            status="dispatched",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

