"""
Interactive Web UI for Theravada Karma HMM
Main FastAPI application with WebSocket support for real-time simulation updates.
"""

import json
import asyncio
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uuid

from karma_model import TheravadaKarmaHMM, TimeUnit, MeditationType, MeditationPractice

app = FastAPI(title="Theravada Karma HMM Web UI", description="Interactive karma simulation and visualization")

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state management
active_sessions: Dict[str, Dict] = {}

# Pydantic models for API requests
class SimulationConfig(BaseModel):
    time_unit: str = "MONTHS"
    time_scale_factor: float = 1.0
    simulation_steps: int = 30

class MeditationPracticeConfig(BaseModel):
    practice_type: str
    daily_duration: float
    consistency: float
    quality: float
    years_practiced: float
    teacher_guidance: float
    retreat_hours: int

class ActionConfig(BaseModel):
    intention_strength: float
    active_kilesas: Dict[str, float]
    object_weight: float = 1.0
    wholesome: bool = False

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_personal_message(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except:
                # Connection might be closed
                self.disconnect(session_id)

    async def broadcast(self, message: dict):
        for session_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, session_id)

manager = WebSocketManager()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main application page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket, session_id)

    # Initialize session
    if session_id not in active_sessions:
        active_sessions[session_id] = {
            "karma_model": None,
            "config": SimulationConfig(),
            "is_running": False
        }

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message["type"] == "init_simulation":
                await handle_init_simulation(session_id, message["data"])
            elif message["type"] == "add_meditation":
                await handle_add_meditation(session_id, message["data"])
            elif message["type"] == "perform_action":
                await handle_perform_action(session_id, message["data"])
            elif message["type"] == "advance_time":
                await handle_advance_time(session_id, message["data"])
            elif message["type"] == "get_visualization_data":
                await handle_get_visualization_data(session_id, message["data"])
            elif message["type"] == "reset_simulation":
                await handle_reset_simulation(session_id)
            elif message["type"] == "start_continuous_simulation":
                await handle_start_continuous_simulation(session_id, message["data"])
            elif message["type"] == "stop_continuous_simulation":
                await handle_stop_continuous_simulation(session_id)

    except WebSocketDisconnect:
        manager.disconnect(session_id)
        if session_id in active_sessions:
            del active_sessions[session_id]

async def handle_init_simulation(session_id: str, data: dict):
    """Initialize a new karma simulation"""
    try:
        config = SimulationConfig(**data)
        time_unit = TimeUnit(config.time_unit)

        karma_model = TheravadaKarmaHMM(
            time_unit=time_unit,
            time_scale_factor=config.time_scale_factor
        )

        active_sessions[session_id]["karma_model"] = karma_model
        active_sessions[session_id]["config"] = config

        response = {
            "type": "simulation_initialized",
            "data": {
                "session_id": session_id,
                "time_label": karma_model.time_scale.get_display_label(),
                "current_time": karma_model.time_scale.format_time_point(karma_model.current_time)
            }
        }

        await manager.send_personal_message(response, session_id)

    except Exception as e:
        error_response = {
            "type": "error",
            "data": {"message": f"Failed to initialize simulation: {str(e)}"}
        }
        await manager.send_personal_message(error_response, session_id)

async def handle_add_meditation(session_id: str, data: dict):
    """Add meditation practice to the session"""
    try:
        session = active_sessions[session_id]
        karma_model = session["karma_model"]

        if not karma_model:
            raise ValueError("No active simulation")

        practice_config = MeditationPracticeConfig(**data)
        practice_type = MeditationType(practice_config.practice_type)

        meditation = MeditationPractice(
            practice_type=practice_type,
            daily_duration=practice_config.daily_duration,
            consistency=practice_config.consistency,
            quality=practice_config.quality,
            years_practiced=practice_config.years_practiced,
            teacher_guidance=practice_config.teacher_guidance,
            retreat_hours=practice_config.retreat_hours
        )

        karma_model.add_meditation_practice(meditation)

        response = {
            "type": "meditation_added",
            "data": {
                "practice_type": practice_config.practice_type,
                "total_practices": len(karma_model.meditation_practices)
            }
        }

        await manager.send_personal_message(response, session_id)

    except Exception as e:
        error_response = {
            "type": "error",
            "data": {"message": f"Failed to add meditation: {str(e)}"}
        }
        await manager.send_personal_message(error_response, session_id)

async def handle_perform_action(session_id: str, data: dict):
    """Perform an action in the karma simulation"""
    try:
        session = active_sessions[session_id]
        karma_model = session["karma_model"]

        if not karma_model:
            raise ValueError("No active simulation")

        action_config = ActionConfig(**data)

        seeds = karma_model.perform_action(
            intention_strength=action_config.intention_strength,
            active_kilesas=action_config.active_kilesas,
            object_weight=action_config.object_weight,
            wholesome=action_config.wholesome
        )

        # Get current state
        state_summary = karma_model.get_state_summary()

        response = {
            "type": "action_performed",
            "data": {
                "seeds_created": len(seeds),
                "state": state_summary,
                "current_time": karma_model.time_scale.format_time_point(karma_model.current_time)
            }
        }

        await manager.send_personal_message(response, session_id)

    except Exception as e:
        error_response = {
            "type": "error",
            "data": {"message": f"Failed to perform action: {str(e)}"}
        }
        await manager.send_personal_message(error_response, session_id)

async def handle_advance_time(session_id: str, data: dict):
    """Advance simulation time"""
    try:
        session = active_sessions[session_id]
        karma_model = session["karma_model"]

        if not karma_model:
            raise ValueError("No active simulation")

        time_steps = data.get("steps", 1)
        context_updates = data.get("context", {})

        for _ in range(time_steps):
            karma_model.advance_time(1, context_updates)

        # Get current state
        state_summary = karma_model.get_state_summary()

        response = {
            "type": "time_advanced",
            "data": {
                "steps": time_steps,
                "state": state_summary,
                "current_time": karma_model.time_scale.format_time_point(karma_model.current_time)
            }
        }

        await manager.send_personal_message(response, session_id)

    except Exception as e:
        error_response = {
            "type": "error",
            "data": {"message": f"Failed to advance time: {str(e)}"}
        }
        await manager.send_personal_message(error_response, session_id)

async def handle_get_visualization_data(session_id: str, data: dict):
    """Get data for visualizations"""
    try:
        session = active_sessions[session_id]
        karma_model = session["karma_model"]

        if not karma_model:
            raise ValueError("No active simulation")

        viz_type = data.get("type", "evolution")

        if viz_type == "evolution":
            # Prepare data for karmic evolution chart
            if karma_model.history_log:
                import pandas as pd
                df = pd.DataFrame(karma_model.history_log)

                # Include both state updates and action logs for more comprehensive data
                relevant_data = df[df['action_type'].isin(['state_update', 'wholesome', 'unwholesome'])]

                if not relevant_data.empty:
                    # Sort by time to ensure proper ordering
                    relevant_data = relevant_data.sort_values('time')
                    display_times = [karma_model.time_scale.convert_to_display_units(t) for t in relevant_data['time']]

                    chart_data = {
                        "time": display_times,
                        "time_label": karma_model.time_scale.get_display_label(),
                        "wholesome": relevant_data['total_accumulated_wholesome'].tolist(),
                        "unwholesome": relevant_data['total_accumulated_unwholesome'].tolist(),
                        "karmic_balance": relevant_data.get('karmic_balance', [0] * len(relevant_data)).tolist(),
                        "active_seeds": relevant_data['active_seeds_count'].fillna(0).tolist(),
                        "path_stage": relevant_data['path_stage'].tolist(),
                        "meditation_suppression": relevant_data.get('meditation_suppression', [0] * len(relevant_data)).tolist(),
                        "meditation_effectiveness": relevant_data.get('meditation_effectiveness', [0] * len(relevant_data)).tolist(),
                        "kilesa_suppression_rate": relevant_data.get('kilesa_suppression_rate', [0] * len(relevant_data)).tolist()
                    }
                else:
                    chart_data = {"message": "No data available yet"}
            else:
                chart_data = {"message": "No simulation data available"}

        elif viz_type == "network":
            # Prepare kilesa network data
            nodes = []
            edges = []

            # Add nodes for current kilesas
            current_kilesas = karma_model.current_kilesas.to_dict()
            for kilesa, strength in current_kilesas.items():
                if strength > 0.01:
                    nodes.append({
                        "id": kilesa,
                        "label": kilesa.replace('_', ' ').title(),
                        "value": strength,
                        "color": "#ff6b6b" if strength > 0.5 else "#feca57"
                    })

            # Add edges for kilesa interactions
            for (k1, k2), weight in karma_model.kilesa_interactions.items():
                if k1 in current_kilesas and k2 in current_kilesas:
                    if current_kilesas[k1] > 0.01 and current_kilesas[k2] > 0.01:
                        edges.append({
                            "from": k1,
                            "to": k2,
                            "value": abs(weight),
                            "color": "#e74c3c" if weight > 0 else "#3498db",
                            "title": f"{k1} ↔ {k2}: {weight:.2f}"
                        })

            chart_data = {"nodes": nodes, "edges": edges}

        elif viz_type == "patterns":
            # Prepare karmic seed patterns data
            if karma_model.history_log:
                patterns = {}
                for entry in karma_model.history_log:
                    if entry.get('action_type') == 'action':
                        seed_type = 'wholesome' if entry.get('wholesome', False) else 'unwholesome'
                        patterns[seed_type] = patterns.get(seed_type, 0) + entry.get('seeds_created', 0)

                chart_data = {"patterns": patterns} if patterns else {"message": "No pattern data available"}
            else:
                chart_data = {"message": "No simulation data available"}

        else:
            chart_data = {"message": f"Unknown visualization type: {viz_type}"}

        response = {
            "type": "visualization_data",
            "data": {
                "viz_type": viz_type,
                "chart_data": chart_data
            }
        }

        await manager.send_personal_message(response, session_id)

    except Exception as e:
        error_response = {
            "type": "error",
            "data": {"message": f"Failed to get visualization data: {str(e)}"}
        }
        await manager.send_personal_message(error_response, session_id)

async def handle_reset_simulation(session_id: str):
    """Reset the karma simulation"""
    try:
        if session_id in active_sessions:
            active_sessions[session_id]["karma_model"] = None
            active_sessions[session_id]["is_running"] = False

        response = {
            "type": "simulation_reset",
            "data": {"message": "Simulation reset successfully"}
        }

        await manager.send_personal_message(response, session_id)

    except Exception as e:
        error_response = {
            "type": "error",
            "data": {"message": f"Failed to reset simulation: {str(e)}"}
        }
        await manager.send_personal_message(error_response, session_id)

async def handle_start_continuous_simulation(session_id: str, data: dict):
    """Start continuous simulation mode"""
    try:
        session = active_sessions[session_id]
        karma_model = session["karma_model"]

        if not karma_model:
            raise ValueError("No active simulation")

        # Set simulation as running
        session["is_running"] = True
        session["continuous_config"] = data

        response = {
            "type": "continuous_simulation_started",
            "data": {
                "speed": data.get("speed", 1000),
                "auto_action_rate": data.get("auto_action_rate", 0.3)
            }
        }

        await manager.send_personal_message(response, session_id)

    except Exception as e:
        error_response = {
            "type": "error",
            "data": {"message": f"Failed to start continuous simulation: {str(e)}"}
        }
        await manager.send_personal_message(error_response, session_id)

async def handle_stop_continuous_simulation(session_id: str):
    """Stop continuous simulation mode"""
    try:
        session = active_sessions[session_id]

        if session_id in active_sessions:
            session["is_running"] = False
            session["continuous_config"] = None

        response = {
            "type": "continuous_simulation_stopped",
            "data": {"message": "Continuous simulation stopped"}
        }

        await manager.send_personal_message(response, session_id)

    except Exception as e:
        error_response = {
            "type": "error",
            "data": {"message": f"Failed to stop continuous simulation: {str(e)}"}
        }
        await manager.send_personal_message(error_response, session_id)

# REST API endpoints for additional functionality
@app.get("/api/session")
async def create_session():
    """Create a new session ID"""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

@app.get("/api/kilesas")
async def get_available_kilesas():
    """Get list of available kilesas for UI"""
    # This would come from the karma model
    kilesas = [
        "anger", "hatred", "greed", "delusion", "conceit", "envy",
        "jealousy", "arrogance", "vanity", "hostility", "resentment",
        "attachment", "craving", "aversion", "ignorance", "pride",
        "sloth", "torpor", "restlessness", "worry", "doubt",
        "shamelessness", "fearlessness", "wrong_view", "wrong_intention",
        "wrong_speech", "wrong_action", "wrong_livelihood"
    ]
    return {"kilesas": kilesas}

@app.get("/api/meditation_types")
async def get_meditation_types():
    """Get available meditation types"""
    types = [type.value for type in MeditationType]
    return {"meditation_types": types}

@app.get("/api/time_units")
async def get_time_units():
    """Get available time units"""
    units = [unit.value for unit in TimeUnit]
    return {"time_units": units}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)