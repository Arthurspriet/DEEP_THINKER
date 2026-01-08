"""
Tests for API Routes.

Tests the FastAPI endpoints including:
- Health check
- Mission CRUD operations
- Agent status
- GPU stats
- Configuration endpoints
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    @pytest.mark.asyncio
    async def test_health_check(self, async_client):
        """Test health check returns healthy status."""
        response = await async_client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "DeepThinker API"
        assert "version" in data


class TestMissionEndpoints:
    """Tests for mission-related endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_missions_empty(self, async_client, temp_dir):
        """Test listing missions when store is empty."""
        with patch('api.routes.missions._store') as mock_store:
            mock_store.list_missions_with_status.return_value = []
            
            response = await async_client.get("/api/missions")
        
        assert response.status_code == 200
        assert response.json() == []
    
    @pytest.mark.asyncio
    async def test_create_mission(self, async_client):
        """Test creating a new mission."""
        with patch('api.routes.missions._get_orchestrator') as mock_orch:
            # Create mock mission state
            mock_state = MagicMock()
            mock_state.mission_id = "test-123"
            mock_state.objective = "Test objective"
            mock_state.status = "pending"
            mock_state.created_at = datetime.now()
            mock_state.remaining_minutes.return_value = 30.0
            mock_state.phases = []
            
            mock_orch.return_value.create_mission.return_value = mock_state
            
            response = await async_client.post(
                "/api/missions",
                json={
                    "objective": "Test objective",
                    "time_budget_minutes": 30,
                    "allow_internet": True,
                    "allow_code_execution": True
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["mission_id"] == "test-123"
        assert data["objective"] == "Test objective"
    
    @pytest.mark.asyncio
    async def test_create_mission_validation_error(self, async_client):
        """Test mission creation with invalid data."""
        response = await async_client.post(
            "/api/missions",
            json={
                "objective": "Test",
                "time_budget_minutes": 0,  # Invalid: must be >= 1
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_get_mission_not_found(self, async_client):
        """Test getting a non-existent mission."""
        with patch('api.routes.missions._store') as mock_store:
            mock_store.exists.return_value = False
            
            response = await async_client.get("/api/missions/nonexistent-id")
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_mission_status(self, async_client):
        """Test getting mission status."""
        with patch('api.routes.missions._store') as mock_store:
            mock_store.exists.return_value = True
            
            mock_state = MagicMock()
            mock_state.mission_id = "test-123"
            mock_state.status = "running"
            mock_state.remaining_minutes.return_value = 25.0
            mock_state.current_phase.return_value = MagicMock(name="Research")
            mock_state.current_phase_index = 1
            mock_state.phases = [MagicMock(), MagicMock(), MagicMock()]
            mock_state.is_expired.return_value = False
            mock_state.is_terminal.return_value = False
            
            mock_store.load.return_value = mock_state
            
            response = await async_client.get("/api/missions/test-123/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["mission_id"] == "test-123"
        assert data["status"] == "running"
    
    @pytest.mark.asyncio
    async def test_abort_mission(self, async_client):
        """Test aborting a mission."""
        with patch('api.routes.missions._store') as mock_store, \
             patch('api.routes.missions._get_orchestrator') as mock_orch, \
             patch('api.routes.missions.sse_manager') as mock_sse:
            
            mock_store.exists.return_value = True
            
            mock_state = MagicMock()
            mock_state.status = "aborted"
            mock_state.final_artifacts = {}
            
            mock_orch.return_value.abort_mission.return_value = mock_state
            mock_sse.publish_mission_completed = AsyncMock()
            
            response = await async_client.post(
                "/api/missions/test-123/abort",
                params={"reason": "User requested"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "aborted"
    
    @pytest.mark.asyncio
    async def test_get_mission_logs(self, async_client):
        """Test getting mission logs."""
        with patch('api.routes.missions._store') as mock_store:
            mock_store.exists.return_value = True
            
            mock_state = MagicMock()
            mock_state.logs = ["Log 1", "Log 2", "Log 3"]
            mock_store.load.return_value = mock_state
            
            response = await async_client.get(
                "/api/missions/test-123/logs",
                params={"limit": 10, "offset": 0}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["logs"]) == 3


class TestAgentEndpoints:
    """Tests for agent-related endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_agents_status(self, async_client):
        """Test getting all agents status."""
        with patch('api.routes.agents._store') as mock_store:
            mock_store.list_missions_with_status.return_value = []
            
            response = await async_client.get("/api/agents/status")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return status for all known agents
        agent_names = [a["name"] for a in data]
        assert "planner" in agent_names
        assert "researcher" in agent_names
        assert "coder" in agent_names
    
    @pytest.mark.asyncio
    async def test_get_agent_traces(self, async_client):
        """Test getting agent traces."""
        with patch('api.routes.agents._store') as mock_store:
            mock_store.list_missions.return_value = []
            
            response = await async_client.get(
                "/api/agents/planner/traces",
                params={"limit": 50}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_get_agent_traces_invalid_agent(self, async_client):
        """Test getting traces for invalid agent."""
        response = await async_client.get("/api/agents/invalid_agent/traces")
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_agent_metrics(self, async_client):
        """Test getting agent metrics."""
        with patch('api.routes.agents._store') as mock_store:
            mock_store.list_missions.return_value = []
            
            response = await async_client.get("/api/agents/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # Each metric should have expected fields
        for metric in data:
            assert "name" in metric
            assert "total_executions" in metric
            assert "successful_executions" in metric


class TestGPUEndpoints:
    """Tests for GPU-related endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_gpu_stats(self, async_client):
        """Test getting GPU statistics."""
        response = await async_client.get("/api/gpu/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have basic structure even if GPU not available
        assert "available" in data
        assert "gpu_count" in data
    
    @pytest.mark.asyncio
    async def test_get_gpu_stats_with_nvidia(self, async_client):
        """Test GPU stats when nvidia-smi is available."""
        mock_nvidia_output = "0, NVIDIA RTX 4090, 24576, 1000, 23576, 50, 45"
        
        with patch('api.routes.gpu.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=mock_nvidia_output
            )
            
            response = await async_client.get("/api/gpu")
        
        assert response.status_code == 200


class TestConfigEndpoints:
    """Tests for configuration endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_models_ollama_unavailable(self, async_client):
        """Test getting models when Ollama is unavailable."""
        with patch('api.routes.config.requests.get') as mock_get:
            mock_get.side_effect = Exception("Connection refused")
            
            response = await async_client.get("/api/config/models")
        
        assert response.status_code == 200
        data = response.json()
        assert data == []
    
    @pytest.mark.asyncio
    async def test_get_councils(self, async_client):
        """Test getting council configurations."""
        response = await async_client.get("/api/config/councils")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return list of councils
        assert isinstance(data, list)


class TestAPIValidation:
    """Tests for API input validation."""
    
    @pytest.mark.asyncio
    async def test_mission_time_budget_range(self, async_client):
        """Test time budget validation range."""
        with patch('api.routes.missions._get_orchestrator'):
            # Too low
            response = await async_client.post(
                "/api/missions",
                json={
                    "objective": "Test",
                    "time_budget_minutes": 0,
                }
            )
            assert response.status_code == 422
            
            # Too high
            response = await async_client.post(
                "/api/missions",
                json={
                    "objective": "Test",
                    "time_budget_minutes": 10000,  # Over 1440
                }
            )
            assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_mission_required_fields(self, async_client):
        """Test required fields validation."""
        # Missing objective
        response = await async_client.post(
            "/api/missions",
            json={
                "time_budget_minutes": 30,
            }
        )
        assert response.status_code == 422
        
        # Missing time_budget_minutes
        response = await async_client.post(
            "/api/missions",
            json={
                "objective": "Test",
            }
        )
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_logs_pagination_validation(self, async_client):
        """Test log pagination parameter validation."""
        with patch('api.routes.missions._store') as mock_store:
            mock_store.exists.return_value = True
            mock_state = MagicMock()
            mock_state.logs = []
            mock_store.load.return_value = mock_state
            
            # Valid pagination
            response = await async_client.get(
                "/api/missions/test-123/logs",
                params={"limit": 100, "offset": 0}
            )
            assert response.status_code == 200
            
            # Invalid limit (too high)
            response = await async_client.get(
                "/api/missions/test-123/logs",
                params={"limit": 10000}
            )
            assert response.status_code == 422


class TestCORSAndHeaders:
    """Tests for CORS and HTTP headers."""
    
    @pytest.mark.asyncio
    async def test_cors_headers(self, async_client):
        """Test CORS headers are set correctly."""
        response = await async_client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            }
        )
        
        # Should allow the request
        assert response.status_code in [200, 204]
    
    @pytest.mark.asyncio
    async def test_json_content_type(self, async_client):
        """Test responses have correct content type."""
        response = await async_client.get("/api/health")
        
        assert "application/json" in response.headers.get("content-type", "")


class TestErrorHandling:
    """Tests for API error handling."""
    
    @pytest.mark.asyncio
    async def test_internal_error_handling(self, async_client):
        """Test internal errors are handled gracefully."""
        # Test with a clearly non-existent mission ID
        # The store.exists() check should return False naturally
        with patch('api.routes.missions._store') as mock_store:
            mock_store.exists.return_value = False
            
            response = await async_client.get("/api/missions/nonexistent-test-id")
        
        # Should return 404 for non-existent mission
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_invalid_json_body(self, async_client):
        """Test handling of invalid JSON body."""
        response = await async_client.post(
            "/api/missions",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422


class TestSSEEndpoints:
    """Tests for Server-Sent Events endpoints."""
    
    @pytest.mark.asyncio
    async def test_mission_events_not_found(self, async_client):
        """Test SSE endpoint for non-existent mission."""
        with patch('api.routes.missions._store') as mock_store:
            mock_store.exists.return_value = False
            
            response = await async_client.get("/api/missions/nonexistent/events")
        
        assert response.status_code == 404


class TestAlignmentEndpoint:
    """Tests for alignment control layer endpoint."""
    
    @pytest.mark.asyncio
    async def test_alignment_not_found(self, async_client):
        """Test alignment endpoint for non-existent mission."""
        with patch('api.routes.missions._store') as mock_store:
            mock_store.exists.return_value = False
            
            response = await async_client.get("/api/missions/nonexistent/alignment")
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_alignment_disabled(self, async_client):
        """Test alignment endpoint when alignment is disabled."""
        with patch('api.routes.missions._store') as mock_store, \
             patch('deepthinker.alignment.get_alignment_config') as mock_config:
            
            mock_store.exists.return_value = True
            
            # Mock mission state without alignment data
            mock_state = MagicMock()
            mock_state.alignment_trajectory = []
            mock_state.alignment_north_star = None
            mock_state.constraints = None
            mock_store.load.return_value = mock_state
            
            # Alignment disabled
            mock_config_instance = MagicMock()
            mock_config_instance.enabled = False
            mock_config.return_value = mock_config_instance
            
            response = await async_client.get("/api/missions/test-123/alignment")
        
        assert response.status_code == 200
        data = response.json()
        assert data["mission_id"] == "test-123"
        assert data["enabled"] is False
    
    @pytest.mark.asyncio
    async def test_alignment_with_trajectory(self, async_client):
        """Test alignment endpoint with trajectory data."""
        with patch('api.routes.missions._store') as mock_store, \
             patch('deepthinker.alignment.get_alignment_config') as mock_config:
            
            mock_store.exists.return_value = True
            
            # Mock mission state with alignment data
            mock_state = MagicMock()
            mock_state.alignment_trajectory = [
                {
                    "t": 0,
                    "phase_name": "planning",
                    "a_t": 0.95,
                    "d_t": 0.0,
                    "cusum_neg": 0.0,
                    "warning": False,
                    "triggered": False,
                    "timestamp_iso": "2024-01-01T00:00:00",
                },
                {
                    "t": 1,
                    "phase_name": "research",
                    "a_t": 0.75,
                    "d_t": -0.20,
                    "cusum_neg": 0.15,
                    "warning": True,
                    "triggered": False,
                    "timestamp_iso": "2024-01-01T00:05:00",
                },
            ]
            mock_state.alignment_north_star = {
                "goal_id": "test-goal",
                "intent_summary": "Test objective",
            }
            mock_state.constraints = None
            mock_store.load.return_value = mock_state
            
            # Alignment enabled
            mock_config_instance = MagicMock()
            mock_config_instance.enabled = True
            mock_config.return_value = mock_config_instance
            
            response = await async_client.get("/api/missions/test-123/alignment")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["mission_id"] == "test-123"
        assert data["enabled"] is True
        assert len(data["trajectory"]) == 2
        assert data["trajectory"][0]["alignment_score"] == 0.95
        assert data["trajectory"][1]["warning"] is True
        assert data["summary"]["total_points"] == 2
        assert data["summary"]["warning_count"] == 1
    
    @pytest.mark.asyncio
    async def test_alignment_response_shape(self, async_client):
        """Test alignment endpoint returns correct response shape."""
        with patch('api.routes.missions._store') as mock_store, \
             patch('deepthinker.alignment.get_alignment_config') as mock_config:
            
            mock_store.exists.return_value = True
            
            mock_state = MagicMock()
            mock_state.alignment_trajectory = []
            mock_state.alignment_north_star = None
            mock_state.constraints = None
            mock_store.load.return_value = mock_state
            
            mock_config_instance = MagicMock()
            mock_config_instance.enabled = True
            mock_config.return_value = mock_config_instance
            
            response = await async_client.get("/api/missions/test-123/alignment")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response shape
        assert "mission_id" in data
        assert "enabled" in data
        assert "north_star" in data
        assert "trajectory" in data
        assert "actions_taken" in data
        assert "summary" in data
        
        # Verify summary shape
        assert "status" in data["summary"]

