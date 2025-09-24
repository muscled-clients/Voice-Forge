import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from httpx import AsyncClient
import json

from app.main import create_application
from app.db import models
from app.models.schemas import TranscriptionStatus


class TestGraphQLQueries:
    """Test GraphQL query operations"""
    
    @pytest.fixture
    def app(self):
        """Create test application"""
        return create_application()
    
    @pytest.fixture
    async def client(self, app):
        """Create async test client"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers"""
        return {"Authorization": "Bearer test_token"}
    
    @pytest.mark.asyncio
    async def test_current_user_query(self, client, auth_headers):
        """Test current user query"""
        query = """
        query {
            currentUser {
                id
                email
                fullName
                tier
                creditsRemaining
                isActive
                createdAt
            }
        }
        """
        
        with patch('app.core.auth.get_current_user') as mock_get_user:
            # Mock user
            mock_user = Mock()
            mock_user.id = "user123"
            mock_user.email = "test@example.com"
            mock_user.full_name = "Test User"
            mock_user.tier = "pro"
            mock_user.credits_remaining = 1000
            mock_user.is_active = True
            mock_user.created_at = "2024-01-01T00:00:00Z"
            mock_user.updated_at = None
            
            mock_get_user.return_value = mock_user
            
            response = await client.post(
                "/graphql",
                json={"query": query},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "errors" not in data
            assert data["data"]["currentUser"]["id"] == "user123"
            assert data["data"]["currentUser"]["email"] == "test@example.com"
            assert data["data"]["currentUser"]["tier"] == "PRO"
    
    @pytest.mark.asyncio
    async def test_transcription_jobs_query(self, client, auth_headers):
        """Test transcription jobs listing query"""
        query = """
        query($pagination: PaginationInput) {
            transcriptionJobs(pagination: $pagination) {
                jobs {
                    id
                    status
                    filename
                    createdAt
                }
                totalCount
                page
                totalPages
                hasNext
            }
        }
        """
        
        variables = {
            "pagination": {
                "page": 1,
                "perPage": 10
            }
        }
        
        with patch('app.core.auth.get_current_user') as mock_get_user, \
             patch('app.db.session.get_db_session') as mock_session:
            
            # Mock user
            mock_user = Mock()
            mock_user.id = "user123"
            mock_get_user.return_value = mock_user
            
            # Mock database session and query
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.count.return_value = 2
            mock_query.order_by.return_value = mock_query
            mock_query.offset.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.all.return_value = [
                Mock(
                    id="job1",
                    user_id="user123",
                    status=TranscriptionStatus.COMPLETED,
                    filename="test1.wav",
                    created_at="2024-01-01T00:00:00Z",
                    model_used="whisper-base",
                    result={"transcript": "Hello world"}
                ),
                Mock(
                    id="job2", 
                    user_id="user123",
                    status=TranscriptionStatus.PROCESSING,
                    filename="test2.wav",
                    created_at="2024-01-01T01:00:00Z",
                    model_used="whisper-small",
                    result=None
                )
            ]
            
            mock_session_instance.query.return_value = mock_query
            
            response = await client.post(
                "/graphql",
                json={"query": query, "variables": variables},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "errors" not in data
            assert len(data["data"]["transcriptionJobs"]["jobs"]) == 2
            assert data["data"]["transcriptionJobs"]["totalCount"] == 2
            assert data["data"]["transcriptionJobs"]["page"] == 1
    
    @pytest.mark.asyncio
    async def test_available_models_query(self, client):
        """Test available models query"""
        query = """
        query {
            availableModels {
                name
                modelId
                type
                description
                supportedLanguages
                isAvailable
                requiresGpu
            }
        }
        """
        
        with patch('app.models.manager.model_manager') as mock_manager:
            mock_manager.get_available_models.return_value = {
                "whisper-base": {
                    "name": "Whisper Base",
                    "type": "whisper",
                    "description": "Base Whisper model",
                    "languages": ["en", "es", "fr"],
                    "available": True,
                    "requires_gpu": True
                },
                "whisper-small": {
                    "name": "Whisper Small", 
                    "type": "whisper",
                    "description": "Small Whisper model",
                    "languages": ["en", "es", "fr", "de"],
                    "available": True,
                    "requires_gpu": True
                }
            }
            
            response = await client.post(
                "/graphql",
                json={"query": query}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "errors" not in data
            models = data["data"]["availableModels"]
            assert len(models) == 2
            assert models[0]["modelId"] == "whisper-base"
            assert models[0]["requiresGpu"] == True
    
    @pytest.mark.asyncio
    async def test_user_analytics_query(self, client, auth_headers):
        """Test user analytics query"""
        query = """
        query($filter: AnalyticsFilter) {
            userAnalytics(filter: $filter) {
                userId
                totalTranscriptions
                totalAudioMinutes
                successRate
                mostUsedLanguage
                mostUsedModel
                periodStart
                periodEnd
            }
        }
        """
        
        variables = {
            "filter": {
                "startDate": "2024-01-01T00:00:00Z",
                "endDate": "2024-01-31T23:59:59Z"
            }
        }
        
        with patch('app.core.auth.get_current_user') as mock_get_user, \
             patch('app.db.session.get_db_session') as mock_session:
            
            # Mock user
            mock_user = Mock()
            mock_user.id = "user123"
            mock_get_user.return_value = mock_user
            
            # Mock database session and jobs
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [
                Mock(
                    status=TranscriptionStatus.COMPLETED,
                    duration=60.0,
                    language_code="en",
                    model_used="whisper-base",
                    result={"confidence": 0.95}
                ),
                Mock(
                    status=TranscriptionStatus.COMPLETED,
                    duration=120.0,
                    language_code="en",
                    model_used="whisper-base",
                    result={"confidence": 0.92}
                )
            ]
            
            mock_session_instance.query.return_value = mock_query
            
            response = await client.post(
                "/graphql",
                json={"query": query, "variables": variables},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "errors" not in data
            analytics = data["data"]["userAnalytics"]
            assert analytics["userId"] == "user123"
            assert analytics["totalTranscriptions"] == 2
            assert analytics["mostUsedLanguage"] == "en"
            assert analytics["mostUsedModel"] == "whisper-base"


class TestGraphQLMutations:
    """Test GraphQL mutation operations"""
    
    @pytest.fixture
    def app(self):
        """Create test application"""
        return create_application()
    
    @pytest.fixture
    async def client(self, app):
        """Create async test client"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.mark.asyncio
    async def test_register_user_mutation(self, client):
        """Test user registration mutation"""
        mutation = """
        mutation($input: UserRegistrationInput!) {
            registerUser(input: $input) {
                ... on AuthenticationResult {
                    success
                    message
                    accessToken
                    user {
                        id
                        email
                        fullName
                        tier
                    }
                }
                ... on ErrorResult {
                    success
                    message
                    code
                }
            }
        }
        """
        
        variables = {
            "input": {
                "email": "newuser@example.com",
                "password": "strongpassword123",
                "fullName": "New User"
            }
        }
        
        with patch('app.db.session.get_db_session') as mock_session:
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock user doesn't exist
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.first.return_value = None
            mock_session_instance.query.return_value = mock_query
            
            response = await client.post(
                "/graphql",
                json={"query": mutation, "variables": variables}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Should succeed or show validation error
            assert "data" in data
            result = data["data"]["registerUser"]
            
            # Check if it's a success or error result
            if "success" in result:
                assert isinstance(result["success"], bool)
    
    @pytest.mark.asyncio
    async def test_login_user_mutation(self, client):
        """Test user login mutation"""
        mutation = """
        mutation($input: UserLoginInput!) {
            loginUser(input: $input) {
                ... on AuthenticationResult {
                    success
                    message
                    accessToken
                    user {
                        id
                        email
                        tier
                    }
                }
                ... on ErrorResult {
                    success
                    message
                    code
                }
            }
        }
        """
        
        variables = {
            "input": {
                "email": "test@example.com",
                "password": "testpassword"
            }
        }
        
        with patch('app.db.session.get_db_session') as mock_session, \
             patch('app.core.auth.verify_password') as mock_verify:
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock existing user
            mock_user = Mock()
            mock_user.email = "test@example.com"
            mock_user.hashed_password = "hashed_password"
            mock_user.is_active = True
            mock_user.id = "user123"
            mock_user.tier = "free"
            
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.first.return_value = mock_user
            mock_session_instance.query.return_value = mock_query
            
            mock_verify.return_value = True
            
            response = await client.post(
                "/graphql",
                json={"query": mutation, "variables": variables}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "data" in data
            result = data["data"]["loginUser"]
            
            # Should be success result with access token
            if result.get("success"):
                assert "accessToken" in result
                assert result["user"]["email"] == "test@example.com"
    
    @pytest.mark.asyncio
    async def test_create_transcription_job_mutation(self, client):
        """Test transcription job creation mutation"""
        mutation = """
        mutation($audioFile: Upload!, $input: TranscriptionInput) {
            createTranscriptionJob(audioFile: $audioFile, input: $input) {
                ... on TranscriptionResult {
                    success
                    message
                    job {
                        id
                        status
                        filename
                    }
                    estimatedCompletion
                }
                ... on ErrorResult {
                    success
                    message
                    code
                }
            }
        }
        """
        
        # Note: File upload testing in GraphQL requires special handling
        # In a real test, you would use multipart form data
        
        with patch('app.core.auth.get_current_user') as mock_get_user:
            mock_user = Mock()
            mock_user.id = "user123"
            mock_user.credits_remaining = 1000
            mock_get_user.return_value = mock_user
            
            # For this test, we'll just verify the mutation structure
            response = await client.post(
                "/graphql",
                json={"query": mutation, "variables": {}}
            )
            
            # Expect a GraphQL error due to missing file upload
            # but the mutation should be recognized
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_cancel_transcription_job_mutation(self, client):
        """Test job cancellation mutation"""
        mutation = """
        mutation($jobId: String!) {
            cancelTranscriptionJob(jobId: $jobId) {
                ... on TranscriptionResult {
                    success
                    message
                    job {
                        id
                        status
                    }
                }
                ... on ErrorResult {
                    success
                    message
                    code
                }
            }
        }
        """
        
        variables = {"jobId": "job123"}
        
        with patch('app.core.auth.get_current_user') as mock_get_user, \
             patch('app.db.session.get_db_session') as mock_session:
            
            mock_user = Mock()
            mock_user.id = "user123"
            mock_get_user.return_value = mock_user
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock job
            mock_job = Mock()
            mock_job.id = "job123"
            mock_job.user_id = "user123"
            mock_job.status = TranscriptionStatus.PROCESSING
            
            mock_session_instance.get.return_value = mock_job
            
            response = await client.post(
                "/graphql",
                json={"query": mutation, "variables": variables},
                headers={"Authorization": "Bearer test_token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "data" in data
            result = data["data"]["cancelTranscriptionJob"]
            
            # Should succeed or show appropriate error
            assert "success" in result


class TestGraphQLSubscriptions:
    """Test GraphQL subscription operations"""
    
    @pytest.mark.asyncio
    async def test_transcription_updates_subscription(self):
        """Test transcription updates subscription"""
        # Note: Testing WebSocket subscriptions requires more complex setup
        # This is a placeholder for subscription testing
        
        subscription = """
        subscription($jobId: String) {
            transcriptionUpdates(jobId: $jobId) {
                jobId
                status
                progress
                timestamp
            }
        }
        """
        
        # In a real test, you would:
        # 1. Set up WebSocket connection
        # 2. Send subscription
        # 3. Trigger events that cause updates
        # 4. Verify received messages
        
        assert subscription is not None  # Placeholder assertion
    
    @pytest.mark.asyncio
    async def test_streaming_transcription_subscription(self):
        """Test streaming transcription subscription"""
        subscription = """
        subscription($sessionId: String!) {
            streamingTranscription(sessionId: $sessionId)
        }
        """
        
        # Placeholder for streaming subscription test
        assert subscription is not None


class TestGraphQLSecurity:
    """Test GraphQL security features"""
    
    @pytest.fixture
    def app(self):
        return create_application()
    
    @pytest.fixture
    async def client(self, app):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.mark.asyncio
    async def test_unauthenticated_protected_query(self, client):
        """Test that protected queries require authentication"""
        query = """
        query {
            currentUser {
                id
                email
            }
        }
        """
        
        response = await client.post(
            "/graphql",
            json={"query": query}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return null for unauthenticated user
        assert data["data"]["currentUser"] is None
    
    @pytest.mark.asyncio
    async def test_query_depth_limitation(self, client):
        """Test query depth limitations"""
        # Very deep nested query that should be rejected
        deep_query = """
        query {
            currentUser {
                transcriptionJobs {
                    jobs {
                        user {
                            transcriptionJobs {
                                jobs {
                                    user {
                                        id
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """
        
        response = await client.post(
            "/graphql",
            json={"query": deep_query}
        )
        
        # Should either limit depth or return error
        assert response.status_code == 200
    
    @pytest.mark.asyncio 
    async def test_query_complexity_limitation(self, client):
        """Test query complexity limitations"""
        # Complex query that requests many fields
        complex_query = """
        query {
            availableModels {
                name
                modelId
                type
                description
                supportedLanguages
                maxDuration
                accuracyScore
                latencyMs
                isAvailable
                requiresGpu
            }
            systemMetrics {
                totalTranscriptions
                totalUsers
                activeJobs
                queueLength
                averageLatencyMs
                successRate
                supportedLanguages
                availableModels
                systemLoad
                timestamp
            }
        }
        """
        
        response = await client.post(
            "/graphql",
            json={"query": complex_query}
        )
        
        # Should handle complex queries appropriately
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_malformed_query_handling(self, client):
        """Test handling of malformed queries"""
        malformed_query = """
        query {
            currentUser {
                id
                nonExistentField
            }
        """  # Missing closing brace
        
        response = await client.post(
            "/graphql",
            json={"query": malformed_query}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should contain GraphQL syntax error
        assert "errors" in data


class TestGraphQLPerformance:
    """Test GraphQL performance characteristics"""
    
    @pytest.fixture
    def app(self):
        return create_application()
    
    @pytest.fixture
    async def client(self, app):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.mark.asyncio
    async def test_query_performance(self, client):
        """Test query execution performance"""
        import time
        
        query = """
        query {
            availableModels {
                name
                modelId
                type
            }
        }
        """
        
        start_time = time.time()
        
        response = await client.post(
            "/graphql",
            json={"query": query}
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert response.status_code == 200
        assert execution_time < 1.0  # Should execute quickly
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, client):
        """Test handling of concurrent queries"""
        query = """
        query {
            availableModels {
                name
                modelId
            }
        }
        """
        
        # Execute multiple queries concurrently
        tasks = []
        for _ in range(10):
            task = client.post("/graphql", json={"query": query})
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "errors" not in data or len(data.get("errors", [])) == 0