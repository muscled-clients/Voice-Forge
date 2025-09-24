import pytest
from httpx import AsyncClient
import uuid


class TestAuthentication:
    """Test authentication endpoints"""
    
    @pytest.mark.asyncio
    async def test_register_user(self, client: AsyncClient, test_user_data):
        """Test user registration"""
        response = await client.post(
            "/api/v1/auth/register",
            json=test_user_data
        )
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["email"] == test_user_data["email"]
        assert data["full_name"] == test_user_data["full_name"]
        assert "api_key" in data
        assert data["tier"] == "free"
        assert data["credits_remaining"] == 1000
        assert data["is_active"] is True
    
    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, client: AsyncClient, test_user_data):
        """Test registration with duplicate email"""
        # Register first user
        await client.post("/api/v1/auth/register", json=test_user_data)
        
        # Try to register with same email
        response = await client.post("/api/v1/auth/register", json=test_user_data)
        
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_login_success(self, client: AsyncClient, test_user):
        """Test successful login"""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user.email,
                "password": "testpassword123"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        assert "refresh_token" in data
    
    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, client: AsyncClient, test_user):
        """Test login with invalid credentials"""
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user.email,
                "password": "wrongpassword"
            }
        )
        
        assert response.status_code == 401
        assert "Invalid email or password" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_get_current_user(self, authenticated_client: AsyncClient, test_user):
        """Test getting current user info"""
        response = await authenticated_client.get("/api/v1/auth/me")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["email"] == test_user.email
        assert data["id"] == test_user.id
        assert "api_key" in data
    
    @pytest.mark.asyncio
    async def test_unauthorized_access(self, client: AsyncClient):
        """Test accessing protected endpoint without auth"""
        response = await client.get("/api/v1/auth/me")
        
        assert response.status_code == 403
    
    @pytest.mark.asyncio
    async def test_invalid_token(self, client: AsyncClient):
        """Test accessing with invalid token"""
        client.headers.update({"Authorization": "Bearer invalid-token"})
        response = await client.get("/api/v1/auth/me")
        
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_logout(self, authenticated_client: AsyncClient):
        """Test logout"""
        response = await authenticated_client.post("/api/v1/auth/logout")
        
        assert response.status_code == 200
        assert "Successfully logged out" in response.json()["message"]


class TestAPIKeyAuthentication:
    """Test API key authentication"""
    
    @pytest.mark.asyncio
    async def test_api_key_access(self, client: AsyncClient, test_user):
        """Test accessing with API key"""
        client.headers.update({"Authorization": f"Bearer {test_user.api_key}"})
        response = await client.get("/api/v1/auth/me")
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user.email
    
    @pytest.mark.asyncio
    async def test_invalid_api_key(self, client: AsyncClient):
        """Test accessing with invalid API key"""
        client.headers.update({"Authorization": "Bearer vf_invalid_api_key"})
        response = await client.get("/api/v1/auth/me")
        
        assert response.status_code == 401