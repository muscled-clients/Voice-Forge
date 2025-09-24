"""
Google OAuth authentication module
"""
import os
import json
import secrets
from typing import Optional
from datetime import datetime, timedelta
from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse
from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.common.urls import add_params_to_uri
import jwt
import logging
import ssl
import httpx

logger = logging.getLogger(__name__)

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

# OAuth URLs
GOOGLE_AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

class GoogleAuth:
    """Handle Google OAuth authentication"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
        
    async def login(self, request: Request):
        """Initiate Google OAuth login"""
        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            raise HTTPException(
                status_code=501,
                detail="Google OAuth not configured. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env"
            )
        
        # Build authorization URL (simplified without state for now)
        params = {
            'client_id': GOOGLE_CLIENT_ID,
            'redirect_uri': GOOGLE_REDIRECT_URI,
            'scope': 'openid email profile',
            'response_type': 'code',
            'access_type': 'offline',
            'prompt': 'consent'
        }
        
        auth_url = add_params_to_uri(GOOGLE_AUTHORIZATION_URL, params)
        return RedirectResponse(url=auth_url)
    
    async def callback(self, request: Request):
        """Handle Google OAuth callback"""
        try:
            # Get authorization code from query parameters
            code = request.query_params.get('code')
            if not code:
                raise HTTPException(status_code=400, detail="Authorization code not found")
            
            # Exchange code for token using httpx directly
            async with httpx.AsyncClient(verify=False) as client:
                # Get access token
                token_data = {
                    'client_id': GOOGLE_CLIENT_ID,
                    'client_secret': GOOGLE_CLIENT_SECRET,
                    'code': code,
                    'grant_type': 'authorization_code',
                    'redirect_uri': GOOGLE_REDIRECT_URI
                }
                
                token_response = await client.post(GOOGLE_TOKEN_URL, data=token_data)
                token_response.raise_for_status()
                token_json = token_response.json()
                
                access_token = token_json.get('access_token')
                if not access_token:
                    raise HTTPException(status_code=400, detail="Failed to get access token")
                
                # Get user info
                headers = {'Authorization': f'Bearer {access_token}'}
                user_response = await client.get(GOOGLE_USERINFO_URL, headers=headers)
                user_response.raise_for_status()
                user_info = user_response.json()
            
            # Extract user data
            google_id = user_info.get('id')
            email = user_info.get('email')
            name = user_info.get('name', '')
            picture = user_info.get('picture', '')
            email_verified = user_info.get('verified_email', False)
            
            if not email:
                raise HTTPException(status_code=400, detail="Email not provided by Google")
            
            # Check if user exists
            user = self.get_or_create_user(
                google_id=google_id,
                email=email,
                name=name,
                picture=picture,
                email_verified=email_verified
            )
            
            # Generate JWT token
            access_token = self.create_access_token(
                user_id=user['id'],
                email=user['email']
            )
            
            # Return redirect with token
            return RedirectResponse(
                url=f"/developer?token={access_token}",
                status_code=302
            )
            
        except Exception as e:
            logger.error(f"Google OAuth callback error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    def get_or_create_user(self, google_id: str, email: str, name: str, picture: str, email_verified: bool):
        """Get existing user or create new one from Google OAuth"""
        from main_api_service import execute_db_query, generate_api_key
        import uuid
        
        # Check if user exists
        existing = execute_db_query(
            "SELECT * FROM voiceforge.users WHERE email = %s",
            (email,),
            fetch_one=True
        )
        
        if existing:
            # Update Google info
            metadata = existing.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            metadata.update({
                'google_id': google_id,
                'picture': picture,
                'email_verified': email_verified,
                'last_login': datetime.now().isoformat(),
                'auth_provider': 'google'
            })
            
            execute_db_query(
                "UPDATE voiceforge.users SET metadata = %s WHERE id = %s",
                (json.dumps(metadata), existing['id'])
            )
            
            return existing
        
        # Create new user
        user_id = str(uuid.uuid4())
        api_key = generate_api_key(user_id, email)
        
        # Get free plan ID
        free_plan = execute_db_query(
            "SELECT id FROM voiceforge.plans WHERE name = 'free'",
            fetch_one=True
        )
        plan_id = free_plan['id'] if free_plan else None
        
        # Create user
        metadata = {
            'google_id': google_id,
            'picture': picture,
            'email_verified': email_verified,
            'registration_date': datetime.now().isoformat(),
            'auth_provider': 'google'
        }
        
        query = """
        INSERT INTO voiceforge.users (id, email, full_name, api_key, plan_id, metadata)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id, email, full_name, api_key, created_at
        """
        
        result = execute_db_query(
            query,
            (user_id, email, name, api_key, plan_id, json.dumps(metadata)),
            fetch_one=True
        )
        
        logger.info(f"✅ New user registered via Google OAuth: {email}")
        return result
    
    def create_access_token(self, user_id: str, email: str, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = {
            "sub": user_id,
            "email": email,
            "auth_provider": "google"
        }
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm="HS256")
        return encoded_jwt