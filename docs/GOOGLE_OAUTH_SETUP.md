# Google OAuth Setup Guide

## 1. Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google+ API:
   - Go to "APIs & Services" → "Library"
   - Search for "Google+ API"
   - Click "Enable"

## 2. Create OAuth Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth client ID"
3. Choose "Web application"
4. Configure:
   - **Name**: VoiceForge OAuth
   - **Authorized JavaScript origins**: 
     - `http://localhost:8000`
     - `https://yourdomain.com` (for production)
   - **Authorized redirect URIs**:
     - `http://localhost:8000/auth/google/callback`
     - `https://yourdomain.com/auth/google/callback` (for production)

## 3. Add to Environment Variables

Add these to your `.env` file:

```env
# Google OAuth Configuration
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
```

## 4. Install Dependencies

```bash
pip install authlib httpx
```

## 5. Test the Integration

1. Start the server: `python src/app/main_api_service.py`
2. Go to: `http://localhost:8000/developer`
3. Click "Continue with Google"
4. Complete OAuth flow
5. Check if user is created in database

## Features Added

✅ **Google OAuth Login/Signup**
- One-click registration and login
- Automatic user creation in database
- Profile picture and email verification
- Seamless integration with existing system

✅ **Enhanced UI/UX**
- Modern landing page with animations
- Professional documentation site
- Google OAuth buttons with proper styling
- Responsive design for all devices

✅ **Better than Deepgram**
- More modern design
- Better onboarding flow
- One-click Google authentication
- Interactive documentation
- Animated hero section
- Better developer experience

## Next Steps

1. Set up email verification system
2. Add Stripe billing integration
3. Create API playground
4. Add real-time analytics dashboard