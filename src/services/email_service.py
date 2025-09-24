"""
VoiceForge Email Service
Handles all email communications including welcome emails, password resets, and notifications
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import logging
from jinja2 import Template
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logger = logging.getLogger(__name__)

class EmailService:
    """Email service for sending transactional emails"""
    
    def __init__(self):
        """Initialize email service with SMTP settings"""
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", "noreply@voiceforge.ai")
        self.from_name = os.getenv("FROM_NAME", "VoiceForge")
        self.use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        
        # Thread pool for async email sending
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Email templates
        self.templates = self._load_templates()
        
        # Verify configuration
        if not self.smtp_user or not self.smtp_password:
            logger.warning("⚠️ Email service not configured. Set SMTP_USER and SMTP_PASSWORD in .env")
    
    def _load_templates(self) -> Dict[str, Template]:
        """Load email templates"""
        return {
            "welcome": Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; }
        .content { background: white; padding: 30px; border: 1px solid #e1e1e1; border-radius: 0 0 10px 10px; }
        .button { display: inline-block; padding: 12px 24px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }
        .api-key { background: #f5f5f5; padding: 15px; border-radius: 5px; font-family: monospace; word-break: break-all; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #e1e1e1; color: #666; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Welcome to VoiceForge! 🎉</h1>
        </div>
        <div class="content">
            <h2>Hi {{ full_name }},</h2>
            <p>Thank you for signing up for VoiceForge! Your account has been successfully created.</p>
            
            <h3>Your API Key:</h3>
            <div class="api-key">{{ api_key }}</div>
            <p><strong>Important:</strong> Keep this API key secure. You can regenerate it anytime from your dashboard.</p>
            
            <h3>Quick Start:</h3>
            <ol>
                <li>Install our Python SDK: <code>pip install voiceforge</code></li>
                <li>Use your API key to authenticate</li>
                <li>Start transcribing audio files!</li>
            </ol>
            
            <a href="{{ dashboard_url }}" class="button">Go to Dashboard</a>
            
            <h3>Your Plan: {{ plan_name }}</h3>
            <p>You have {{ monthly_limit }} transcription minutes per month.</p>
            
            <div class="footer">
                <p>Need help? Check out our <a href="{{ docs_url }}">documentation</a> or reply to this email.</p>
                <p>© {{ current_year }} VoiceForge. All rights reserved.</p>
            </div>
        </div>
    </div>
</body>
</html>
            """),
            
            "password_reset": Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; }
        .content { background: white; padding: 30px; border: 1px solid #e1e1e1; border-radius: 0 0 10px 10px; }
        .button { display: inline-block; padding: 12px 24px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #e1e1e1; color: #666; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Password Reset Request</h1>
        </div>
        <div class="content">
            <h2>Hi {{ full_name }},</h2>
            <p>We received a request to reset your password. Click the button below to create a new password:</p>
            
            <a href="{{ reset_url }}" class="button">Reset Password</a>
            
            <p>Or copy and paste this link into your browser:</p>
            <p style="word-break: break-all; color: #667eea;">{{ reset_url }}</p>
            
            <p><strong>This link will expire in 1 hour.</strong></p>
            
            <p>If you didn't request a password reset, please ignore this email or contact support if you have concerns.</p>
            
            <div class="footer">
                <p>© {{ current_year }} VoiceForge. All rights reserved.</p>
            </div>
        </div>
    </div>
</body>
</html>
            """),
            
            "usage_alert": Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; }
        .content { background: white; padding: 30px; border: 1px solid #e1e1e1; border-radius: 0 0 10px 10px; }
        .usage-bar { background: #f5f5f5; border-radius: 10px; overflow: hidden; height: 30px; margin: 20px 0; }
        .usage-fill { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 100%; display: flex; align-items: center; justify-content: center; color: white; }
        .button { display: inline-block; padding: 12px 24px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #e1e1e1; color: #666; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Usage Alert ⚠️</h1>
        </div>
        <div class="content">
            <h2>Hi {{ full_name }},</h2>
            <p>You've used <strong>{{ usage_percentage }}%</strong> of your monthly transcription limit.</p>
            
            <div class="usage-bar">
                <div class="usage-fill" style="width: {{ usage_percentage }}%">
                    {{ usage_percentage }}%
                </div>
            </div>
            
            <p><strong>Usage Details:</strong></p>
            <ul>
                <li>Used: {{ minutes_used }} minutes</li>
                <li>Limit: {{ monthly_limit }} minutes</li>
                <li>Remaining: {{ minutes_remaining }} minutes</li>
            </ul>
            
            <p>Consider upgrading your plan to avoid service interruption.</p>
            
            <a href="{{ upgrade_url }}" class="button">Upgrade Plan</a>
            
            <div class="footer">
                <p>© {{ current_year }} VoiceForge. All rights reserved.</p>
            </div>
        </div>
    </div>
</body>
</html>
            """),
            
            "api_key_regenerated": Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; }
        .content { background: white; padding: 30px; border: 1px solid #e1e1e1; border-radius: 0 0 10px 10px; }
        .api-key { background: #f5f5f5; padding: 15px; border-radius: 5px; font-family: monospace; word-break: break-all; }
        .warning { background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #e1e1e1; color: #666; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>API Key Regenerated 🔑</h1>
        </div>
        <div class="content">
            <h2>Hi {{ full_name }},</h2>
            <p>Your API key has been successfully regenerated.</p>
            
            <h3>New API Key:</h3>
            <div class="api-key">{{ api_key }}</div>
            
            <div class="warning">
                <strong>⚠️ Important:</strong> Your old API key is now invalid. Update your applications with the new key.
            </div>
            
            <p><strong>Security Information:</strong></p>
            <ul>
                <li>Regenerated at: {{ regenerated_at }}</li>
                <li>IP Address: {{ ip_address }}</li>
                <li>User Agent: {{ user_agent }}</li>
            </ul>
            
            <p>If you didn't request this change, please contact support immediately.</p>
            
            <div class="footer">
                <p>© {{ current_year }} VoiceForge. All rights reserved.</p>
            </div>
        </div>
    </div>
</body>
</html>
            """)
        }
    
    def _send_email_sync(self, to_email: str, subject: str, html_content: str, 
                         text_content: Optional[str] = None) -> bool:
        """Synchronous email sending"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = to_email
            
            # Add text and HTML parts
            if text_content:
                text_part = MIMEText(text_content, 'plain')
                msg.attach(text_part)
            
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"✅ Email sent to {to_email}: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send email to {to_email}: {e}")
            return False
    
    async def send_email(self, to_email: str, subject: str, html_content: str,
                        text_content: Optional[str] = None) -> bool:
        """Async wrapper for sending email"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._send_email_sync,
            to_email, subject, html_content, text_content
        )
    
    async def send_welcome_email(self, user_data: Dict) -> bool:
        """Send welcome email to new user"""
        subject = "Welcome to VoiceForge! 🎉"
        
        context = {
            "full_name": user_data.get("full_name", "Developer"),
            "api_key": user_data.get("api_key", ""),
            "plan_name": user_data.get("plan_name", "Free"),
            "monthly_limit": user_data.get("monthly_limit", 1000),
            "dashboard_url": f"{os.getenv('APP_URL', 'http://localhost:8000')}/developer",
            "docs_url": f"{os.getenv('APP_URL', 'http://localhost:8000')}/docs",
            "current_year": datetime.now().year
        }
        
        html_content = self.templates["welcome"].render(**context)
        
        # Simple text version
        text_content = f"""
Welcome to VoiceForge!

Hi {context['full_name']},

Thank you for signing up! Your account has been created successfully.

Your API Key: {context['api_key']}

Quick Start:
1. Install our Python SDK: pip install voiceforge
2. Use your API key to authenticate
3. Start transcribing!

Dashboard: {context['dashboard_url']}
Documentation: {context['docs_url']}

Best regards,
The VoiceForge Team
        """
        
        return await self.send_email(
            user_data["email"],
            subject,
            html_content,
            text_content
        )
    
    async def send_password_reset_email(self, user_data: Dict, reset_token: str) -> bool:
        """Send password reset email"""
        subject = "Password Reset Request - VoiceForge"
        
        reset_url = f"{os.getenv('APP_URL', 'http://localhost:8000')}/reset-password?token={reset_token}"
        
        context = {
            "full_name": user_data.get("full_name", "User"),
            "reset_url": reset_url,
            "current_year": datetime.now().year
        }
        
        html_content = self.templates["password_reset"].render(**context)
        
        text_content = f"""
Password Reset Request

Hi {context['full_name']},

We received a request to reset your password. Visit this link to create a new password:

{reset_url}

This link will expire in 1 hour.

If you didn't request this, please ignore this email.

Best regards,
The VoiceForge Team
        """
        
        return await self.send_email(
            user_data["email"],
            subject,
            html_content,
            text_content
        )
    
    async def send_usage_alert(self, user_data: Dict, usage_data: Dict) -> bool:
        """Send usage alert when user reaches certain thresholds"""
        subject = f"Usage Alert - {usage_data['usage_percentage']}% of limit reached"
        
        context = {
            "full_name": user_data.get("full_name", "User"),
            "usage_percentage": usage_data["usage_percentage"],
            "minutes_used": usage_data["minutes_used"],
            "monthly_limit": usage_data["monthly_limit"],
            "minutes_remaining": usage_data["minutes_remaining"],
            "upgrade_url": f"{os.getenv('APP_URL', 'http://localhost:8000')}/pricing",
            "current_year": datetime.now().year
        }
        
        html_content = self.templates["usage_alert"].render(**context)
        
        text_content = f"""
Usage Alert

Hi {context['full_name']},

You've used {context['usage_percentage']}% of your monthly transcription limit.

Usage Details:
- Used: {context['minutes_used']} minutes
- Limit: {context['monthly_limit']} minutes  
- Remaining: {context['minutes_remaining']} minutes

Consider upgrading your plan: {context['upgrade_url']}

Best regards,
The VoiceForge Team
        """
        
        return await self.send_email(
            user_data["email"],
            subject,
            html_content,
            text_content
        )
    
    async def send_api_key_regenerated_email(self, user_data: Dict, 
                                            new_api_key: str,
                                            request_info: Dict) -> bool:
        """Send notification when API key is regenerated"""
        subject = "API Key Regenerated - VoiceForge"
        
        context = {
            "full_name": user_data.get("full_name", "User"),
            "api_key": new_api_key,
            "regenerated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "ip_address": request_info.get("ip_address", "Unknown"),
            "user_agent": request_info.get("user_agent", "Unknown"),
            "current_year": datetime.now().year
        }
        
        html_content = self.templates["api_key_regenerated"].render(**context)
        
        text_content = f"""
API Key Regenerated

Hi {context['full_name']},

Your API key has been regenerated.

New API Key: {context['api_key']}

Security Information:
- Regenerated at: {context['regenerated_at']}
- IP Address: {context['ip_address']}
- User Agent: {context['user_agent']}

If you didn't request this change, contact support immediately.

Best regards,
The VoiceForge Team
        """
        
        return await self.send_email(
            user_data["email"],
            subject,
            html_content,
            text_content
        )
    
    async def send_batch_notification_emails(self, email_list: List[Dict]) -> Dict:
        """Send multiple emails in batch"""
        results = {"sent": 0, "failed": 0, "errors": []}
        
        tasks = []
        for email_data in email_list:
            task = self.send_email(
                email_data["to_email"],
                email_data["subject"],
                email_data["html_content"],
                email_data.get("text_content")
            )
            tasks.append(task)
        
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results_list):
            if isinstance(result, Exception):
                results["failed"] += 1
                results["errors"].append({
                    "email": email_list[i]["to_email"],
                    "error": str(result)
                })
            elif result:
                results["sent"] += 1
            else:
                results["failed"] += 1
        
        return results

# Singleton instance
email_service = EmailService()