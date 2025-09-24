"""
Usage Monitoring Service
Tracks API usage and sends alerts at defined thresholds
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from src.database.connection import get_db_session
from src.database.models import User, Transcription, UsageRecord
from src.services.email_service import email_service
from sqlalchemy import func, and_

logger = logging.getLogger(__name__)

class UsageMonitor:
    """Monitor and alert on API usage"""
    
    # Alert thresholds (percentage of monthly limit)
    ALERT_THRESHOLDS = [50, 75, 90, 95, 100]
    
    def __init__(self):
        """Initialize usage monitor"""
        self.alerts_sent = {}  # Track which alerts have been sent
    
    async def check_user_usage(self, user_id: str) -> Dict:
        """Check a user's current usage statistics"""
        try:
            with get_db_session() as session:
                # Get user and plan info
                user = session.query(User).filter(User.id == user_id).first()
                if not user or not user.plan:
                    return None
                
                # Calculate current month's usage
                start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                
                # Get total minutes used this month
                total_seconds = session.query(
                    func.coalesce(func.sum(Transcription.duration_seconds), 0)
                ).filter(
                    and_(
                        Transcription.user_id == user_id,
                        Transcription.created_at >= start_of_month,
                        Transcription.status == 'COMPLETED'
                    )
                ).scalar()
                
                minutes_used = round(total_seconds / 60, 2)
                monthly_limit = user.plan.monthly_limit if user.plan else 1000
                
                # Calculate percentage
                usage_percentage = round((minutes_used / monthly_limit) * 100, 2) if monthly_limit > 0 else 100
                minutes_remaining = max(0, monthly_limit - minutes_used)
                
                usage_data = {
                    "user_id": user_id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "minutes_used": minutes_used,
                    "monthly_limit": monthly_limit,
                    "minutes_remaining": minutes_remaining,
                    "usage_percentage": usage_percentage,
                    "plan_name": user.plan.display_name if user.plan else "Free"
                }
                
                # Check if we need to send alerts
                await self._check_and_send_alerts(usage_data)
                
                return usage_data
                
        except Exception as e:
            logger.error(f"Error checking usage for user {user_id}: {e}")
            return None
    
    async def _check_and_send_alerts(self, usage_data: Dict):
        """Check if alerts need to be sent based on usage"""
        user_id = usage_data["user_id"]
        usage_percentage = usage_data["usage_percentage"]
        
        # Initialize user alert tracking if needed
        if user_id not in self.alerts_sent:
            self.alerts_sent[user_id] = {
                "month": datetime.now().month,
                "thresholds": []
            }
        
        # Reset alerts for new month
        if self.alerts_sent[user_id]["month"] != datetime.now().month:
            self.alerts_sent[user_id] = {
                "month": datetime.now().month,
                "thresholds": []
            }
        
        # Check each threshold
        for threshold in self.ALERT_THRESHOLDS:
            if (usage_percentage >= threshold and 
                threshold not in self.alerts_sent[user_id]["thresholds"]):
                
                # Send alert
                await self._send_usage_alert(usage_data, threshold)
                
                # Mark as sent
                self.alerts_sent[user_id]["thresholds"].append(threshold)
    
    async def _send_usage_alert(self, usage_data: Dict, threshold: int):
        """Send usage alert email"""
        try:
            # Customize message based on threshold
            if threshold >= 100:
                subject_prefix = "⛔ Limit Reached"
                urgency = "You have reached your monthly limit."
            elif threshold >= 90:
                subject_prefix = "⚠️ Critical Usage"
                urgency = "You're approaching your monthly limit."
            elif threshold >= 75:
                subject_prefix = "📊 High Usage"
                urgency = "You're using your allocation quickly."
            else:
                subject_prefix = "📈 Usage Update"
                urgency = "You're halfway through your monthly allocation."
            
            # Send email
            await email_service.send_usage_alert(
                {
                    "email": usage_data["email"],
                    "full_name": usage_data["full_name"]
                },
                {
                    "usage_percentage": threshold,
                    "minutes_used": usage_data["minutes_used"],
                    "monthly_limit": usage_data["monthly_limit"],
                    "minutes_remaining": usage_data["minutes_remaining"],
                    "urgency": urgency
                }
            )
            
            logger.info(f"Sent {threshold}% usage alert to {usage_data['email']}")
            
        except Exception as e:
            logger.error(f"Failed to send usage alert: {e}")
    
    async def check_all_users(self):
        """Check usage for all active users"""
        try:
            with get_db_session() as session:
                # Get all active users
                active_users = session.query(User).filter(
                    User.is_active == True
                ).all()
                
                logger.info(f"Checking usage for {len(active_users)} active users")
                
                # Check each user
                for user in active_users:
                    await self.check_user_usage(str(user.id))
                    await asyncio.sleep(0.1)  # Rate limit
                
                logger.info("Usage check completed for all users")
                
        except Exception as e:
            logger.error(f"Error checking all users: {e}")
    
    async def run_monitoring_loop(self, interval_minutes: int = 60):
        """Run continuous monitoring loop"""
        logger.info(f"Starting usage monitoring loop (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                await self.check_all_users()
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
            
            # Wait for next check
            await asyncio.sleep(interval_minutes * 60)
    
    def get_usage_report(self, user_id: str, days: int = 30) -> Dict:
        """Generate detailed usage report for a user"""
        try:
            with get_db_session() as session:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # Daily usage over period
                daily_usage = session.query(
                    func.date(Transcription.created_at).label('date'),
                    func.count(Transcription.id).label('count'),
                    func.sum(Transcription.duration_seconds).label('total_seconds')
                ).filter(
                    and_(
                        Transcription.user_id == user_id,
                        Transcription.created_at >= cutoff_date,
                        Transcription.status == 'COMPLETED'
                    )
                ).group_by(func.date(Transcription.created_at)).all()
                
                # Format report
                report = {
                    "user_id": user_id,
                    "period_days": days,
                    "daily_usage": [
                        {
                            "date": str(day.date),
                            "transcriptions": day.count,
                            "minutes": round(day.total_seconds / 60, 2) if day.total_seconds else 0
                        }
                        for day in daily_usage
                    ],
                    "total_transcriptions": sum(d.count for d in daily_usage),
                    "total_minutes": sum(d.total_seconds / 60 if d.total_seconds else 0 for d in daily_usage)
                }
                
                return report
                
        except Exception as e:
            logger.error(f"Error generating usage report: {e}")
            return None

# Singleton instance
usage_monitor = UsageMonitor()