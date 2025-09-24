-- Add subscription plans and user subscriptions tables

-- Subscription plans
CREATE TABLE IF NOT EXISTS voiceforge.plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(50) NOT NULL UNIQUE,
    display_name VARCHAR(100),
    monthly_limit INTEGER,
    price_monthly DECIMAL(10,2),
    price_yearly DECIMAL(10,2),
    features JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User subscriptions
CREATE TABLE IF NOT EXISTS voiceforge.user_subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE,
    plan_id UUID REFERENCES voiceforge.plans(id),
    status VARCHAR(50) DEFAULT 'active', -- active, cancelled, expired, trial
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Insert default plans
INSERT INTO voiceforge.plans (name, display_name, monthly_limit, price_monthly, price_yearly, features) VALUES
('free', 'Free Plan', 1000, 0.00, 0.00, '{"max_file_size_mb": 10, "rate_limit_per_minute": 10, "support": "community", "languages": "all", "models": ["whisper-tiny"]}'),
('starter', 'Starter Plan', 10000, 29.99, 299.99, '{"max_file_size_mb": 50, "rate_limit_per_minute": 50, "support": "email", "languages": "all", "models": ["whisper-tiny", "whisper-base"]}'),
('professional', 'Professional Plan', 50000, 99.99, 999.99, '{"max_file_size_mb": 100, "rate_limit_per_minute": 100, "support": "priority", "languages": "all", "models": ["whisper-tiny", "whisper-base", "whisper-small"]}'),
('enterprise', 'Enterprise Plan', -1, 499.99, 4999.99, '{"max_file_size_mb": 500, "rate_limit_per_minute": 1000, "support": "dedicated", "languages": "all", "models": ["all"], "custom_deployment": true}')
ON CONFLICT (name) DO NOTHING;

-- Add plan_id to users table if not exists
ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS plan_id UUID REFERENCES voiceforge.plans(id);

-- Set all existing users to free plan
UPDATE voiceforge.users 
SET plan_id = (SELECT id FROM voiceforge.plans WHERE name = 'free')
WHERE plan_id IS NULL;

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_user_subscriptions_user_id ON voiceforge.user_subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_subscriptions_status ON voiceforge.user_subscriptions(status);

-- Add view for user plan details
CREATE OR REPLACE VIEW voiceforge.user_plan_details AS
SELECT 
    u.id as user_id,
    u.email,
    u.full_name,
    p.name as plan_name,
    p.display_name as plan_display_name,
    p.monthly_limit,
    p.features,
    us.status as subscription_status,
    us.expires_at
FROM voiceforge.users u
LEFT JOIN voiceforge.plans p ON u.plan_id = p.id
LEFT JOIN voiceforge.user_subscriptions us ON u.id = us.user_id AND us.status = 'active';