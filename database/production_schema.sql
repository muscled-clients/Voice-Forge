-- VoiceForge Production Database Schema
-- PostgreSQL initialization script for production deployment
-- Date: 2025-08-25
-- Version: 1.0.0

-- Create database (run as superuser)
-- CREATE DATABASE voiceforge;
-- CREATE USER voiceforge WITH ENCRYPTED PASSWORD 'your_secure_password_here';
-- GRANT ALL PRIVILEGES ON DATABASE voiceforge TO voiceforge;

-- Connect to database
\c voiceforge;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;

-- Create main schema (aligning with code expectations)
CREATE SCHEMA IF NOT EXISTS voiceforge;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path to include voiceforge schema
ALTER DATABASE voiceforge SET search_path = voiceforge, public;

-- =======================
-- CORE BUSINESS TABLES
-- =======================

-- Plans table (subscription tiers)
CREATE TABLE voiceforge.plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    description TEXT,
    monthly_limit INTEGER NOT NULL,  -- Minutes per month
    price_monthly DECIMAL(10, 2) NOT NULL,
    price_yearly DECIMAL(10, 2),
    features JSONB DEFAULT '[]'::jsonb,
    is_active BOOLEAN DEFAULT true,
    stripe_price_id VARCHAR(255),
    stripe_product_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Users table (primary user accounts)
CREATE TABLE voiceforge.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    company VARCHAR(255),
    api_key VARCHAR(255) UNIQUE,
    role VARCHAR(50) DEFAULT 'user',
    plan_id UUID REFERENCES voiceforge.plans(id),
    is_active BOOLEAN DEFAULT true,
    email_verified BOOLEAN DEFAULT false,
    email_verification_token VARCHAR(255),
    password_reset_token VARCHAR(255),
    password_reset_expires TIMESTAMP WITH TIME ZONE,
    stripe_customer_id VARCHAR(255),
    trial_ends_at TIMESTAMP WITH TIME ZONE,
    subscription_status VARCHAR(50) DEFAULT 'inactive',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- User subscriptions (for detailed subscription tracking)
CREATE TABLE voiceforge.user_subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE,
    plan_id UUID REFERENCES voiceforge.plans(id),
    stripe_subscription_id VARCHAR(255) UNIQUE,
    status VARCHAR(50) NOT NULL, -- active, canceled, past_due, unpaid
    current_period_start TIMESTAMP WITH TIME ZONE,
    current_period_end TIMESTAMP WITH TIME ZONE,
    cancel_at_period_end BOOLEAN DEFAULT false,
    canceled_at TIMESTAMP WITH TIME ZONE,
    trial_start TIMESTAMP WITH TIME ZONE,
    trial_end TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Developers table (API key management)
CREATE TABLE voiceforge.developers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    company VARCHAR(255),
    api_keys_created INTEGER DEFAULT 0,
    last_api_key_created TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- API Keys table (enhanced security)
CREATE TABLE voiceforge.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    key_prefix VARCHAR(20) NOT NULL, -- For display purposes (vf_xxxx)
    name VARCHAR(255),
    last_used TIMESTAMP WITH TIME ZONE,
    usage_count INTEGER DEFAULT 0,
    rate_limit INTEGER DEFAULT 100,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    scopes JSONB DEFAULT '["transcribe:read", "transcribe:write"]'::jsonb,
    ip_whitelist INET[],
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Transcriptions table (enhanced with Phase 3 features)
CREATE TABLE voiceforge.transcriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE,
    session_id VARCHAR(255), -- For WebSocket sessions
    file_name VARCHAR(255),
    file_path TEXT, -- S3 path or local path
    file_size BIGINT,
    file_hash VARCHAR(64),
    mime_type VARCHAR(100),
    duration_seconds FLOAT,
    language VARCHAR(10),
    detected_languages JSONB, -- Multiple language detection results
    transcription_text TEXT,
    segments JSONB, -- Word-level timestamps
    confidence_score FLOAT,
    word_count INTEGER,
    
    -- Phase 3: Advanced features
    speakers JSONB, -- Speaker diarization results
    speaker_count INTEGER,
    noise_reduction_applied BOOLEAN DEFAULT false,
    noise_level_db FLOAT,
    
    processing_time_ms INTEGER,
    model_used VARCHAR(50),
    whisper_version VARCHAR(20),
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    
    -- Billing
    cost_cents INTEGER DEFAULT 0,
    billing_duration_seconds FLOAT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Convert to TimescaleDB hypertable for performance
SELECT create_hypertable('voiceforge.transcriptions', 'created_at', if_not_exists => TRUE);

-- =======================
-- STREAMING & WEBSOCKETS
-- =======================

-- WebSocket Sessions
CREATE TABLE voiceforge.websocket_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE,
    connection_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'active',
    
    -- Audio processing stats
    duration_seconds FLOAT,
    total_audio_seconds FLOAT,
    bytes_processed BIGINT DEFAULT 0,
    transcription_count INTEGER DEFAULT 0,
    
    -- Phase 3 configuration
    vad_enabled BOOLEAN DEFAULT true,
    speaker_diarization BOOLEAN DEFAULT false,
    language_detection BOOLEAN DEFAULT false,
    noise_reduction BOOLEAN DEFAULT false,
    
    -- Connection details
    ip_address INET,
    user_agent TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    metadata JSONB DEFAULT '{}'::jsonb
);

-- =======================
-- BILLING & PAYMENTS
-- =======================

-- Usage tracking (for billing)
CREATE TABLE voiceforge.usage_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE,
    transcription_id UUID REFERENCES voiceforge.transcriptions(id) ON DELETE SET NULL,
    subscription_id UUID REFERENCES voiceforge.user_subscriptions(id) ON DELETE SET NULL,
    
    -- Usage metrics
    usage_type VARCHAR(50) NOT NULL, -- 'transcription', 'streaming', 'api_call'
    quantity FLOAT NOT NULL, -- minutes, requests, etc.
    unit VARCHAR(20) NOT NULL, -- 'minutes', 'requests', 'mb'
    unit_cost_cents INTEGER,
    total_cost_cents INTEGER,
    
    -- Billing period
    billing_period_start DATE,
    billing_period_end DATE,
    billed BOOLEAN DEFAULT false,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Invoices
CREATE TABLE voiceforge.invoices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE,
    subscription_id UUID REFERENCES voiceforge.user_subscriptions(id),
    stripe_invoice_id VARCHAR(255) UNIQUE,
    
    -- Invoice details
    invoice_number VARCHAR(50) UNIQUE,
    subtotal_cents INTEGER NOT NULL,
    tax_cents INTEGER DEFAULT 0,
    total_cents INTEGER NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    
    -- Status
    status VARCHAR(50) NOT NULL, -- draft, open, paid, void, uncollectible
    paid BOOLEAN DEFAULT false,
    
    -- Dates
    billing_period_start DATE,
    billing_period_end DATE,
    issued_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    due_date DATE,
    paid_at TIMESTAMP WITH TIME ZONE,
    
    -- URLs
    invoice_pdf_url TEXT,
    hosted_invoice_url TEXT,
    
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Payment methods
CREATE TABLE voiceforge.payment_methods (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE,
    stripe_payment_method_id VARCHAR(255) UNIQUE NOT NULL,
    
    -- Card details (masked)
    type VARCHAR(20), -- card, bank_account
    brand VARCHAR(20), -- visa, mastercard, etc.
    last4 VARCHAR(4),
    exp_month INTEGER,
    exp_year INTEGER,
    
    is_default BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    metadata JSONB DEFAULT '{}'::jsonb
);

-- =======================
-- BATCH PROCESSING
-- =======================

-- Batch Jobs
CREATE TABLE voiceforge.batch_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE,
    job_name VARCHAR(255),
    job_type VARCHAR(50) DEFAULT 'transcription',
    
    -- Status
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    
    -- Progress
    total_files INTEGER,
    processed_files INTEGER DEFAULT 0,
    failed_files INTEGER DEFAULT 0,
    success_files INTEGER DEFAULT 0,
    
    -- Configuration
    callback_url TEXT,
    webhook_events JSONB DEFAULT '[]'::jsonb,
    processing_options JSONB DEFAULT '{}'::jsonb,
    
    -- Results
    results JSONB,
    error_log JSONB,
    
    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    processing_time_seconds FLOAT,
    
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Batch Job Files
CREATE TABLE voiceforge.batch_job_files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    batch_job_id UUID REFERENCES voiceforge.batch_jobs(id) ON DELETE CASCADE,
    transcription_id UUID REFERENCES voiceforge.transcriptions(id) ON DELETE CASCADE,
    
    file_name VARCHAR(255) NOT NULL,
    file_size BIGINT,
    file_path TEXT,
    processing_order INTEGER,
    
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    processing_time_ms INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =======================
-- ANALYTICS & MONITORING
-- =======================

-- Usage Analytics (TimescaleDB optimized)
CREATE TABLE analytics.usage_metrics (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    user_id UUID,
    endpoint VARCHAR(255),
    method VARCHAR(10),
    status_code INTEGER,
    response_time_ms INTEGER,
    
    -- Audio metrics
    audio_duration_seconds FLOAT,
    file_size_bytes BIGINT,
    model_used VARCHAR(50),
    language VARCHAR(10),
    
    -- Connection details
    ip_address INET,
    user_agent TEXT,
    api_key_id UUID,
    session_id VARCHAR(255),
    
    -- Error tracking
    error_type VARCHAR(100),
    error_message TEXT,
    
    metadata JSONB DEFAULT '{}'::jsonb
);

SELECT create_hypertable('analytics.usage_metrics', 'time', if_not_exists => TRUE);

-- Rate limiting
CREATE TABLE monitoring.rate_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    identifier VARCHAR(255) NOT NULL, -- IP, user_id, api_key
    identifier_type VARCHAR(20) NOT NULL, -- 'ip', 'user', 'api_key'
    endpoint VARCHAR(255),
    
    requests_count INTEGER DEFAULT 0,
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    window_duration_seconds INTEGER DEFAULT 3600,
    limit_per_window INTEGER DEFAULT 1000,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System health metrics
CREATE TABLE monitoring.health_checks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    check_type VARCHAR(50) NOT NULL, -- 'database', 'whisper', 'storage', 'redis'
    status VARCHAR(20) NOT NULL, -- 'healthy', 'degraded', 'unhealthy'
    response_time_ms INTEGER,
    error_message TEXT,
    details JSONB,
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =======================
-- INDEXES FOR PERFORMANCE
-- =======================

-- Users table indexes
CREATE INDEX idx_users_email ON voiceforge.users(email);
CREATE INDEX idx_users_api_key ON voiceforge.users(api_key) WHERE api_key IS NOT NULL;
CREATE INDEX idx_users_stripe_customer ON voiceforge.users(stripe_customer_id) WHERE stripe_customer_id IS NOT NULL;
CREATE INDEX idx_users_plan ON voiceforge.users(plan_id);
CREATE INDEX idx_users_active ON voiceforge.users(is_active) WHERE is_active = true;

-- Transcriptions table indexes
CREATE INDEX idx_transcriptions_user_id ON voiceforge.transcriptions(user_id);
CREATE INDEX idx_transcriptions_status ON voiceforge.transcriptions(status);
CREATE INDEX idx_transcriptions_created_at ON voiceforge.transcriptions(created_at DESC);
CREATE INDEX idx_transcriptions_session ON voiceforge.transcriptions(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX idx_transcriptions_user_created ON voiceforge.transcriptions(user_id, created_at DESC);

-- API Keys indexes
CREATE INDEX idx_api_keys_user_id ON voiceforge.api_keys(user_id);
CREATE INDEX idx_api_keys_hash ON voiceforge.api_keys(key_hash);
CREATE INDEX idx_api_keys_active ON voiceforge.api_keys(is_active) WHERE is_active = true;
CREATE INDEX idx_api_keys_expires ON voiceforge.api_keys(expires_at) WHERE expires_at IS NOT NULL;

-- Subscriptions indexes
CREATE INDEX idx_subscriptions_user ON voiceforge.user_subscriptions(user_id);
CREATE INDEX idx_subscriptions_stripe ON voiceforge.user_subscriptions(stripe_subscription_id);
CREATE INDEX idx_subscriptions_status ON voiceforge.user_subscriptions(status);
CREATE INDEX idx_subscriptions_period ON voiceforge.user_subscriptions(current_period_end);

-- Usage records indexes
CREATE INDEX idx_usage_user_period ON voiceforge.usage_records(user_id, billing_period_start, billing_period_end);
CREATE INDEX idx_usage_transcription ON voiceforge.usage_records(transcription_id);
CREATE INDEX idx_usage_unbilled ON voiceforge.usage_records(billed) WHERE billed = false;

-- WebSocket sessions indexes
CREATE INDEX idx_websocket_user ON voiceforge.websocket_sessions(user_id);
CREATE INDEX idx_websocket_status ON voiceforge.websocket_sessions(status);
CREATE INDEX idx_websocket_session_id ON voiceforge.websocket_sessions(session_id);

-- Analytics indexes
CREATE INDEX idx_usage_metrics_user_time ON analytics.usage_metrics(user_id, time DESC);
CREATE INDEX idx_usage_metrics_endpoint ON analytics.usage_metrics(endpoint, time DESC);
CREATE INDEX idx_usage_metrics_status ON analytics.usage_metrics(status_code, time DESC);

-- Rate limiting indexes
CREATE INDEX idx_rate_limits_identifier ON monitoring.rate_limits(identifier, identifier_type, window_start);
CREATE INDEX idx_rate_limits_window ON monitoring.rate_limits(window_start) WHERE window_start > NOW() - INTERVAL '1 day';

-- =======================
-- MATERIALIZED VIEWS
-- =======================

-- Daily usage summary
CREATE MATERIALIZED VIEW analytics.daily_usage_summary AS
SELECT 
    DATE(time) as usage_date,
    user_id,
    COUNT(*) as total_requests,
    COUNT(CASE WHEN status_code = 200 THEN 1 END) as successful_requests,
    SUM(audio_duration_seconds) as total_audio_seconds,
    SUM(file_size_bytes) as total_bytes_processed,
    AVG(response_time_ms) as avg_response_time_ms,
    COUNT(DISTINCT session_id) as unique_sessions
FROM analytics.usage_metrics
WHERE time >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY DATE(time), user_id;

-- Monthly billing summary
CREATE MATERIALIZED VIEW analytics.monthly_billing_summary AS
SELECT 
    user_id,
    DATE_TRUNC('month', created_at) as billing_month,
    COUNT(*) as transcription_count,
    SUM(duration_seconds) / 60.0 as total_minutes,
    SUM(cost_cents) as total_cost_cents,
    COUNT(DISTINCT DATE(created_at)) as active_days
FROM voiceforge.transcriptions
WHERE status = 'completed'
  AND created_at >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY user_id, DATE_TRUNC('month', created_at);

-- Top languages by usage
CREATE MATERIALIZED VIEW analytics.language_usage_stats AS
SELECT 
    language,
    COUNT(*) as transcription_count,
    AVG(confidence_score) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time_ms,
    SUM(duration_seconds) / 60.0 as total_minutes,
    COUNT(DISTINCT user_id) as unique_users
FROM voiceforge.transcriptions
WHERE status = 'completed'
  AND language IS NOT NULL
  AND created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY language;

-- Create unique indexes for materialized views (for concurrent refresh)
CREATE UNIQUE INDEX idx_daily_usage_summary_unique ON analytics.daily_usage_summary(usage_date, user_id);
CREATE UNIQUE INDEX idx_monthly_billing_summary_unique ON analytics.monthly_billing_summary(user_id, billing_month);
CREATE UNIQUE INDEX idx_language_usage_stats_unique ON analytics.language_usage_stats(language);

-- =======================
-- FUNCTIONS & TRIGGERS
-- =======================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update triggers
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON voiceforge.users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_plans_updated_at
    BEFORE UPDATE ON voiceforge.plans
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_subscriptions_updated_at
    BEFORE UPDATE ON voiceforge.user_subscriptions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_analytics_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.daily_usage_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.monthly_billing_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.language_usage_stats;
    
    -- Log the refresh
    INSERT INTO monitoring.health_checks (service_name, check_type, status, details)
    VALUES ('analytics', 'materialized_view_refresh', 'healthy', 
            jsonb_build_object('refreshed_at', CURRENT_TIMESTAMP));
END;
$$ LANGUAGE plpgsql;

-- Function to calculate user's current usage
CREATE OR REPLACE FUNCTION get_user_monthly_usage(p_user_id UUID, p_month DATE DEFAULT CURRENT_DATE)
RETURNS TABLE (
    transcription_minutes FLOAT,
    api_requests INTEGER,
    cost_cents INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(SUM(t.duration_seconds) / 60.0, 0)::FLOAT as transcription_minutes,
        COUNT(*)::INTEGER as api_requests,
        COALESCE(SUM(t.cost_cents), 0)::INTEGER as cost_cents
    FROM voiceforge.transcriptions t
    WHERE t.user_id = p_user_id
      AND DATE_TRUNC('month', t.created_at) = DATE_TRUNC('month', p_month)
      AND t.status = 'completed';
END;
$$ LANGUAGE plpgsql;

-- Function to clean old data
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Clean old rate limit records (older than 7 days)
    DELETE FROM monitoring.rate_limits 
    WHERE window_start < NOW() - INTERVAL '7 days';
    
    -- Clean old health check records (older than 30 days)
    DELETE FROM monitoring.health_checks 
    WHERE checked_at < NOW() - INTERVAL '30 days';
    
    -- Clean old WebSocket sessions (older than 90 days)
    DELETE FROM voiceforge.websocket_sessions 
    WHERE ended_at < NOW() - INTERVAL '90 days'
      AND status = 'disconnected';
    
    -- Log cleanup
    INSERT INTO monitoring.health_checks (service_name, check_type, status, details)
    VALUES ('database', 'cleanup', 'healthy', 
            jsonb_build_object('cleaned_at', CURRENT_TIMESTAMP));
END;
$$ LANGUAGE plpgsql;

-- =======================
-- SAMPLE DATA
-- =======================

-- Insert default plans
INSERT INTO voiceforge.plans (name, display_name, description, monthly_limit, price_monthly, price_yearly, features) VALUES
('free', 'Free Plan', 'Perfect for trying out VoiceForge', 60, 0.00, 0.00, 
 '["60 minutes/month", "Basic transcription", "Standard models", "Email support"]'::jsonb),
('starter', 'Starter Plan', 'Great for individuals and small teams', 300, 19.99, 199.99,
 '["300 minutes/month", "Advanced transcription", "All models", "Speaker diarization", "Priority support"]'::jsonb),
('pro', 'Professional Plan', 'Perfect for businesses', 1500, 79.99, 799.99,
 '["1500 minutes/month", "All features", "Batch processing", "Custom vocabulary", "API access", "Priority support"]'::jsonb),
('enterprise', 'Enterprise Plan', 'For large organizations', 10000, 299.99, 2999.99,
 '["10000 minutes/month", "All features", "Dedicated support", "SLA guarantee", "Custom integrations", "White-label options"]'::jsonb);

-- Insert admin user
INSERT INTO voiceforge.users (email, password_hash, full_name, company, role, plan_id, is_active, email_verified) 
SELECT 
    'admin@voiceforge.ai', 
    '$2b$12$KIXxPfx0h3zH9Qq1U8xQz.7rV5dVMfLs3.4P3hTt5ZvH3cJhGQKGm', 
    'Admin User', 
    'VoiceForge Inc.', 
    'admin', 
    p.id, 
    true, 
    true
FROM voiceforge.plans p WHERE p.name = 'enterprise';

-- Insert demo user
INSERT INTO voiceforge.users (email, password_hash, full_name, company, role, plan_id, is_active, email_verified)
SELECT 
    'demo@voiceforge.ai', 
    '$2b$12$KIXxPfx0h3zH9Qq1U8xQz.7rV5dVMfLs3.4P3hTt5ZvH3cJhGQKGm', 
    'Demo User', 
    'Demo Corp', 
    'user', 
    p.id, 
    true, 
    true
FROM voiceforge.plans p WHERE p.name = 'free';

-- =======================
-- PERMISSIONS & SECURITY
-- =======================

-- Create application role
CREATE ROLE voiceforge_app WITH LOGIN;
GRANT CONNECT ON DATABASE voiceforge TO voiceforge_app;

-- Grant schema permissions
GRANT USAGE ON SCHEMA voiceforge TO voiceforge_app;
GRANT USAGE ON SCHEMA analytics TO voiceforge_app;
GRANT USAGE ON SCHEMA monitoring TO voiceforge_app;

-- Grant table permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA voiceforge TO voiceforge_app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO voiceforge_app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO voiceforge_app;

-- Grant sequence permissions
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA voiceforge TO voiceforge_app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO voiceforge_app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO voiceforge_app;

-- Grant function execute permissions
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA voiceforge TO voiceforge_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA analytics TO voiceforge_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA monitoring TO voiceforge_app;

-- Row Level Security (RLS) for multi-tenancy
ALTER TABLE voiceforge.transcriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE voiceforge.api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE voiceforge.websocket_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE voiceforge.usage_records ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (users can only see their own data)
CREATE POLICY user_transcriptions_policy ON voiceforge.transcriptions
    FOR ALL TO voiceforge_app
    USING (user_id = current_setting('app.current_user_id', true)::UUID);

CREATE POLICY user_api_keys_policy ON voiceforge.api_keys
    FOR ALL TO voiceforge_app
    USING (user_id = current_setting('app.current_user_id', true)::UUID);

-- =======================
-- MONITORING SETUP
-- =======================

-- Create monitoring views for health checks
CREATE VIEW monitoring.system_health AS
SELECT 
    'database' as component,
    CASE WHEN COUNT(*) > 0 THEN 'healthy' ELSE 'unhealthy' END as status,
    NOW() as last_check
FROM voiceforge.users
WHERE is_active = true
UNION ALL
SELECT 
    'transcription_service' as component,
    CASE WHEN COUNT(*) > 0 THEN 'healthy' ELSE 'degraded' END as status,
    MAX(created_at) as last_check
FROM voiceforge.transcriptions
WHERE created_at > NOW() - INTERVAL '1 hour';

-- Final message
SELECT 
    'VoiceForge Production Database Schema Initialized Successfully!' as status,
    COUNT(DISTINCT table_name) as tables_created,
    NOW() as initialized_at
FROM information_schema.tables 
WHERE table_schema IN ('voiceforge', 'analytics', 'monitoring');

-- Instructions for next steps
SELECT 'NEXT STEPS:' as instruction, '1. Update DATABASE_URL in .env file' as action
UNION ALL
SELECT '', '2. Run database migrations if needed'
UNION ALL
SELECT '', '3. Configure Stripe webhooks'
UNION ALL
SELECT '', '4. Setup monitoring alerts'
UNION ALL
SELECT '', '5. Test all database operations'
ORDER BY instruction DESC;