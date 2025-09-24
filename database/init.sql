-- VoiceForge Database Schema
-- PostgreSQL initialization script

-- Create database
CREATE DATABASE IF NOT EXISTS voiceforge;
\c voiceforge;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Users table
CREATE TABLE core.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    company VARCHAR(255),
    api_key VARCHAR(255) UNIQUE,
    role VARCHAR(50) DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    email_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Transcriptions table
CREATE TABLE core.transcriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES core.users(id) ON DELETE CASCADE,
    file_name VARCHAR(255) NOT NULL,
    file_size BIGINT,
    file_hash VARCHAR(64),
    mime_type VARCHAR(100),
    duration_seconds FLOAT,
    language VARCHAR(10),
    detected_languages JSONB,
    transcription_text TEXT,
    confidence_score FLOAT,
    word_count INTEGER,
    segments JSONB,
    speakers JSONB,  -- For diarization results
    processing_time_ms INTEGER,
    model_used VARCHAR(50),
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Convert to TimescaleDB hypertable for time-series optimization
SELECT create_hypertable('core.transcriptions', 'created_at', if_not_exists => TRUE);

-- API Keys table
CREATE TABLE core.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES core.users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    last_used TIMESTAMP WITH TIME ZONE,
    usage_count INTEGER DEFAULT 0,
    rate_limit INTEGER DEFAULT 100,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    scopes JSONB DEFAULT '["transcribe:read", "transcribe:write"]'::jsonb
);

-- Batch Jobs table
CREATE TABLE core.batch_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES core.users(id) ON DELETE CASCADE,
    job_name VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    total_files INTEGER,
    processed_files INTEGER DEFAULT 0,
    failed_files INTEGER DEFAULT 0,
    callback_url TEXT,
    options JSONB DEFAULT '{}'::jsonb,
    results JSONB,
    error_log JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    processing_time_seconds FLOAT
);

-- WebSocket Sessions table
CREATE TABLE core.websocket_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID REFERENCES core.users(id) ON DELETE CASCADE,
    connection_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'active',
    duration_seconds FLOAT,
    total_audio_seconds FLOAT,
    transcription_count INTEGER DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Usage Analytics table (TimescaleDB optimized)
CREATE TABLE analytics.usage_metrics (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    user_id UUID,
    endpoint VARCHAR(255),
    method VARCHAR(10),
    status_code INTEGER,
    response_time_ms INTEGER,
    tokens_used INTEGER,
    audio_duration_seconds FLOAT,
    file_size_bytes BIGINT,
    model_used VARCHAR(50),
    language VARCHAR(10),
    ip_address INET,
    user_agent TEXT,
    error_type VARCHAR(100),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Convert to hypertable
SELECT create_hypertable('analytics.usage_metrics', 'time', if_not_exists => TRUE);

-- Speaker Diarization Results table
CREATE TABLE core.diarization_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transcription_id UUID REFERENCES core.transcriptions(id) ON DELETE CASCADE,
    speaker_count INTEGER,
    speakers JSONB NOT NULL, -- Array of speaker objects with segments
    confidence_scores JSONB,
    overlap_duration_seconds FLOAT,
    silence_duration_seconds FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Billing table
CREATE TABLE core.billing (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES core.users(id) ON DELETE CASCADE,
    billing_period_start DATE NOT NULL,
    billing_period_end DATE NOT NULL,
    total_minutes FLOAT DEFAULT 0,
    total_cost DECIMAL(10, 2) DEFAULT 0,
    plan_type VARCHAR(50) DEFAULT 'free',
    invoice_url TEXT,
    payment_status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    paid_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for performance
CREATE INDEX idx_users_email ON core.users(email);
CREATE INDEX idx_users_api_key ON core.users(api_key);
CREATE INDEX idx_transcriptions_user_id ON core.transcriptions(user_id);
CREATE INDEX idx_transcriptions_status ON core.transcriptions(status);
CREATE INDEX idx_transcriptions_created_at ON core.transcriptions(created_at DESC);
CREATE INDEX idx_api_keys_user_id ON core.api_keys(user_id);
CREATE INDEX idx_api_keys_key_hash ON core.api_keys(key_hash);
CREATE INDEX idx_batch_jobs_user_id ON core.batch_jobs(user_id);
CREATE INDEX idx_batch_jobs_status ON core.batch_jobs(status);
CREATE INDEX idx_websocket_sessions_user_id ON core.websocket_sessions(user_id);
CREATE INDEX idx_diarization_transcription_id ON core.diarization_results(transcription_id);
CREATE INDEX idx_usage_metrics_user_time ON analytics.usage_metrics(user_id, time DESC);

-- Create materialized views for analytics
CREATE MATERIALIZED VIEW analytics.daily_usage AS
SELECT 
    DATE(time) as date,
    user_id,
    COUNT(*) as request_count,
    SUM(audio_duration_seconds) as total_audio_seconds,
    AVG(response_time_ms) as avg_response_time,
    SUM(file_size_bytes) as total_bytes_processed
FROM analytics.usage_metrics
GROUP BY DATE(time), user_id;

CREATE MATERIALIZED VIEW analytics.language_stats AS
SELECT 
    language,
    COUNT(*) as transcription_count,
    AVG(confidence_score) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time
FROM core.transcriptions
WHERE status = 'completed'
GROUP BY language;

-- Refresh materialized views periodically
CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.daily_usage;
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.language_stats;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON core.users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Sample data for testing
INSERT INTO core.users (email, password_hash, full_name, company, role) VALUES
('admin@voiceforge.ai', '$2b$12$KIXxPfx0h3zH9Qq1U8xQz.7rV5dVMfLs3.4P3hTt5ZvH3cJhGQKGm', 'Admin User', 'VoiceForge', 'admin'),
('demo@example.com', '$2b$12$KIXxPfx0h3zH9Qq1U8xQz.7rV5dVMfLs3.4P3hTt5ZvH3cJhGQKGm', 'Demo User', 'Demo Corp', 'user');

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE voiceforge TO voiceforge;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA core TO voiceforge;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO voiceforge;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA core TO voiceforge;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO voiceforge;