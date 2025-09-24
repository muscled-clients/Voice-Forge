-- VoiceForge Database Schema (Simplified)
-- PostgreSQL initialization script for existing postgres user

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS voiceforge;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Users table (simplified)
CREATE TABLE IF NOT EXISTS voiceforge.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE,
    full_name VARCHAR(255),
    api_key VARCHAR(255) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Transcriptions table (main table)
CREATE TABLE IF NOT EXISTS voiceforge.transcriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES voiceforge.users(id) ON DELETE SET NULL,
    file_name VARCHAR(255) NOT NULL,
    file_size BIGINT,
    file_hash VARCHAR(64),
    duration_seconds FLOAT,
    language VARCHAR(10),
    transcription_text TEXT,
    confidence_score FLOAT,
    word_count INTEGER,
    segments JSONB,
    processing_time_ms INTEGER,
    model_used VARCHAR(50) DEFAULT 'whisper-tiny',
    status VARCHAR(50) DEFAULT 'completed',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Usage metrics table (for analytics)
CREATE TABLE IF NOT EXISTS analytics.usage_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    endpoint VARCHAR(255),
    method VARCHAR(10),
    status_code INTEGER,
    response_time_ms INTEGER,
    file_size_bytes BIGINT,
    audio_duration_seconds FLOAT,
    language VARCHAR(10),
    model_used VARCHAR(50),
    ip_address INET,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Session tracking (simplified)
CREATE TABLE IF NOT EXISTS voiceforge.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_start TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    session_end TIMESTAMP WITH TIME ZONE,
    total_transcriptions INTEGER DEFAULT 0,
    total_audio_duration FLOAT DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_transcriptions_created_at ON voiceforge.transcriptions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_transcriptions_language ON voiceforge.transcriptions(language);
CREATE INDEX IF NOT EXISTS idx_transcriptions_status ON voiceforge.transcriptions(status);
CREATE INDEX IF NOT EXISTS idx_usage_metrics_timestamp ON analytics.usage_metrics(timestamp DESC);

-- Insert sample data
INSERT INTO voiceforge.users (email, full_name, api_key) VALUES
('demo@voiceforge.ai', 'Demo User', 'demo_api_key_' || substring(md5(random()::text), 1, 10))
ON CONFLICT (email) DO NOTHING;

-- Create a view for easy stats access
CREATE OR REPLACE VIEW voiceforge.stats_view AS
SELECT 
    (SELECT COUNT(*) FROM voiceforge.transcriptions) as total_transcriptions,
    (SELECT COUNT(*) FROM voiceforge.sessions WHERE session_end IS NULL) as active_sessions,
    (SELECT COUNT(DISTINCT language) FROM voiceforge.transcriptions WHERE language IS NOT NULL) as languages_detected,
    (SELECT AVG(processing_time_ms) FROM voiceforge.transcriptions WHERE processing_time_ms IS NOT NULL) as avg_processing_time,
    (SELECT SUM(audio_duration_seconds) FROM analytics.usage_metrics WHERE audio_duration_seconds IS NOT NULL) as total_audio_processed;

-- Test data insertion
INSERT INTO voiceforge.transcriptions 
(file_name, file_size, language, transcription_text, confidence_score, processing_time_ms, status)
VALUES 
('test_audio.wav', 1024000, 'en', 'This is a test transcription to verify database setup.', 0.95, 2500, 'completed')
ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA voiceforge TO postgres;
GRANT ALL PRIVILEGES ON SCHEMA analytics TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA voiceforge TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA voiceforge TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO postgres;

-- Confirmation message
SELECT 'VoiceForge database schema created successfully!' as message;