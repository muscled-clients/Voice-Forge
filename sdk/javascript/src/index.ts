/**
 * VoiceForge JavaScript/TypeScript SDK
 * 
 * Official SDK for the VoiceForge Speech-to-Text API
 * Provides easy-to-use interfaces for transcription, streaming, and batch processing.
 */

export { VoiceForgeClient } from './client';
export * from './types';
export * from './errors';
export * from './utils';

// Re-export commonly used types
export type {
  TranscriptionJob,
  TranscriptionOptions,
  StreamingResult,
  Word,
  DiarizationSegment,
  LanguageDetectionResult,
  UserInfo,
  ModelInfo,
  TranscriptionStatus,
  AudioFormat,
  UserTier,
} from './types';