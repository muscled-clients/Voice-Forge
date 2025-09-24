/**
 * VoiceForge SDK Types
 * 
 * TypeScript type definitions for the VoiceForge API
 */

// Enums
export enum TranscriptionStatus {
  PENDING = 'pending',
  PROCESSING = 'processing', 
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

export enum AudioFormat {
  WAV = 'wav',
  MP3 = 'mp3',
  M4A = 'm4a',
  FLAC = 'flac',
  OGG = 'ogg',
  WEBM = 'webm',
}

export enum UserTier {
  FREE = 'free',
  BASIC = 'basic',
  PRO = 'pro',
  ENTERPRISE = 'enterprise',
}

export enum LanguageCode {
  EN = 'en',
  ES = 'es',
  FR = 'fr',
  DE = 'de',
  IT = 'it',
  PT = 'pt',
  RU = 'ru',
  JA = 'ja',
  KO = 'ko',
  ZH = 'zh',
  AR = 'ar',
  HI = 'hi',
  TR = 'tr',
  PL = 'pl',
  NL = 'nl',
  SV = 'sv',
  DA = 'da',
  NO = 'no',
  FI = 'fi',
  CS = 'cs',
}

// Core interfaces
export interface Word {
  text: string;
  startTime: number;
  endTime: number;
  confidence: number;
  speakerId?: number;
}

export interface DiarizationSegment {
  speakerId: number;
  startTime: number;
  endTime: number;
  confidence: number;
  text?: string;
}

export interface LanguageAlternative {
  code: string;
  name: string;
  confidence: number;
}

export interface LanguageDetectionResult {
  code: string;
  name: string;
  confidence: number;
  method: string;
  alternatives?: LanguageAlternative[];
}

export interface TranscriptionOptions {
  model?: string;
  language?: string;
  enableDiarization?: boolean;
  enablePunctuation?: boolean;
  enableWordTimestamps?: boolean;
  enableLanguageDetection?: boolean;
  maxSpeakers?: number;
  boostKeywords?: string[];
  customVocabulary?: string[];
  temperature?: number;
  compressionRatioThreshold?: number;
  logprobThreshold?: number;
  noSpeechThreshold?: number;
}

export interface TranscriptionJob {
  id: string;
  userId: string;
  status: TranscriptionStatus;
  createdAt: string;
  updatedAt?: string;
  completedAt?: string;
  
  // Input details
  filename: string;
  audioFormat?: AudioFormat;
  audioDuration?: number;
  audioSize?: number;
  
  // Processing details
  modelUsed?: string;
  languageCode?: string;
  processingTime?: number;
  
  // Results
  transcript?: string;
  confidence?: number;
  words?: Word[];
  diarization?: DiarizationSegment[];
  detectedLanguage?: LanguageDetectionResult;
  
  // Metadata
  wordCount?: number;
  metadata?: Record<string, any>;
  
  // Error info
  error?: string;
  errorDetails?: Record<string, any>;
}

export interface StreamingResult {
  transcript: string;
  isFinal: boolean;
  confidence: number;
  words?: Word[];
  alternatives?: Record<string, any>[];
  timestamp: number;
  duration?: number;
  language?: string;
  metadata?: Record<string, any>;
}

export interface ModelInfo {
  name: string;
  modelId: string;
  type: string;
  description: string;
  supportedLanguages: string[];
  maxDuration?: number;
  accuracyScore?: number;
  latencyMs?: number;
  isAvailable: boolean;
  requiresGpu: boolean;
}

export interface UserInfo {
  id: string;
  email: string;
  fullName: string;
  tier: UserTier;
  creditsRemaining: number;
  isActive: boolean;
  createdAt: string;
  updatedAt?: string;
}

export interface UserAnalytics {
  userId: string;
  totalTranscriptions: number;
  totalAudioMinutes: number;
  totalCreditsUsed: number;
  averageConfidence?: number;
  mostUsedLanguage?: string;
  mostUsedModel?: string;
  successRate: number;
  periodStart: string;
  periodEnd: string;
}

export interface QuotaInfo {
  userId: string;
  tier: UserTier;
  creditsTotal: number;
  creditsUsed: number;
  creditsRemaining: number;
  monthlyLimit: number;
  monthlyUsed: number;
  resetDate: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  totalCount: number;
  page: number;
  perPage: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
}

export interface BatchJob {
  batchId: string;
  jobs: TranscriptionJob[];
  createdAt: string;
  totalJobs: number;
  completedJobs: number;
  failedJobs: number;
  status: string;
}

// Client configuration
export interface ClientConfig {
  apiKey?: string;
  baseUrl?: string;
  timeout?: number;
  maxRetries?: number;
  retryDelay?: number;
  userAgent?: string;
}

// Callback types
export type ProgressCallback = (progress: number) => void;
export type BatchProgressCallback = (jobId: string, progress: number) => void;
export type StreamingCallback = (result: StreamingResult) => void;
export type ErrorCallback = (error: Error) => void;

// Request options
export interface RequestOptions {
  timeout?: number;
  retries?: number;
  headers?: Record<string, string>;
}

// WebSocket options
export interface StreamingOptions extends TranscriptionOptions {
  audioFormat?: AudioFormat;
  sampleRate?: number;
  chunkDuration?: number;
  onResult?: StreamingCallback;
  onError?: ErrorCallback;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

// File upload types
export interface FileUpload {
  file: File | Blob;
  filename: string;
  options?: TranscriptionOptions;
}

export interface BatchUpload {
  files: FileUpload[];
  priority?: number;
  callbackUrl?: string;
}

// Export utility types
export interface ExportOptions {
  format: 'txt' | 'json' | 'srt' | 'vtt';
  includeTimestamps?: boolean;
  includeSpeakers?: boolean;
  maxLineLength?: number;
}

// HTTP response types
export interface ApiResponse<T = any> {
  data?: T;
  error?: {
    message: string;
    code?: string;
    details?: Record<string, any>;
  };
  status: number;
  headers: Record<string, string>;
}

// WebSocket message types
export interface WebSocketMessage {
  type: 'config' | 'audio' | 'result' | 'error' | 'end';
  data: any;
  timestamp?: number;
}

export interface WebSocketConfig {
  audioFormat: AudioFormat;
  sampleRate: number;
  chunkDuration: number;
  options?: TranscriptionOptions;
}

// Event types for event emitter
export interface ClientEvents {
  'transcription:started': (jobId: string) => void;
  'transcription:progress': (jobId: string, progress: number) => void;
  'transcription:completed': (job: TranscriptionJob) => void;
  'transcription:failed': (jobId: string, error: string) => void;
  'stream:connected': () => void;
  'stream:disconnected': () => void;
  'stream:result': (result: StreamingResult) => void;
  'stream:error': (error: Error) => void;
  'error': (error: Error) => void;
}