import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from app.services.language_detection import (
    AudioLanguageDetector,
    TextLanguageDetector, 
    HybridLanguageDetector,
    LanguageDetectionResult,
    detect_language
)


class TestAudioLanguageDetector:
    """Test audio-based language detection"""
    
    @pytest.fixture
    def audio_detector(self):
        return AudioLanguageDetector()
    
    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio data"""
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return audio, sample_rate
    
    @pytest.mark.asyncio
    async def test_audio_detection_english(self, audio_detector, sample_audio):
        """Test audio language detection for English"""
        audio_data, sample_rate = sample_audio
        
        with patch('whisper.load_model') as mock_load_model:
            # Mock Whisper model
            mock_model = Mock()
            mock_model.device = 'cpu'
            mock_model.detect_language.return_value = ('en', {'en': 0.95, 'es': 0.03, 'fr': 0.02})
            mock_load_model.return_value = mock_model
            
            with patch('whisper.pad_or_trim') as mock_pad, \
                 patch('whisper.log_mel_spectrogram') as mock_mel:
                
                mock_pad.return_value = audio_data
                mock_mel.return_value = Mock()
                mock_mel.return_value.to.return_value = Mock()
                
                result = await audio_detector.detect_from_audio(audio_data, sample_rate)
                
                assert isinstance(result, LanguageDetectionResult)
                assert result.language_code == 'en'
                assert result.language_name == 'English'
                assert result.confidence > 0.9
                assert result.method == 'audio'
                assert len(result.alternative_languages) > 0
    
    @pytest.mark.asyncio
    async def test_audio_detection_bytes_input(self, audio_detector):
        """Test audio detection with bytes input"""
        # Create fake audio bytes
        audio_bytes = np.random.randint(-32768, 32767, 16000, dtype=np.int16).tobytes()
        
        with patch('whisper.load_model') as mock_load_model:
            mock_model = Mock()
            mock_model.device = 'cpu'
            mock_model.detect_language.return_value = ('es', {'es': 0.88, 'en': 0.10, 'fr': 0.02})
            mock_load_model.return_value = mock_model
            
            with patch('whisper.pad_or_trim') as mock_pad, \
                 patch('whisper.log_mel_spectrogram') as mock_mel:
                
                mock_pad.return_value = Mock()
                mock_mel.return_value = Mock()
                mock_mel.return_value.to.return_value = Mock()
                
                result = await audio_detector.detect_from_audio(audio_bytes)
                
                assert result.language_code == 'es'
                assert result.language_name == 'Spanish'
    
    @pytest.mark.asyncio
    async def test_audio_detection_error_fallback(self, audio_detector, sample_audio):
        """Test audio detection error handling"""
        audio_data, sample_rate = sample_audio
        
        with patch('whisper.load_model', side_effect=Exception("Model loading failed")):
            result = await audio_detector.detect_from_audio(audio_data, sample_rate)
            
            assert result.language_code == 'en'  # Fallback to English
            assert result.confidence == 0.5


class TestTextLanguageDetector:
    """Test text-based language detection"""
    
    @pytest.fixture
    def text_detector(self):
        return TextLanguageDetector()
    
    @pytest.mark.asyncio
    async def test_text_detection_english(self, text_detector):
        """Test text language detection for English"""
        text = "Hello, this is a sample text in English language for testing purposes."
        
        with patch.object(text_detector, 'detect_with_langdetect') as mock_langdetect, \
             patch.object(text_detector, 'detect_with_transformer') as mock_transformer, \
             patch.object(text_detector, 'detect_with_patterns') as mock_patterns:
            
            mock_langdetect.return_value = {"language": "en", "confidence": 0.95}
            mock_transformer.return_value = {"language": "en", "confidence": 0.92}
            mock_patterns.return_value = {"language": "en", "confidence": 0.75}
            
            result = await text_detector.detect_from_text(text)
            
            assert result.language_code == 'en'
            assert result.language_name == 'English'
            assert result.confidence > 0.8
            assert result.method == 'text'
    
    @pytest.mark.asyncio
    async def test_text_detection_spanish(self, text_detector):
        """Test text language detection for Spanish"""
        text = "Hola, este es un texto de muestra en español para propósitos de prueba."
        
        with patch.object(text_detector, 'detect_with_langdetect') as mock_langdetect, \
             patch.object(text_detector, 'detect_with_transformer') as mock_transformer, \
             patch.object(text_detector, 'detect_with_patterns') as mock_patterns:
            
            mock_langdetect.return_value = {"language": "es", "confidence": 0.93}
            mock_transformer.return_value = {"language": "es", "confidence": 0.89}
            mock_patterns.return_value = {"language": "es", "confidence": 0.72}
            
            result = await text_detector.detect_from_text(text)
            
            assert result.language_code == 'es'
            assert result.language_name == 'Spanish'
    
    @pytest.mark.asyncio
    async def test_text_detection_empty_text(self, text_detector):
        """Test text detection with empty/short text"""
        result = await text_detector.detect_from_text("")
        
        assert result.language_code == 'en'
        assert result.confidence == 0.3
        
        result = await text_detector.detect_from_text("Hi")
        
        assert result.language_code == 'en'
        assert result.confidence == 0.3
    
    def test_langdetect_method(self, text_detector):
        """Test langdetect method"""
        text = "Bonjour, comment ça va? C'est un test en français."
        
        with patch('langdetect.detect_langs') as mock_detect:
            from langdetect.lang_detect_exception import LangDetectException
            
            # Mock language detection
            mock_lang = Mock()
            mock_lang.lang = 'fr'
            mock_lang.prob = 0.91
            mock_detect.return_value = [mock_lang]
            
            result = text_detector.detect_with_langdetect(text)
            
            assert result["language"] == 'fr'
            assert result["confidence"] == 0.91
    
    def test_transformer_method(self, text_detector):
        """Test transformer method"""
        text = "Dies ist ein deutscher Satz zum Testen der Spracherkennung."
        
        # Mock transformer pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "de", "score": 0.94},
            {"label": "en", "score": 0.04},
            {"label": "fr", "score": 0.02}
        ]
        text_detector.transformer_detector = mock_pipeline
        
        result = text_detector.detect_with_transformer(text)
        
        assert result["language"] == "de"
        assert result["confidence"] == 0.94
        assert len(result["alternatives"]) == 2
    
    def test_pattern_method(self, text_detector):
        """Test pattern-based method"""
        # English text with common English character patterns
        text = "The quick brown fox jumps over the lazy dog. This sentence contains every letter."
        
        result = text_detector.detect_with_patterns(text)
        
        assert result["language"] in ['en', 'es', 'fr', 'de', 'ru', 'zh']
        assert 0.1 <= result["confidence"] <= 1.0


class TestHybridLanguageDetector:
    """Test hybrid language detection"""
    
    @pytest.fixture
    def hybrid_detector(self):
        return HybridLanguageDetector()
    
    @pytest.fixture
    def sample_audio_and_text(self):
        """Sample data for hybrid testing"""
        audio = np.random.randn(16000).astype(np.float32)
        text = "This is a test sentence in English for hybrid language detection."
        return audio, text
    
    @pytest.mark.asyncio
    async def test_hybrid_detection_agreement(self, hybrid_detector, sample_audio_and_text):
        """Test hybrid detection when audio and text agree"""
        audio, text = sample_audio_and_text
        
        with patch.object(hybrid_detector.audio_detector, 'detect_from_audio') as mock_audio, \
             patch.object(hybrid_detector.text_detector, 'detect_from_text') as mock_text:
            
            # Both methods detect English
            mock_audio.return_value = LanguageDetectionResult(
                language_code='en',
                language_name='English',
                confidence=0.90,
                method='audio'
            )
            
            mock_text.return_value = LanguageDetectionResult(
                language_code='en',
                language_name='English', 
                confidence=0.85,
                method='text'
            )
            
            result = await hybrid_detector.detect_language(audio, text, "hybrid")
            
            assert result.language_code == 'en'
            assert result.confidence > 0.8  # Should be increased due to agreement
            assert result.method == 'hybrid'
    
    @pytest.mark.asyncio
    async def test_hybrid_detection_disagreement(self, hybrid_detector, sample_audio_and_text):
        """Test hybrid detection when audio and text disagree"""
        audio, text = sample_audio_and_text
        
        with patch.object(hybrid_detector.audio_detector, 'detect_from_audio') as mock_audio, \
             patch.object(hybrid_detector.text_detector, 'detect_from_text') as mock_text:
            
            # Methods disagree
            mock_audio.return_value = LanguageDetectionResult(
                language_code='en',
                language_name='English',
                confidence=0.88,
                method='audio'
            )
            
            mock_text.return_value = LanguageDetectionResult(
                language_code='es',
                language_name='Spanish',
                confidence=0.75,
                method='text'
            )
            
            result = await hybrid_detector.detect_language(audio, text, "hybrid")
            
            # Should pick the higher confidence one but with reduced confidence
            assert result.language_code == 'en'  # Higher confidence
            assert result.confidence < 0.88  # Reduced due to disagreement
            assert result.method == 'hybrid'
            assert len(result.alternative_languages) > 0
    
    @pytest.mark.asyncio
    async def test_audio_only_detection(self, hybrid_detector, sample_audio_and_text):
        """Test audio-only detection mode"""
        audio, _ = sample_audio_and_text
        
        with patch.object(hybrid_detector.audio_detector, 'detect_from_audio') as mock_audio:
            mock_audio.return_value = LanguageDetectionResult(
                language_code='fr',
                language_name='French',
                confidence=0.82,
                method='audio'
            )
            
            result = await hybrid_detector.detect_language(audio, None, "audio")
            
            assert result.language_code == 'fr'
            assert result.method == 'audio'
    
    @pytest.mark.asyncio
    async def test_text_only_detection(self, hybrid_detector, sample_audio_and_text):
        """Test text-only detection mode"""
        _, text = sample_audio_and_text
        
        with patch.object(hybrid_detector.text_detector, 'detect_from_text') as mock_text:
            mock_text.return_value = LanguageDetectionResult(
                language_code='de',
                language_name='German',
                confidence=0.91,
                method='text'
            )
            
            result = await hybrid_detector.detect_language(None, text, "text")
            
            assert result.language_code == 'de'
            assert result.method == 'text'
    
    @pytest.mark.asyncio
    async def test_fallback_detection(self, hybrid_detector):
        """Test fallback when no input provided"""
        result = await hybrid_detector.detect_language(None, None, "hybrid")
        
        assert result.language_code == 'en'  # Fallback to English
        assert result.confidence == 0.3
        assert result.method == 'fallback'


class TestLanguageDetectionIntegration:
    """Integration tests for language detection"""
    
    @pytest.mark.asyncio
    async def test_detect_language_function_audio(self):
        """Test the convenience function with audio"""
        audio = np.random.randn(8000).astype(np.float32)
        
        with patch('app.services.language_detection.language_detector') as mock_detector:
            mock_result = LanguageDetectionResult(
                language_code='en',
                language_name='English',
                confidence=0.87,
                method='audio'
            )
            mock_detector.detect_language = AsyncMock(return_value=mock_result)
            
            result = await detect_language(audio_data=audio, method="audio")
            
            assert result.language_code == 'en'
            mock_detector.detect_language.assert_called_once_with(audio, None, "audio")
    
    @pytest.mark.asyncio
    async def test_detect_language_function_text(self):
        """Test the convenience function with text"""
        text = "Ceci est un test en français."
        
        with patch('app.services.language_detection.language_detector') as mock_detector:
            mock_result = LanguageDetectionResult(
                language_code='fr',
                language_name='French',
                confidence=0.93,
                method='text'
            )
            mock_detector.detect_language = AsyncMock(return_value=mock_result)
            
            result = await detect_language(text=text, method="text")
            
            assert result.language_code == 'fr'
            mock_detector.detect_language.assert_called_once_with(None, text, "text")
    
    @pytest.mark.asyncio
    async def test_detect_language_function_hybrid(self):
        """Test the convenience function with hybrid mode"""
        audio = np.random.randn(12000).astype(np.float32)
        text = "This is a hybrid language detection test."
        
        with patch('app.services.language_detection.language_detector') as mock_detector:
            mock_result = LanguageDetectionResult(
                language_code='en',
                language_name='English',
                confidence=0.95,
                method='hybrid'
            )
            mock_detector.detect_language = AsyncMock(return_value=mock_result)
            
            result = await detect_language(audio_data=audio, text=text, method="hybrid")
            
            assert result.language_code == 'en'
            assert result.confidence == 0.95
            mock_detector.detect_language.assert_called_once_with(audio, text, "hybrid")


@pytest.fixture
def mock_whisper():
    """Mock Whisper dependencies"""
    with patch('whisper.load_model') as mock_load, \
         patch('whisper.pad_or_trim') as mock_pad, \
         patch('whisper.log_mel_spectrogram') as mock_mel:
        
        mock_model = Mock()
        mock_model.device = 'cpu'
        mock_model.detect_language.return_value = ('en', {'en': 0.95, 'es': 0.03, 'fr': 0.02})
        mock_load.return_value = mock_model
        
        mock_pad.return_value = Mock()
        mock_mel.return_value = Mock()
        mock_mel.return_value.to.return_value = Mock()
        
        yield mock_model


@pytest.fixture
def mock_transformers():
    """Mock transformers pipeline"""
    with patch('transformers.pipeline') as mock_pipeline:
        mock_pipe = Mock()
        mock_pipe.return_value = [
            {"label": "en", "score": 0.92},
            {"label": "es", "score": 0.05},
            {"label": "fr", "score": 0.03}
        ]
        mock_pipeline.return_value = mock_pipe
        yield mock_pipe