#!/usr/bin/env python3
"""
VoiceForge STT Performance Benchmarking Tool

This script benchmarks the performance of the VoiceForge STT service,
measuring latency, accuracy, throughput, and resource usage.
"""

import asyncio
import time
import statistics
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import httpx
import aiofiles
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor
import wave
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test"""
    test_name: str
    latency_ms: float
    accuracy_wer: Optional[float]
    accuracy_cer: Optional[float]
    confidence: float
    throughput_fps: float  # Files per second
    cpu_usage: float
    memory_usage_mb: float
    model_used: str
    audio_duration: float
    timestamp: float


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results"""
    results: List[BenchmarkResult]
    summary: Dict
    config: Dict


class PerformanceBenchmark:
    """
    Performance benchmarking for VoiceForge STT
    """
    
    def __init__(
        self,
        api_endpoint: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        output_dir: str = "./benchmark_results"
    ):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.client = httpx.AsyncClient(
            timeout=300.0,  # 5 minutes timeout
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {}
        )
        
        self.results: List[BenchmarkResult] = []
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.client.aclose()
    
    def generate_test_audio(
        self,
        duration: float,
        sample_rate: int = 16000,
        frequency: float = 440.0
    ) -> bytes:
        """
        Generate test audio data
        """
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        # Generate sine wave with some noise
        signal = np.sin(2 * np.pi * frequency * t) * 0.3
        noise = np.random.normal(0, 0.05, samples)
        audio = signal + noise
        
        # Convert to 16-bit PCM
        audio_int = (audio * 32767).astype(np.int16)
        
        # Create WAV file in memory
        import io
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int.tobytes())
        
        return buffer.getvalue()
    
    async def measure_latency(
        self,
        audio_data: bytes,
        model: str = "whisper-base",
        runs: int = 10
    ) -> Dict:
        """
        Measure transcription latency
        """
        latencies = []
        
        for i in range(runs):
            start_time = time.perf_counter()
            
            try:
                files = {"audio": ("test.wav", audio_data, "audio/wav")}
                data = {"model": model}
                
                response = await self.client.post(
                    f"{self.api_endpoint}/api/v1/transcribe/sync",
                    files=files,
                    data=data
                )
                
                end_time = time.perf_counter()
                
                if response.status_code == 200:
                    latency = (end_time - start_time) * 1000
                    latencies.append(latency)
                else:
                    logger.error(f"Request failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Request error: {e}")
                continue
            
            # Small delay between requests
            await asyncio.sleep(0.1)
        
        if not latencies:
            return {"error": "No successful requests"}
        
        return {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "min": min(latencies),
            "max": max(latencies),
            "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "samples": latencies
        }
    
    async def measure_throughput(
        self,
        concurrent_requests: int = 10,
        total_requests: int = 100,
        audio_duration: float = 5.0
    ) -> Dict:
        """
        Measure transcription throughput
        """
        audio_data = self.generate_test_audio(audio_duration)
        
        async def single_request():
            try:
                files = {"audio": ("test.wav", audio_data, "audio/wav")}
                data = {"model": "whisper-base"}
                
                start_time = time.perf_counter()
                response = await self.client.post(
                    f"{self.api_endpoint}/api/v1/transcribe/sync",
                    files=files,
                    data=data
                )
                end_time = time.perf_counter()
                
                return {
                    "success": response.status_code == 200,
                    "latency": (end_time - start_time) * 1000,
                    "status_code": response.status_code
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "latency": 0
                }
        
        start_time = time.perf_counter()
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def limited_request():
            async with semaphore:
                return await single_request()
        
        # Execute all requests
        tasks = [limited_request() for _ in range(total_requests)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        latencies = [r["latency"] for r in successful]
        
        return {
            "total_requests": total_requests,
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / total_requests,
            "total_time": total_time,
            "requests_per_second": total_requests / total_time,
            "avg_latency": statistics.mean(latencies) if latencies else 0,
            "p95_latency": np.percentile(latencies, 95) if latencies else 0,
            "concurrent_requests": concurrent_requests
        }
    
    async def measure_accuracy(
        self,
        test_data: List[Tuple[bytes, str]],
        model: str = "whisper-base"
    ) -> Dict:
        """
        Measure transcription accuracy using test data with ground truth
        """
        results = []
        
        for audio_data, ground_truth in test_data:
            try:
                files = {"audio": ("test.wav", audio_data, "audio/wav")}
                data = {"model": model}
                
                response = await self.client.post(
                    f"{self.api_endpoint}/api/v1/transcribe/sync",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    transcript = result.get("transcript", "")
                    confidence = result.get("confidence", 0.0)
                    
                    # Calculate WER (Word Error Rate)
                    wer = self.calculate_wer(ground_truth, transcript)
                    
                    # Calculate CER (Character Error Rate)
                    cer = self.calculate_cer(ground_truth, transcript)
                    
                    results.append({
                        "ground_truth": ground_truth,
                        "transcript": transcript,
                        "wer": wer,
                        "cer": cer,
                        "confidence": confidence
                    })
                    
            except Exception as e:
                logger.error(f"Accuracy test error: {e}")
                continue
        
        if not results:
            return {"error": "No successful transcriptions"}
        
        wer_scores = [r["wer"] for r in results]
        cer_scores = [r["cer"] for r in results]
        confidences = [r["confidence"] for r in results]
        
        return {
            "avg_wer": statistics.mean(wer_scores),
            "avg_cer": statistics.mean(cer_scores),
            "avg_confidence": statistics.mean(confidences),
            "median_wer": statistics.median(wer_scores),
            "median_cer": statistics.median(cer_scores),
            "samples": len(results),
            "detailed_results": results
        }
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate (WER)
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Simple edit distance calculation
        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j
        
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(
                        d[i-1][j] + 1,    # deletion
                        d[i][j-1] + 1,    # insertion
                        d[i-1][j-1] + 1   # substitution
                    )
        
        return d[len(ref_words)][len(hyp_words)] / len(ref_words) if ref_words else 0.0
    
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate (CER)
        """
        ref_chars = list(reference.lower())
        hyp_chars = list(hypothesis.lower())
        
        # Simple edit distance for characters
        d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
        
        for i in range(len(ref_chars) + 1):
            d[i][0] = i
        for j in range(len(hyp_chars) + 1):
            d[0][j] = j
        
        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                if ref_chars[i-1] == hyp_chars[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(
                        d[i-1][j] + 1,
                        d[i][j-1] + 1,
                        d[i-1][j-1] + 1
                    )
        
        return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars) if ref_chars else 0.0
    
    async def run_comprehensive_benchmark(
        self,
        models: List[str] = None,
        audio_durations: List[float] = None
    ) -> BenchmarkSuite:
        """
        Run comprehensive benchmark suite
        """
        if models is None:
            models = ["whisper-base", "whisper-small"]
        
        if audio_durations is None:
            audio_durations = [1.0, 5.0, 10.0, 30.0]
        
        logger.info("Starting comprehensive benchmark...")
        
        # System info
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
            "timestamp": time.time()
        }
        
        for model in models:
            for duration in audio_durations:
                logger.info(f"Testing {model} with {duration}s audio...")
                
                # Generate test audio
                audio_data = self.generate_test_audio(duration)
                
                # Measure system resources before test
                process = psutil.Process()
                cpu_before = process.cpu_percent()
                memory_before = process.memory_info().rss / (1024**2)  # MB
                
                # Latency test
                latency_results = await self.measure_latency(
                    audio_data, model, runs=5
                )
                
                # Measure system resources after test
                cpu_after = process.cpu_percent()
                memory_after = process.memory_info().rss / (1024**2)  # MB
                
                if "error" not in latency_results:
                    result = BenchmarkResult(
                        test_name=f"{model}_{duration}s",
                        latency_ms=latency_results["mean"],
                        accuracy_wer=None,  # Would need ground truth
                        accuracy_cer=None,
                        confidence=0.95,  # Placeholder
                        throughput_fps=1.0 / (latency_results["mean"] / 1000),
                        cpu_usage=(cpu_before + cpu_after) / 2,
                        memory_usage_mb=(memory_before + memory_after) / 2,
                        model_used=model,
                        audio_duration=duration,
                        timestamp=time.time()
                    )
                    
                    self.results.append(result)
        
        # Throughput tests
        logger.info("Running throughput tests...")
        throughput_results = await self.measure_throughput(
            concurrent_requests=10,
            total_requests=50,
            audio_duration=5.0
        )
        
        # Calculate summary statistics
        summary = self.calculate_summary()
        
        return BenchmarkSuite(
            results=self.results,
            summary=summary,
            config={
                "system_info": system_info,
                "throughput": throughput_results,
                "models_tested": models,
                "audio_durations": audio_durations
            }
        )
    
    def calculate_summary(self) -> Dict:
        """
        Calculate summary statistics from results
        """
        if not self.results:
            return {}
        
        latencies = [r.latency_ms for r in self.results]
        throughputs = [r.throughput_fps for r in self.results]
        
        return {
            "total_tests": len(self.results),
            "avg_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "avg_throughput_fps": statistics.mean(throughputs),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "latency_target_met": all(l < 150 for l in latencies),  # <150ms target
            "models_tested": list(set(r.model_used for r in self.results))
        }
    
    def generate_report(self, suite: BenchmarkSuite, output_file: str = None):
        """
        Generate benchmark report
        """
        if output_file is None:
            timestamp = int(time.time())
            output_file = self.output_dir / f"benchmark_report_{timestamp}.json"
        
        # Save detailed results
        with open(output_file, 'w') as f:
            json.dump(asdict(suite), f, indent=2, default=str)
        
        # Generate plots
        self.generate_plots(suite)
        
        # Print summary
        self.print_summary(suite)
        
        logger.info(f"Benchmark report saved to {output_file}")
    
    def generate_plots(self, suite: BenchmarkSuite):
        """
        Generate visualization plots
        """
        if not suite.results:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Latency by model
        models = list(set(r.model_used for r in suite.results))
        latency_by_model = {model: [] for model in models}
        
        for result in suite.results:
            latency_by_model[result.model_used].append(result.latency_ms)
        
        axes[0, 0].boxplot(latency_by_model.values(), labels=latency_by_model.keys())
        axes[0, 0].set_title('Latency by Model')
        axes[0, 0].set_ylabel('Latency (ms)')
        axes[0, 0].axhline(y=150, color='r', linestyle='--', label='Target (150ms)')
        axes[0, 0].legend()
        
        # Latency vs Audio Duration
        durations = [r.audio_duration for r in suite.results]
        latencies = [r.latency_ms for r in suite.results]
        
        axes[0, 1].scatter(durations, latencies, alpha=0.7)
        axes[0, 1].set_title('Latency vs Audio Duration')
        axes[0, 1].set_xlabel('Audio Duration (s)')
        axes[0, 1].set_ylabel('Latency (ms)')
        
        # Throughput by Model
        throughput_by_model = {model: [] for model in models}
        
        for result in suite.results:
            throughput_by_model[result.model_used].append(result.throughput_fps)
        
        axes[1, 0].bar(throughput_by_model.keys(), 
                      [statistics.mean(v) for v in throughput_by_model.values()])
        axes[1, 0].set_title('Average Throughput by Model')
        axes[1, 0].set_ylabel('Files per Second')
        
        # Resource Usage
        cpu_usage = [r.cpu_usage for r in suite.results]
        memory_usage = [r.memory_usage_mb for r in suite.results]
        
        axes[1, 1].scatter(cpu_usage, memory_usage, alpha=0.7)
        axes[1, 1].set_title('Resource Usage')
        axes[1, 1].set_xlabel('CPU Usage (%)')
        axes[1, 1].set_ylabel('Memory Usage (MB)')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = int(time.time())
        plot_file = self.output_dir / f"benchmark_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {plot_file}")
    
    def print_summary(self, suite: BenchmarkSuite):
        """
        Print benchmark summary
        """
        print("\n" + "="*60)
        print("VOICEFORGE STT BENCHMARK RESULTS")
        print("="*60)
        
        summary = suite.summary
        
        print(f"Total Tests: {summary.get('total_tests', 0)}")
        print(f"Models Tested: {', '.join(summary.get('models_tested', []))}")
        print()
        
        print("LATENCY RESULTS:")
        print(f"  Average: {summary.get('avg_latency_ms', 0):.1f}ms")
        print(f"  P95: {summary.get('p95_latency_ms', 0):.1f}ms")
        print(f"  Min: {summary.get('min_latency_ms', 0):.1f}ms")
        print(f"  Max: {summary.get('max_latency_ms', 0):.1f}ms")
        print(f"  Target (<150ms): {'✓ PASS' if summary.get('latency_target_met') else '✗ FAIL'}")
        print()
        
        print("THROUGHPUT RESULTS:")
        if 'throughput' in suite.config:
            throughput = suite.config['throughput']
            print(f"  Requests/sec: {throughput.get('requests_per_second', 0):.1f}")
            print(f"  Success Rate: {throughput.get('success_rate', 0)*100:.1f}%")
            print(f"  Concurrent Requests: {throughput.get('concurrent_requests', 0)}")
        print()
        
        print("SYSTEM INFO:")
        if 'system_info' in suite.config:
            sys_info = suite.config['system_info']
            print(f"  CPU Cores: {sys_info.get('cpu_count', 'N/A')}")
            print(f"  Memory: {sys_info.get('memory_total', 0):.1f} GB")
        print("="*60)


async def main():
    """
    Main benchmark runner
    """
    parser = argparse.ArgumentParser(description="VoiceForge STT Performance Benchmark")
    parser.add_argument("--endpoint", default="http://localhost:8000", help="API endpoint")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Output directory")
    parser.add_argument("--models", nargs="+", default=["whisper-base"], help="Models to test")
    parser.add_argument("--durations", nargs="+", type=float, default=[1.0, 5.0, 10.0], help="Audio durations to test")
    
    args = parser.parse_args()
    
    async with PerformanceBenchmark(
        api_endpoint=args.endpoint,
        api_key=args.api_key,
        output_dir=args.output_dir
    ) as benchmark:
        
        # Run comprehensive benchmark
        suite = await benchmark.run_comprehensive_benchmark(
            models=args.models,
            audio_durations=args.durations
        )
        
        # Generate report
        benchmark.generate_report(suite)


if __name__ == "__main__":
    asyncio.run(main())