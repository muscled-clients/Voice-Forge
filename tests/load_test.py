"""
Performance Load Testing for VoiceForge API
Tests API performance under various load conditions
"""
import asyncio
import aiohttp
import time
import json
import statistics
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
import os
import random
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TestResult:
    """Store individual test result"""
    request_id: int
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: datetime
    success: bool
    error: str = None
    response_size: int = 0

class LoadTester:
    """Comprehensive load testing for VoiceForge API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        self.test_files = self._prepare_test_files()
        self.auth_token = None
        
    def _prepare_test_files(self) -> List[str]:
        """Prepare test audio files"""
        test_dir = Path("tests/audio_samples")
        test_dir.mkdir(exist_ok=True)
        
        # Create dummy test files if they don't exist
        test_files = []
        for i in range(5):
            file_path = test_dir / f"test_audio_{i}.wav"
            if not file_path.exists():
                # Create a dummy WAV file for testing
                self._create_dummy_audio(file_path)
            test_files.append(str(file_path))
            
        return test_files
        
    def _create_dummy_audio(self, file_path: Path):
        """Create dummy audio file for testing"""
        import wave
        import struct
        
        # Generate 10 seconds of silence
        sample_rate = 16000
        duration = 10
        num_samples = sample_rate * duration
        
        with wave.open(str(file_path), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Write silence (zeros)
            for _ in range(num_samples):
                wav_file.writeframes(struct.pack('h', 0))
                
    async def authenticate(self):
        """Get authentication token"""
        async with aiohttp.ClientSession() as session:
            # Register test user
            register_data = {
                "email": f"loadtest_{random.randint(1000, 9999)}@test.com",
                "password": "TestPassword123!",
                "full_name": "Load Tester",
                "company": "Test Corp"
            }
            
            async with session.post(
                f"{self.base_url}/api/v1/auth/register",
                json=register_data
            ) as response:
                if response.status == 201:
                    # Login to get token
                    login_data = {
                        "email": register_data["email"],
                        "password": register_data["password"]
                    }
                    
                    async with session.post(
                        f"{self.base_url}/api/v1/auth/login",
                        json=login_data
                    ) as login_response:
                        if login_response.status == 200:
                            data = await login_response.json()
                            self.auth_token = data["access_token"]
                            print(f"✓ Authenticated successfully")
                            
    async def test_endpoint(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        method: str = "GET",
        data: Dict = None,
        files: Dict = None,
        request_id: int = 0
    ) -> TestResult:
        """Test a single endpoint"""
        
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        if self.auth_token and "/protected" in endpoint:
            headers["Authorization"] = f"Bearer {self.auth_token}"
            
        start_time = time.time()
        
        try:
            if method == "GET":
                async with session.get(url, headers=headers) as response:
                    await response.text()
                    response_time = time.time() - start_time
                    return TestResult(
                        request_id=request_id,
                        endpoint=endpoint,
                        method=method,
                        status_code=response.status,
                        response_time=response_time * 1000,  # Convert to ms
                        timestamp=datetime.now(),
                        success=response.status < 400,
                        response_size=len(await response.read())
                    )
                    
            elif method == "POST":
                if files:
                    # File upload
                    form_data = aiohttp.FormData()
                    for key, file_path in files.items():
                        form_data.add_field(
                            key,
                            open(file_path, 'rb'),
                            filename=os.path.basename(file_path)
                        )
                        
                    async with session.post(
                        url,
                        data=form_data,
                        headers=headers
                    ) as response:
                        await response.text()
                        response_time = time.time() - start_time
                        return TestResult(
                            request_id=request_id,
                            endpoint=endpoint,
                            method=method,
                            status_code=response.status,
                            response_time=response_time * 1000,
                            timestamp=datetime.now(),
                            success=response.status < 400,
                            response_size=len(await response.read())
                        )
                else:
                    # JSON data
                    async with session.post(
                        url,
                        json=data,
                        headers=headers
                    ) as response:
                        await response.text()
                        response_time = time.time() - start_time
                        return TestResult(
                            request_id=request_id,
                            endpoint=endpoint,
                            method=method,
                            status_code=response.status,
                            response_time=response_time * 1000,
                            timestamp=datetime.now(),
                            success=response.status < 400,
                            response_size=len(await response.read())
                        )
                        
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                request_id=request_id,
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time=response_time * 1000,
                timestamp=datetime.now(),
                success=False,
                error=str(e)
            )
            
    async def run_concurrent_requests(
        self,
        endpoint: str,
        method: str,
        num_requests: int,
        data: Dict = None,
        files: Dict = None
    ) -> List[TestResult]:
        """Run multiple concurrent requests"""
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_requests):
                task = self.test_endpoint(
                    session,
                    endpoint,
                    method,
                    data,
                    files,
                    request_id=i
                )
                tasks.append(task)
                
            results = await asyncio.gather(*tasks)
            return results
            
    async def stress_test(self):
        """Run comprehensive stress test"""
        
        print("\n" + "="*60)
        print("🚀 VoiceForge API Load Testing")
        print("="*60)
        
        # Test scenarios
        test_scenarios = [
            {
                "name": "Health Check",
                "endpoint": "/health",
                "method": "GET",
                "concurrent": [1, 10, 50, 100],
                "iterations": 100
            },
            {
                "name": "API Status",
                "endpoint": "/api/v1/stats",
                "method": "GET",
                "concurrent": [1, 10, 50, 100],
                "iterations": 100
            },
            {
                "name": "Transcription",
                "endpoint": "/api/v1/transcribe",
                "method": "POST",
                "concurrent": [1, 5, 10, 20],
                "iterations": 20,
                "files": {"file": random.choice(self.test_files)}
            }
        ]
        
        all_results = []
        
        for scenario in test_scenarios:
            print(f"\n📊 Testing: {scenario['name']}")
            print("-" * 40)
            
            for concurrent_users in scenario["concurrent"]:
                print(f"\n  Users: {concurrent_users}")
                
                scenario_results = []
                iterations = scenario["iterations"] // concurrent_users
                
                for i in range(iterations):
                    results = await self.run_concurrent_requests(
                        scenario["endpoint"],
                        scenario["method"],
                        concurrent_users,
                        data=scenario.get("data"),
                        files=scenario.get("files")
                    )
                    
                    scenario_results.extend(results)
                    all_results.extend(results)
                    
                    # Progress indicator
                    if (i + 1) % 10 == 0:
                        print(f"    Completed: {(i + 1) * concurrent_users} requests")
                        
                # Calculate metrics for this scenario
                self._print_scenario_metrics(scenario_results, concurrent_users)
                
        self.results = all_results
        return all_results
        
    def _print_scenario_metrics(self, results: List[TestResult], concurrent_users: int):
        """Print metrics for a test scenario"""
        
        response_times = [r.response_time for r in results if r.success]
        success_count = sum(1 for r in results if r.success)
        
        if response_times:
            metrics = {
                "Total Requests": len(results),
                "Successful": success_count,
                "Failed": len(results) - success_count,
                "Success Rate": f"{(success_count / len(results) * 100):.1f}%",
                "Avg Response": f"{statistics.mean(response_times):.2f}ms",
                "Min Response": f"{min(response_times):.2f}ms",
                "Max Response": f"{max(response_times):.2f}ms",
                "P50 (Median)": f"{statistics.median(response_times):.2f}ms",
                "P95": f"{np.percentile(response_times, 95):.2f}ms",
                "P99": f"{np.percentile(response_times, 99):.2f}ms",
                "Throughput": f"{len(results) / (sum(response_times) / 1000):.1f} req/s"
            }
            
            for key, value in metrics.items():
                print(f"    {key}: {value}")
                
    def generate_report(self):
        """Generate comprehensive performance report"""
        
        if not self.results:
            print("No test results to report")
            return
            
        print("\n" + "="*60)
        print("📈 Performance Test Report")
        print("="*60)
        
        # Convert results to DataFrame
        df = pd.DataFrame([
            {
                "request_id": r.request_id,
                "endpoint": r.endpoint,
                "method": r.method,
                "status_code": r.status_code,
                "response_time": r.response_time,
                "timestamp": r.timestamp,
                "success": r.success
            }
            for r in self.results
        ])
        
        # Overall statistics
        print("\n🎯 Overall Statistics")
        print("-" * 40)
        
        total_requests = len(df)
        successful_requests = df["success"].sum()
        failed_requests = total_requests - successful_requests
        success_rate = (successful_requests / total_requests) * 100
        
        print(f"Total Requests: {total_requests}")
        print(f"Successful: {successful_requests}")
        print(f"Failed: {failed_requests}")
        print(f"Success Rate: {success_rate:.2f}%")
        
        # Response time statistics
        response_times = df[df["success"]]["response_time"]
        
        print(f"\nResponse Times:")
        print(f"  Average: {response_times.mean():.2f}ms")
        print(f"  Median: {response_times.median():.2f}ms")
        print(f"  Min: {response_times.min():.2f}ms")
        print(f"  Max: {response_times.max():.2f}ms")
        print(f"  Std Dev: {response_times.std():.2f}ms")
        
        # Percentiles
        percentiles = [50, 75, 90, 95, 99]
        print(f"\nPercentiles:")
        for p in percentiles:
            value = np.percentile(response_times, p)
            print(f"  P{p}: {value:.2f}ms")
            
        # Per endpoint statistics
        print("\n📊 Per Endpoint Statistics")
        print("-" * 40)
        
        for endpoint in df["endpoint"].unique():
            endpoint_df = df[df["endpoint"] == endpoint]
            endpoint_success = endpoint_df["success"].sum()
            endpoint_total = len(endpoint_df)
            endpoint_rate = (endpoint_success / endpoint_total) * 100
            
            print(f"\n{endpoint}:")
            print(f"  Requests: {endpoint_total}")
            print(f"  Success Rate: {endpoint_rate:.2f}%")
            
            if endpoint_success > 0:
                endpoint_times = endpoint_df[endpoint_df["success"]]["response_time"]
                print(f"  Avg Response: {endpoint_times.mean():.2f}ms")
                print(f"  P95: {np.percentile(endpoint_times, 95):.2f}ms")
                
        # Generate visualizations
        self._generate_charts(df)
        
        # Save detailed report
        report_path = f"load_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(report_path, index=False)
        print(f"\n💾 Detailed report saved to: {report_path}")
        
    def _generate_charts(self, df: pd.DataFrame):
        """Generate performance visualization charts"""
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Response time distribution
            axes[0, 0].hist(
                df[df["success"]]["response_time"],
                bins=50,
                edgecolor='black'
            )
            axes[0, 0].set_title("Response Time Distribution")
            axes[0, 0].set_xlabel("Response Time (ms)")
            axes[0, 0].set_ylabel("Frequency")
            
            # Response time over time
            axes[0, 1].plot(
                df["timestamp"],
                df["response_time"],
                alpha=0.6
            )
            axes[0, 1].set_title("Response Time Over Time")
            axes[0, 1].set_xlabel("Time")
            axes[0, 1].set_ylabel("Response Time (ms)")
            
            # Success rate by endpoint
            endpoint_stats = df.groupby("endpoint")["success"].mean() * 100
            axes[1, 0].bar(
                range(len(endpoint_stats)),
                endpoint_stats.values
            )
            axes[1, 0].set_xticks(range(len(endpoint_stats)))
            axes[1, 0].set_xticklabels(
                [e.split("/")[-1] for e in endpoint_stats.index],
                rotation=45
            )
            axes[1, 0].set_title("Success Rate by Endpoint")
            axes[1, 0].set_ylabel("Success Rate (%)")
            
            # Percentile chart
            percentiles = [50, 75, 90, 95, 99]
            percentile_values = [
                np.percentile(df[df["success"]]["response_time"], p)
                for p in percentiles
            ]
            axes[1, 1].bar(
                [f"P{p}" for p in percentiles],
                percentile_values
            )
            axes[1, 1].set_title("Response Time Percentiles")
            axes[1, 1].set_ylabel("Response Time (ms)")
            
            plt.tight_layout()
            chart_path = f"load_test_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path)
            print(f"\n📊 Charts saved to: {chart_path}")
            
        except ImportError:
            print("\n⚠️ Matplotlib not installed. Skipping chart generation.")
            
async def main():
    """Run load testing"""
    
    tester = LoadTester()
    
    # Authenticate first
    await tester.authenticate()
    
    # Run stress test
    await tester.stress_test()
    
    # Generate report
    tester.generate_report()
    
    print("\n✅ Load testing complete!")
    
if __name__ == "__main__":
    asyncio.run(main())