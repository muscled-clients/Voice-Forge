#!/usr/bin/env python
"""
VoiceForge Python SDK Setup
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read version
version_file = this_directory / "voiceforge" / "__version__.py"
version_dict = {}
exec(version_file.read_text(), version_dict)
version = version_dict["__version__"]

setup(
    name="voiceforge-python",
    version=version,
    author="VoiceForge Team",
    author_email="support@voiceforge.ai",
    description="Official Python SDK for VoiceForge Speech-to-Text API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/voiceforge/voiceforge-python",
    project_urls={
        "Documentation": "https://docs.voiceforge.ai/python-sdk",
        "Source": "https://github.com/voiceforge/voiceforge-python",
        "Bug Tracker": "https://github.com/voiceforge/voiceforge-python/issues",
        "Homepage": "https://voiceforge.ai",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "httpx>=0.24.0",
        "pydantic>=2.0.0",
        "websockets>=11.0.0",
        "aiofiles>=23.0.0",
        "typing-extensions>=4.0.0",
        "python-dateutil>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.11.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "audio": [
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
            "pydub>=0.25.0",
        ],
        "cli": [
            "click>=8.0.0",
            "rich>=13.0.0",
            "typer>=0.9.0",
        ],
        "all": [
            "librosa>=0.10.0",
            "soundfile>=0.12.0", 
            "pydub>=0.25.0",
            "click>=8.0.0",
            "rich>=13.0.0",
            "typer>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "voiceforge=voiceforge.cli.main:app",
        ],
    },
    keywords=[
        "speech-to-text",
        "stt",
        "transcription",
        "voice recognition",
        "audio processing",
        "ai",
        "machine learning",
        "api",
        "sdk",
    ],
    include_package_data=True,
    zip_safe=False,
)