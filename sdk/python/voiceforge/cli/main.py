#!/usr/bin/env python3
"""
VoiceForge CLI Tool

Command-line interface for the VoiceForge Speech-to-Text API.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional, List

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
except ImportError:
    print("CLI dependencies not installed. Install with: pip install voiceforge-python[cli]")
    sys.exit(1)

from ..client import VoiceForgeClient
from ..models import TranscriptionOptions, TranscriptionStatus
from ..exceptions import VoiceForgeError
from ..utils import (
    validate_audio_file,
    find_audio_files,
    export_to_srt,
    export_to_vtt,
    estimate_transcription_time,
)

app = typer.Typer(
    name="voiceforge",
    help="VoiceForge Speech-to-Text CLI",
    epilog="For more information, visit: https://docs.voiceforge.ai"
)

console = Console()


def get_client(api_key: Optional[str] = None) -> VoiceForgeClient:
    """Get VoiceForge client with API key validation"""
    if not api_key:
        api_key = os.getenv("VOICEFORGE_API_KEY")
    
    if not api_key:
        console.print("[red]Error:[/red] API key not found")
        console.print("Set VOICEFORGE_API_KEY environment variable or use --api-key option")
        raise typer.Exit(1)
    
    return VoiceForgeClient(api_key=api_key)


@app.command()
def transcribe(
    file_path: str = typer.Argument(..., help="Path to audio file"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="VoiceForge API key"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Audio language"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("txt", "--format", "-f", help="Output format (txt, json, srt, vtt)"),
    diarization: bool = typer.Option(False, "--diarization", "-d", help="Enable speaker diarization"),
    timestamps: bool = typer.Option(False, "--timestamps", "-t", help="Include word timestamps"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for completion"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Transcribe an audio file"""
    
    async def _transcribe():
        try:
            # Validate input file
            console.print(f"[blue]Validating:[/blue] {file_path}")
            file_info = validate_audio_file(file_path)
            
            if verbose:
                console.print(f"File size: {file_info['size']:,} bytes")
                console.print(f"Format: {file_info['format'].value}")
            
            # Create client
            client = get_client(api_key)
            
            # Create transcription options
            options = TranscriptionOptions(
                model=model,
                language=language,
                enable_diarization=diarization,
                enable_word_timestamps=timestamps,
            )
            
            # Estimate processing time
            estimated_time = estimate_transcription_time(
                file_info['size'],
                model_type=model or "whisper-base"
            )
            
            if verbose:
                console.print(f"Estimated processing time: {estimated_time:.1f} seconds")
            
            # Start transcription
            console.print("[blue]Starting transcription...[/blue]")
            
            if wait:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task("Transcribing...", total=100)
                    
                    def progress_callback(percent: float):
                        progress.update(task, completed=int(percent * 100))
                    
                    async with client:
                        job = await client.transcribe_file(
                            file_path,
                            options=options,
                            wait_for_completion=True,
                            progress_callback=progress_callback,
                        )
            else:
                async with client:
                    job = await client.transcribe_file(
                        file_path,
                        options=options,
                        wait_for_completion=False,
                    )
            
            # Handle output
            if not wait:
                console.print(f"[green]Job created:[/green] {job.id}")
                console.print(f"Status: {job.status.value}")
                return
            
            if job.status == TranscriptionStatus.COMPLETED:
                console.print("[green]Transcription completed![/green]")
                
                # Format output
                if format == "json":
                    output_data = job.model_dump(mode="json")
                    output_text = json.dumps(output_data, indent=2, default=str)
                elif format == "srt" and job.words:
                    word_dicts = [word.model_dump() for word in job.words]
                    output_text = export_to_srt(word_dicts)
                elif format == "vtt" and job.words:
                    word_dicts = [word.model_dump() for word in job.words]
                    output_text = export_to_vtt(word_dicts)
                else:  # txt format
                    output_text = job.transcript or ""
                
                # Save or print output
                if output:
                    Path(output).write_text(output_text, encoding="utf-8")
                    console.print(f"[green]Output saved to:[/green] {output}")
                else:
                    console.print("\n[bold]Transcript:[/bold]")
                    console.print(Panel(output_text, border_style="green"))
                
                # Show additional info
                if verbose and job.confidence:
                    console.print(f"Confidence: {job.confidence:.2%}")
                if verbose and job.processing_time:
                    console.print(f"Processing time: {job.processing_time:.1f}s")
                if verbose and job.detected_language:
                    console.print(f"Detected language: {job.detected_language.name} ({job.detected_language.confidence:.2%})")
                
            else:
                console.print(f"[red]Transcription failed:[/red] {job.error}")
                raise typer.Exit(1)
        
        except VoiceForgeError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")
            if verbose:
                import traceback
                console.print(traceback.format_exc())
            raise typer.Exit(1)
    
    asyncio.run(_transcribe())


@app.command()
def batch(
    directory: str = typer.Argument(..., help="Directory containing audio files"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="VoiceForge API key"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Audio language"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    format: str = typer.Option("txt", "--format", "-f", help="Output format"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r", help="Search subdirectories"),
    max_files: int = typer.Option(50, "--max-files", help="Maximum files to process"),
    diarization: bool = typer.Option(False, "--diarization", "-d", help="Enable speaker diarization"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Batch transcribe audio files in a directory"""
    
    async def _batch_transcribe():
        try:
            # Find audio files
            console.print(f"[blue]Scanning:[/blue] {directory}")
            audio_files = find_audio_files(Path(directory), recursive=recursive)
            
            if not audio_files:
                console.print("[yellow]No audio files found[/yellow]")
                return
            
            if len(audio_files) > max_files:
                audio_files = audio_files[:max_files]
                console.print(f"[yellow]Limiting to {max_files} files[/yellow]")
            
            console.print(f"Found {len(audio_files)} audio files")
            
            # Create output directory
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Create client
            client = get_client(api_key)
            
            # Create options
            options = TranscriptionOptions(
                model=model,
                language=language,
                enable_diarization=diarization,
                enable_word_timestamps=format in ["srt", "vtt"],
            )
            
            # Process files
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing files...", total=len(audio_files))
                
                async with client:
                    for i, file_path in enumerate(audio_files):
                        try:
                            progress.update(task, description=f"Processing {file_path.name}")
                            
                            # Transcribe file
                            job = await client.transcribe_file(
                                file_path,
                                options=options,
                                wait_for_completion=True,
                            )
                            
                            if job.status == TranscriptionStatus.COMPLETED:
                                # Save output
                                if output_dir:
                                    output_file = Path(output_dir) / f"{file_path.stem}.{format}"
                                    
                                    if format == "json":
                                        output_data = job.model_dump(mode="json")
                                        output_text = json.dumps(output_data, indent=2, default=str)
                                    elif format == "srt" and job.words:
                                        word_dicts = [word.model_dump() for word in job.words]
                                        output_text = export_to_srt(word_dicts)
                                    elif format == "vtt" and job.words:
                                        word_dicts = [word.model_dump() for word in job.words]
                                        output_text = export_to_vtt(word_dicts)
                                    else:  # txt
                                        output_text = job.transcript or ""
                                    
                                    output_file.write_text(output_text, encoding="utf-8")
                                
                                if verbose:
                                    console.print(f"[green]✓[/green] {file_path.name}")
                            else:
                                console.print(f"[red]✗[/red] {file_path.name}: {job.error}")
                        
                        except Exception as e:
                            console.print(f"[red]✗[/red] {file_path.name}: {e}")
                        
                        progress.advance(task)
            
            console.print("[green]Batch processing completed![/green]")
        
        except VoiceForgeError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")
            if verbose:
                import traceback
                console.print(traceback.format_exc())
            raise typer.Exit(1)
    
    asyncio.run(_batch_transcribe())


@app.command()
def jobs(
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="VoiceForge API key"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of jobs to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List transcription jobs"""
    
    async def _list_jobs():
        try:
            client = get_client(api_key)
            
            status_filter = None
            if status:
                try:
                    status_filter = TranscriptionStatus(status.lower())
                except ValueError:
                    console.print(f"[red]Invalid status:[/red] {status}")
                    console.print("Valid statuses: pending, processing, completed, failed, cancelled")
                    raise typer.Exit(1)
            
            async with client:
                result = await client.list_jobs(status=status_filter, limit=limit)
            
            if json_output:
                jobs_data = [job.model_dump(mode="json") for job in result.items]
                console.print(json.dumps(jobs_data, indent=2, default=str))
                return
            
            if not result.items:
                console.print("[yellow]No jobs found[/yellow]")
                return
            
            # Create table
            table = Table(title="Transcription Jobs")
            table.add_column("Job ID", style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Filename")
            table.add_column("Created", style="dim")
            table.add_column("Duration")
            table.add_column("Model")
            
            for job in result.items:
                status_style = {
                    TranscriptionStatus.COMPLETED: "green",
                    TranscriptionStatus.FAILED: "red",
                    TranscriptionStatus.PROCESSING: "yellow",
                    TranscriptionStatus.PENDING: "blue",
                    TranscriptionStatus.CANCELLED: "dim",
                }.get(job.status, "")
                
                duration = f"{job.audio_duration:.1f}s" if job.audio_duration else "-"
                
                table.add_row(
                    job.id[:8] + "...",
                    f"[{status_style}]{job.status.value}[/{status_style}]",
                    job.filename,
                    job.created_at.strftime("%Y-%m-%d %H:%M"),
                    duration,
                    job.model_used or "-",
                )
            
            console.print(table)
            console.print(f"\nShowing {len(result.items)} of {result.total_count} jobs")
        
        except VoiceForgeError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
    
    asyncio.run(_list_jobs())


@app.command()
def job(
    job_id: str = typer.Argument(..., help="Job ID"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="VoiceForge API key"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    save_transcript: Optional[str] = typer.Option(None, "--save", "-s", help="Save transcript to file"),
):
    """Get details of a specific transcription job"""
    
    async def _get_job():
        try:
            client = get_client(api_key)
            
            async with client:
                job = await client.get_job(job_id)
            
            if json_output:
                console.print(json.dumps(job.model_dump(mode="json"), indent=2, default=str))
                return
            
            # Display job details
            console.print(f"[bold]Job ID:[/bold] {job.id}")
            console.print(f"[bold]Status:[/bold] {job.status.value}")
            console.print(f"[bold]Filename:[/bold] {job.filename}")
            console.print(f"[bold]Created:[/bold] {job.created_at}")
            
            if job.model_used:
                console.print(f"[bold]Model:[/bold] {job.model_used}")
            if job.language_code:
                console.print(f"[bold]Language:[/bold] {job.language_code}")
            if job.audio_duration:
                console.print(f"[bold]Duration:[/bold] {job.audio_duration:.1f}s")
            if job.processing_time:
                console.print(f"[bold]Processing Time:[/bold] {job.processing_time:.1f}s")
            if job.confidence:
                console.print(f"[bold]Confidence:[/bold] {job.confidence:.2%}")
            
            if job.transcript:
                console.print("\n[bold]Transcript:[/bold]")
                console.print(Panel(job.transcript, border_style="green"))
                
                if save_transcript:
                    Path(save_transcript).write_text(job.transcript, encoding="utf-8")
                    console.print(f"[green]Transcript saved to:[/green] {save_transcript}")
            
            if job.error:
                console.print(f"\n[red]Error:[/red] {job.error}")
        
        except VoiceForgeError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
    
    asyncio.run(_get_job())


@app.command()
def models(
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="VoiceForge API key"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List available transcription models"""
    
    async def _list_models():
        try:
            client = get_client(api_key)
            
            async with client:
                models = await client.list_models()
            
            if json_output:
                models_data = [model.model_dump(mode="json") for model in models]
                console.print(json.dumps(models_data, indent=2, default=str))
                return
            
            # Create table
            table = Table(title="Available Models")
            table.add_column("Model ID", style="cyan")
            table.add_column("Name", style="bold")
            table.add_column("Type")
            table.add_column("Languages")
            table.add_column("GPU", justify="center")
            table.add_column("Available", justify="center")
            
            for model in models:
                languages = ", ".join(model.supported_languages[:3])
                if len(model.supported_languages) > 3:
                    languages += f" (+{len(model.supported_languages) - 3})"
                
                table.add_row(
                    model.model_id,
                    model.name,
                    model.type,
                    languages,
                    "✓" if model.requires_gpu else "-",
                    "[green]✓[/green]" if model.is_available else "[red]✗[/red]",
                )
            
            console.print(table)
        
        except VoiceForgeError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
    
    asyncio.run(_list_models())


@app.command()
def configure(
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="VoiceForge API key"),
):
    """Configure VoiceForge CLI"""
    
    config_dir = Path.home() / ".voiceforge"
    config_file = config_dir / "config.json"
    
    # Load existing config
    config = {}
    if config_file.exists():
        config = json.loads(config_file.read_text())
    
    console.print("[bold]VoiceForge CLI Configuration[/bold]")
    
    # Get API key
    if not api_key:
        current_key = config.get("api_key", "")
        api_key = Prompt.ask(
            "API Key",
            default=current_key,
            password=True if not current_key else False
        )
    
    # Test API key
    if api_key:
        console.print("Testing API key...")
        try:
            client = VoiceForgeClient(api_key=api_key)
            asyncio.run(client.get_user_info())
            console.print("[green]✓ API key is valid[/green]")
            
            config["api_key"] = api_key
            
        except Exception as e:
            console.print(f"[red]✗ API key test failed:[/red] {e}")
            if not Confirm.ask("Save anyway?"):
                raise typer.Exit(1)
    
    # Save config
    config_dir.mkdir(exist_ok=True)
    config_file.write_text(json.dumps(config, indent=2))
    
    console.print(f"[green]Configuration saved to:[/green] {config_file}")


@app.command()
def version():
    """Show version information"""
    from .. import __version__
    console.print(f"VoiceForge Python SDK v{__version__}")


if __name__ == "__main__":
    app()