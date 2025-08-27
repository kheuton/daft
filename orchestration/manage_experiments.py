#!/usr/bin/env python3
"""
DAFT Experiment Management Utilities

This script provides utilities for managing experiments:
- List running experiments
- Check experiment status
- Cancel experiments
- Clean up old experiments

Usage:
    python manage_experiments.py status
    python manage_experiments.py list
    python manage_experiments.py cancel <experiment_id>
    python manage_experiments.py cleanup --older-than 7  # days
"""

import argparse
import subprocess
import yaml
import os
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re

class ExperimentManager:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.logs_dir = self.base_dir / "logs"
        self.scripts_dir = self.base_dir / "scripts"
        self.configs_dir = self.base_dir / "configs"
        self.outputs_dir = self.base_dir / "outputs"
    
    def get_running_jobs(self) -> List[Dict]:
        """Get all running SLURM jobs for the current user."""
        try:
            result = subprocess.run(
                ["squeue", "-u", os.getenv("USER"), "--format=%i,%j,%t,%M,%N,%R"],
                capture_output=True, text=True, check=True
            )
            
            jobs = []
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 6:
                        jobs.append({
                            'job_id': parts[0],
                            'name': parts[1],
                            'state': parts[2],
                            'time': parts[3],
                            'nodes': parts[4],
                            'reason': parts[5]
                        })
            
            return jobs
        except subprocess.CalledProcessError:
            return []
    
    def get_experiment_jobs(self) -> Dict[str, List[Dict]]:
        """Group running jobs by experiment ID."""
        jobs = self.get_running_jobs()
        experiments = {}
        
        for job in jobs:
            # Extract experiment ID from job name
            # Expected format: stepname_experimentname_timestamp_taskid
            match = re.search(r'_([a-zA-Z0-9_]+_\d{8}_\d{6})(?:_task\d+)?$', job['name'])
            if match:
                exp_id = match.group(1)
                if exp_id not in experiments:
                    experiments[exp_id] = []
                experiments[exp_id].append(job)
        
        return experiments
    
    def get_experiment_directories(self) -> List[Path]:
        """Get all experiment directories."""
        if not self.logs_dir.exists():
            return []
        
        return [d for d in self.logs_dir.iterdir() if d.is_dir()]
    
    def get_experiment_metadata(self, exp_dir: Path) -> Optional[Dict]:
        """Get metadata for an experiment."""
        # Look for experiment metadata in various locations
        metadata_locations = [
            exp_dir / "experiment_metadata.json",
            self.outputs_dir / exp_dir.name / "experiment_metadata.json"
        ]
        
        for location in metadata_locations:
            if location.exists():
                try:
                    with open(location, 'r') as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def list_experiments(self):
        """List all experiments with their status."""
        print("🔬 DAFT Experiments Overview")
        print("=" * 60)
        
        # Get running jobs grouped by experiment
        running_experiments = self.get_experiment_jobs()
        
        # Get all experiment directories
        exp_dirs = self.get_experiment_directories()
        
        if not exp_dirs and not running_experiments:
            print("No experiments found.")
            return
        
        # Combine running and historical experiments
        all_experiments = set()
        for exp_id in running_experiments.keys():
            all_experiments.add(exp_id)
        for exp_dir in exp_dirs:
            all_experiments.add(exp_dir.name)
        
        for exp_id in sorted(all_experiments):
            print(f"\n📊 Experiment: {exp_id}")
            
            # Check if running
            if exp_id in running_experiments:
                jobs = running_experiments[exp_id]
                print(f"   Status: 🟢 RUNNING ({len(jobs)} jobs)")
                
                for job in jobs:
                    step = self.extract_step_from_job_name(job['name'])
                    print(f"     - {step}: {job['job_id']} ({job['state']}) - {job['time']}")
            else:
                # Check if completed
                exp_dir = self.logs_dir / exp_id
                if exp_dir.exists():
                    status = self.check_experiment_completion(exp_dir)
                    if status == "completed":
                        print("   Status: ✅ COMPLETED")
                    elif status == "failed":
                        print("   Status: ❌ FAILED")
                    else:
                        print("   Status: ⏸️  STOPPED/UNKNOWN")
                else:
                    print("   Status: 🔍 NO LOGS FOUND")
            
            # Show metadata if available
            metadata = None
            for exp_dir in exp_dirs:
                if exp_dir.name == exp_id:
                    metadata = self.get_experiment_metadata(exp_dir)
                    break
            
            if metadata:
                if 'training_config' in metadata:
                    config = metadata['training_config']
                    print(f"     Model: {config.get('model_name', 'Unknown')}")
                    print(f"     Dataset: {config.get('train_dataset_name', 'Unknown')}")
                if 'completed_at' in metadata:
                    print(f"     Completed: {metadata['completed_at']}")
    
    def extract_step_from_job_name(self, job_name: str) -> str:
        """Extract the pipeline step from a job name."""
        if 'train' in job_name.lower():
            return "training"
        elif 'upload' in job_name.lower():
            return "upload"
        elif 'eval' in job_name.lower():
            return "evaluation"
        elif 'process' in job_name.lower():
            return "processing"
        else:
            return "unknown"
    
    def check_experiment_completion(self, exp_dir: Path) -> str:
        """Check if an experiment completed successfully."""
        # Look for completion indicators in log files
        log_files = list(exp_dir.glob("*.out")) + list(exp_dir.glob("*.err"))
        
        has_training_complete = False
        has_upload_complete = False
        has_eval_complete = False
        has_processing_complete = False
        has_errors = False
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    
                    if "Training completed for experiment" in content:
                        has_training_complete = True
                    if "Upload completed for experiment" in content:
                        has_upload_complete = True
                    if "Evaluation task" in content and "completed for experiment" in content:
                        has_eval_complete = True
                    if "Results processing completed for experiment" in content:
                        has_processing_complete = True
                    
                    # Check for common error patterns
                    if any(error in content.lower() for error in ["error", "failed", "killed", "cancelled"]):
                        has_errors = True
                        
            except Exception:
                continue
        
        if has_processing_complete:
            return "completed"
        elif has_errors:
            return "failed"
        else:
            return "incomplete"
    
    def show_experiment_status(self, experiment_id: str = None):
        """Show detailed status for a specific experiment or all experiments."""
        if experiment_id:
            print(f"🔍 Status for experiment: {experiment_id}")
            print("=" * 50)
            
            # Check running jobs
            running_experiments = self.get_experiment_jobs()
            if experiment_id in running_experiments:
                jobs = running_experiments[experiment_id]
                print(f"Running jobs ({len(jobs)}):")
                for job in jobs:
                    step = self.extract_step_from_job_name(job['name'])
                    print(f"  {step:12} | {job['job_id']:8} | {job['state']:10} | {job['time']:10} | {job['nodes']}")
            
            # Check logs
            exp_dir = self.logs_dir / experiment_id
            if exp_dir.exists():
                print(f"\nLog files in {exp_dir}:")
                for log_file in sorted(exp_dir.glob("*")):
                    size = log_file.stat().st_size
                    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    print(f"  {log_file.name:30} | {size:8} bytes | {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Check outputs
            output_dir = self.outputs_dir / experiment_id
            if output_dir.exists():
                print(f"\nOutput files in {output_dir}:")
                for output_file in sorted(output_dir.rglob("*")):
                    if output_file.is_file():
                        rel_path = output_file.relative_to(output_dir)
                        size = output_file.stat().st_size
                        print(f"  {str(rel_path):50} | {size:8} bytes")
        else:
            self.list_experiments()
    
    def cancel_experiment(self, experiment_id: str):
        """Cancel all jobs for an experiment."""
        running_experiments = self.get_experiment_jobs()
        
        if experiment_id not in running_experiments:
            print(f"❌ No running jobs found for experiment: {experiment_id}")
            return
        
        jobs = running_experiments[experiment_id]
        print(f"🛑 Cancelling {len(jobs)} jobs for experiment: {experiment_id}")
        
        success_count = 0
        for job in jobs:
            try:
                subprocess.run(["scancel", job['job_id']], check=True)
                print(f"  ✅ Cancelled job {job['job_id']} ({self.extract_step_from_job_name(job['name'])})")
                success_count += 1
            except subprocess.CalledProcessError:
                print(f"  ❌ Failed to cancel job {job['job_id']}")
        
        print(f"🎯 Successfully cancelled {success_count}/{len(jobs)} jobs")
    
    def cleanup_old_experiments(self, days: int = 7, dry_run: bool = False):
        """Clean up experiment files older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        print(f"🧹 Cleaning up experiments older than {days} days (before {cutoff_date.strftime('%Y-%m-%d')})")
        
        if dry_run:
            print("🔍 DRY RUN - no files will be deleted")
        
        # Clean up log directories
        deleted_count = 0
        preserved_count = 0
        
        for exp_dir in self.get_experiment_directories():
            dir_mtime = datetime.fromtimestamp(exp_dir.stat().st_mtime)
            
            if dir_mtime < cutoff_date:
                # Check if experiment is still running
                running_experiments = self.get_experiment_jobs()
                if exp_dir.name in running_experiments:
                    print(f"  ⏸️  Skipping {exp_dir.name} (still running)")
                    preserved_count += 1
                    continue
                
                print(f"  🗑️  Deleting {exp_dir.name} (modified: {dir_mtime.strftime('%Y-%m-%d %H:%M:%S')})")
                
                if not dry_run:
                    import shutil
                    shutil.rmtree(exp_dir)
                
                deleted_count += 1
            else:
                preserved_count += 1
        
        # Clean up old scripts
        script_files = list(self.scripts_dir.glob("*_20*.slurm"))
        for script_file in script_files:
            file_mtime = datetime.fromtimestamp(script_file.stat().st_mtime)
            if file_mtime < cutoff_date:
                print(f"  📄 Deleting old script: {script_file.name}")
                if not dry_run:
                    script_file.unlink()
        
        print(f"✅ Cleanup complete: {deleted_count} experiments deleted, {preserved_count} preserved")
    
    def watch_experiment(self, experiment_id: str):
        """Watch an experiment's progress in real-time."""
        print(f"👀 Watching experiment: {experiment_id}")
        print("Press Ctrl+C to stop watching")
        print("=" * 50)
        
        try:
            while True:
                os.system('clear')  # Clear screen
                print(f"👀 Watching experiment: {experiment_id}")
                print(f"⏰ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 50)
                
                self.show_experiment_status(experiment_id)
                
                time.sleep(30)  # Update every 30 seconds
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopped watching")

def main():
    parser = argparse.ArgumentParser(description="Manage DAFT experiments")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all experiments')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show experiment status')
    status_parser.add_argument('experiment_id', nargs='?', help='Specific experiment ID')
    
    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel experiment')
    cancel_parser.add_argument('experiment_id', help='Experiment ID to cancel')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old experiments')
    cleanup_parser.add_argument('--older-than', type=int, default=7, 
                               help='Delete experiments older than N days (default: 7)')
    cleanup_parser.add_argument('--dry-run', action='store_true',
                               help='Show what would be deleted without actually deleting')
    
    # Watch command
    watch_parser = subparsers.add_parser('watch', help='Watch experiment progress')
    watch_parser.add_argument('experiment_id', help='Experiment ID to watch')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ExperimentManager()
    
    if args.command == 'list':
        manager.list_experiments()
    elif args.command == 'status':
        manager.show_experiment_status(args.experiment_id)
    elif args.command == 'cancel':
        manager.cancel_experiment(args.experiment_id)
    elif args.command == 'cleanup':
        manager.cleanup_old_experiments(args.older_than, args.dry_run)
    elif args.command == 'watch':
        manager.watch_experiment(args.experiment_id)

if __name__ == "__main__":
    main()
