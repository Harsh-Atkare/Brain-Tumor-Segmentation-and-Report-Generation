# cleanup.py - Cleanup utility for old files
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

class CleanupManager:
    def __init__(self):
        self.upload_dir = Path("data/uploads")
        self.output_dir = Path("data/outputs")
        
    def cleanup_old_files(self, days_old=7):
        """Clean up files older than specified days."""
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        cleaned_count = 0
        cleaned_size = 0
        
        for directory in [self.upload_dir, self.output_dir]:
            if not directory.exists():
                continue
                
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    # Skip .gitkeep files
                    if file_path.name == '.gitkeep':
                        continue
                        
                    if file_path.stat().st_mtime < cutoff_time:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        cleaned_count += 1
                        cleaned_size += file_size
                        print(f"Removed: {file_path}")
                        
            # Remove empty directories
            for dir_path in directory.rglob("*"):
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    try:
                        dir_path.rmdir()
                        print(f"Removed empty directory: {dir_path}")
                    except OSError:
                        pass  # Directory not empty or other error
                        
        print(f"\nCleanup complete:")
        print(f"Files removed: {cleaned_count}")
        print(f"Space freed: {cleaned_size / (1024*1024):.2f} MB")
        
    def get_disk_usage(self):
        """Get disk usage statistics."""
        stats = {}
        
        for name, directory in [("uploads", self.upload_dir), ("outputs", self.output_dir)]:
            if directory.exists():
                total_size = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
                file_count = sum(1 for f in directory.rglob("*") if f.is_file())
                
                stats[name] = {
                    'total_size_mb': total_size / (1024*1024),
                    'file_count': file_count
                }
                
        return stats

if __name__ == "__main__":
    cleanup_manager = CleanupManager()
    
    print("Current disk usage:")
    usage = cleanup_manager.get_disk_usage()
    for name, stats in usage.items():
        print(f"{name}: {stats['file_count']} files, {stats['total_size_mb']:.2f} MB")
        
    print("\nCleaning up files older than 7 days...")
    cleanup_manager.cleanup_old_files(7)