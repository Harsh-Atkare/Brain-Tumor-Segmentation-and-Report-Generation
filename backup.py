import shutil
import os
from datetime import datetime
from pathlib import Path
import zipfile

class BackupManager:
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        
    def backup_data(self):
        """Create backup of data directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}.zip"
        backup_path = self.backup_dir / backup_name
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.base_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(self.base_dir.parent)
                    zipf.write(file_path, arcname)
                    
        print(f"Backup created: {backup_path}")
        return backup_path
        
    def cleanup_old_backups(self, keep_days=7):
        """Remove backups older than specified days."""
        cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
        
        for backup_file in self.backup_dir.glob("backup_*.zip"):
            if backup_file.stat().st_mtime < cutoff_time:
                backup_file.unlink()
                print(f"Removed old backup: {backup_file}")

if __name__ == "__main__":
    backup_manager = BackupManager()
    backup_manager.backup_data()
    backup_manager.cleanup_old_backups()