# logs_analyzer.py - Log analysis utility
import re
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter

class LogAnalyzer:
    def __init__(self, log_file="app.log"):
        self.log_file = Path(log_file)
        
    def parse_log_line(self, line):
        """Parse a single log line."""

        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - ([^-]+) - ([^-]+) - (.+)'
        match = re.match(pattern, line.strip())
        
        if match:
            return {
                'timestamp': datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S'),
                'logger': match.group(2).strip(),
                'level': match.group(3).strip(),
                'message': match.group(4).strip()
            }
        return None
        
    def analyze_logs(self, hours_back=24):
        """Analyze logs from the last N hours."""
        if not self.log_file.exists():
            print("Log file not found")
            return
            
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        log_entries = []
        error_count = 0
        warning_count = 0
        info_count = 0
        
        with open(self.log_file, 'r') as f:
            for line in f:
                entry = self.parse_log_line(line)
                if entry and entry['timestamp'] >= cutoff_time:
                    log_entries.append(entry)
                    
                    if entry['level'] == 'ERROR':
                        error_count += 1
                    elif entry['level'] == 'WARNING':
                        warning_count += 1
                    elif entry['level'] == 'INFO':
                        info_count += 1
                        
        # Generate report
        print(f"Log Analysis Report (Last {hours_back} hours)")
        print("="*50)
        print(f"Total entries: {len(log_entries)}")
        print(f"Errors: {error_count}")
        print(f"Warnings: {warning_count}")
        print(f"Info: {info_count}")
        
        if error_count > 0:
            print("\nRecent Errors:")
            for entry in log_entries:
                if entry['level'] == 'ERROR':
                    print(f"  {entry['timestamp']}: {entry['message']}")
                    
        # Most common messages
        messages = [entry['message'] for entry in log_entries]
        common_messages = Counter(messages).most_common(5)
        
        print("\nMost Common Messages:")
        for message, count in common_messages:
            print(f"  {count}x: {message[:80]}...")
            
        return {
            'total_entries': len(log_entries),
            'errors': error_count,
            'warnings': warning_count,
            'info': info_count,
            'common_messages': common_messages
        }

if __name__ == "__main__":
    analyzer = LogAnalyzer()
    analyzer.analyze_logs()