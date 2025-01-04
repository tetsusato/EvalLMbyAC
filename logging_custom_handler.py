import logging
import os
from datetime import datetime, timedelta
import re

class TimeBasedFileHandler(logging.Handler):
    def __init__(self, log_dir, minutes=10, *args, **kwargs):
        """
        ログファイルの切り替え時間間隔を設定
        :param log_dir: ログファイルを保存するディレクトリ
        :param minutes: ファイル切り替えの時間間隔（分）
        """
        super().__init__(*args, **kwargs)
        self.log_dir = log_dir
        self.minutes = minutes
        self.last_log_file = {}
        self.streams = {}

        # 初期化時に既存の最新ファイルを確認
        self._initialize_last_log_files()

    def _initialize_last_log_files(self):
        """既存ログファイルを確認して最新のファイルを設定"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            return

        log_files = [f for f in os.listdir(self.log_dir) if os.path.isfile(os.path.join(self.log_dir, f))]
        pattern = re.compile(r'^(?P<level>\w+)_(?P<timestamp>\d{8}-\d{4})\.log$')

        for log_file in log_files:
            match = pattern.match(log_file)
            if match:
                level_name = match.group('level')
                timestamp = match.group('timestamp')
                log_time = datetime.strptime(timestamp, '%Y%m%d-%H%M')
                # 最新のログファイルを設定（10分以内のもののみ）
                if level_name not in self.last_log_file or self.last_log_file[level_name]['time'] < log_time:
                    if datetime.now() - log_time < timedelta(minutes=self.minutes):
                        self.last_log_file[level_name] = {
                            'file': os.path.join(self.log_dir, log_file),
                            'time': log_time,
                        }

    def _get_filename(self, level):
        """ログレベルに基づいてファイル名を生成"""
        now = datetime.now()
        level_name = logging.getLevelName(level).lower()
        return os.path.join(self.log_dir, f"{level_name}_{now.strftime('%Y%m%d-%H%M')}.log")

    def _get_stream(self, level):
        """指定されたレベルのストリームを取得"""
        level_name = logging.getLevelName(level).lower()

        # 初期化時に設定されたファイルを利用
        if level_name in self.last_log_file:
            if level_name not in self.streams:
                self.streams[level_name] = open(self.last_log_file[level_name]['file'], 'a')
            return self.streams[level_name]

        # 初期化で該当ファイルがない場合、新しいファイルを作成
        if level_name not in self.streams:
            filename = self._get_filename(level)
            stream = open(filename, 'a')
            self.streams[level_name] = stream
            self.last_log_file[level_name] = {'file': filename, 'time': datetime.now()}
            return stream

        return self.streams[level_name]

    def emit(self, record):
        """ログの出力"""
        stream = self._get_stream(record.levelno)
        msg = self.format(record)
        stream.write(f"{msg}\n")
        stream.flush()

    def close(self):
        """全ストリームを閉じる"""
        for stream in self.streams.values():
            stream.close()
        super().close()
