import sqlite3
import json
import secrets
import string
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import threading
from contextlib import contextmanager


class Database:
    def __init__(self, db_path: str = None):
        # Render 环境下使用持久化路径
        if db_path is None:
            if os.getenv('RENDER_EXTERNAL_URL'):
                # Render 环境
                db_path = "/opt/render/project/src/gemini_proxy.db"
                # 确保目录存在
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
            else:
                # 本地环境
                db_path = "gemini_proxy.db"

        self.db_path = db_path
        self.local = threading.local()

        # 初始化数据库
        self.init_db()

    @contextmanager
    def get_connection(self):
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(self.db_path)
            self.local.conn.row_factory = sqlite3.Row
            # 设置WAL模式以提高并发性能
            self.local.conn.execute("PRAGMA journal_mode=WAL")
            self.local.conn.execute("PRAGMA synchronous=NORMAL")
            self.local.conn.execute("PRAGMA cache_size=1000")
        yield self.local.conn

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 系统配置表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Gemini Keys表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gemini_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    status INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 模型配置表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT UNIQUE NOT NULL,
                    single_api_rpm_limit INTEGER NOT NULL,
                    single_api_tpm_limit INTEGER NOT NULL,
                    single_api_rpd_limit INTEGER NOT NULL,
                    status INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 用户API Keys表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    name TEXT,
                    status INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP
                )
            ''')

            # 使用记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS usage_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gemini_key_id INTEGER,
                    user_key_id INTEGER,
                    model_name TEXT,
                    requests INTEGER DEFAULT 0,
                    tokens INTEGER DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (gemini_key_id) REFERENCES gemini_keys (id),
                    FOREIGN KEY (user_key_id) REFERENCES user_keys (id)
                )
            ''')

            # 检查并迁移旧表结构
            self._migrate_database(cursor)

            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_usage_model ON usage_logs(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_gemini_key_status ON gemini_keys(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_config_key ON system_config(key)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_configs_name ON model_configs(model_name)')

            # 初始化系统配置
            self._init_system_config(cursor)

            # 初始化模型配置
            self._init_model_configs(cursor)

            conn.commit()

    def _migrate_database(self, cursor):
        """迁移旧的数据库结构"""
        # 检查model_configs表结构
        cursor.execute("PRAGMA table_info(model_configs)")
        columns = [column[1] for column in cursor.fetchall()]

        if 'rpm_limit' in columns:
            # 需要迁移到新的单API限制结构
            print("正在迁移模型配置数据库结构...")

            # 获取旧数据
            cursor.execute("SELECT * FROM model_configs")
            old_configs = cursor.fetchall()

            # 备份旧表
            cursor.execute("ALTER TABLE model_configs RENAME TO model_configs_old")

            # 创建新表
            cursor.execute('''
                CREATE TABLE model_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT UNIQUE NOT NULL,
                    single_api_rpm_limit INTEGER NOT NULL,
                    single_api_tpm_limit INTEGER NOT NULL,
                    single_api_rpd_limit INTEGER NOT NULL,
                    status INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 迁移数据
            for old_config in old_configs:
                cursor.execute('''
                    INSERT INTO model_configs (id, model_name, single_api_rpm_limit, single_api_tpm_limit, single_api_rpd_limit, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (old_config['id'], old_config['model_name'],
                      old_config.get('rpm_limit', 1000), old_config.get('tpm_limit', 2000000),
                      old_config.get('rpd_limit', 50000),
                      old_config['status'], old_config['created_at'], old_config['updated_at']))

            print("模型配置数据库迁移完成")

        # 移除旧的conversations表（如果存在）
        cursor.execute("DROP TABLE IF EXISTS conversations")

    def _init_system_config(self, cursor):
        """初始化系统配置"""
        default_configs = [
            ('default_model_name', 'gemini-2.5-flash', '默认模型名称'),
            ('max_retries', '3', 'API请求最大重试次数'),
            ('request_timeout', '60', 'API请求超时时间（秒）'),
            ('load_balance_strategy', 'least_used', '负载均衡策略: least_used, round_robin'),
            # 思考功能配置
            ('thinking_enabled', 'true', '是否启用思考功能'),
            ('thinking_budget', '-1', '思考预算（token数）：-1=自动，0=禁用，1-32768=固定预算'),
            ('include_thoughts', 'false', '是否在响应中包含思考过程'),
            # 注入prompt配置
            ('inject_prompt_enabled', 'false', '是否启用注入prompt'),
            ('inject_prompt_content', '', '注入的prompt内容'),
            ('inject_prompt_position', 'system', '注入位置: system, user_prefix, user_suffix'),
        ]

        for key, value, description in default_configs:
            cursor.execute('''
                INSERT OR IGNORE INTO system_config (key, value, description)
                VALUES (?, ?, ?)
            ''', (key, value, description))

    def _init_model_configs(self, cursor):
        """初始化模型配置（单个API限制）"""
        default_models = [
            ('gemini-2.5-flash', 1000, 2000000, 50000),  # 单API: RPM, TPM, RPD
            ('gemini-2.5-pro', 100, 1000000, 10000),  # 单API: RPM, TPM, RPD
        ]

        for model_name, rpm, tpm, rpd in default_models:
            cursor.execute('''
                INSERT OR IGNORE INTO model_configs (model_name, single_api_rpm_limit, single_api_tpm_limit, single_api_rpd_limit)
                VALUES (?, ?, ?, ?)
            ''', (model_name, rpm, tpm, rpd))

    # 系统配置管理
    def get_config(self, key: str, default: str = None) -> str:
        """获取系统配置"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM system_config WHERE key = ?', (key,))
            row = cursor.fetchone()
            return row['value'] if row else default

    def set_config(self, key: str, value: str) -> bool:
        """设置系统配置"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO system_config (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, value))
            conn.commit()
            return True

    def get_all_configs(self) -> List[Dict]:
        """获取所有系统配置"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM system_config ORDER BY key')
            return [dict(row) for row in cursor.fetchall()]

    def get_thinking_config(self) -> Dict[str, any]:
        """获取思考配置"""
        return {
            'enabled': self.get_config('thinking_enabled', 'true').lower() == 'true',
            'budget': int(self.get_config('thinking_budget', '-1')),
            'include_thoughts': self.get_config('include_thoughts', 'false').lower() == 'true'
        }

    def set_thinking_config(self, enabled: bool = None, budget: int = None, include_thoughts: bool = None) -> bool:
        """设置思考配置"""
        if enabled is not None:
            self.set_config('thinking_enabled', 'true' if enabled else 'false')

        if budget is not None:
            if not (-1 <= budget <= 32768):
                raise ValueError("thinking_budget must be between -1 and 32768")
            self.set_config('thinking_budget', str(budget))

        if include_thoughts is not None:
            self.set_config('include_thoughts', 'true' if include_thoughts else 'false')

        return True

    def get_inject_prompt_config(self) -> Dict[str, any]:
        """获取注入prompt配置"""
        return {
            'enabled': self.get_config('inject_prompt_enabled', 'false').lower() == 'true',
            'content': self.get_config('inject_prompt_content', ''),
            'position': self.get_config('inject_prompt_position', 'system')
        }

    def set_inject_prompt_config(self, enabled: bool = None, content: str = None, position: str = None) -> bool:
        """设置注入prompt配置"""
        if enabled is not None:
            self.set_config('inject_prompt_enabled', 'true' if enabled else 'false')

        if content is not None:
            self.set_config('inject_prompt_content', content)

        if position is not None:
            if position not in ['system', 'user_prefix', 'user_suffix']:
                raise ValueError("position must be one of: system, user_prefix, user_suffix")
            self.set_config('inject_prompt_position', position)

        return True

    # 模型配置管理（更新为单API限制）
    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表"""
        return ['gemini-2.5-flash', 'gemini-2.5-pro']

    def get_model_config(self, model_name: str) -> Optional[Dict]:
        """获取模型配置（包含计算的总限制）"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM model_configs 
                WHERE model_name = ? AND status = 1
            ''', (model_name,))
            row = cursor.fetchone()

            if not row:
                return None

            config = dict(row)

            # 获取可用的API key数量
            available_keys_count = len(self.get_available_gemini_keys())

            # 计算总限制
            config['total_rpm_limit'] = config['single_api_rpm_limit'] * available_keys_count
            config['total_tpm_limit'] = config['single_api_tpm_limit'] * available_keys_count
            config['total_rpd_limit'] = config['single_api_rpd_limit'] * available_keys_count

            # 为了兼容原有代码，保留旧字段名
            config['rpm_limit'] = config['total_rpm_limit']
            config['tpm_limit'] = config['total_tpm_limit']
            config['rpd_limit'] = config['total_rpd_limit']

            return config

    def get_all_model_configs(self) -> List[Dict]:
        """获取所有模型配置（包含计算的总限制）"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM model_configs ORDER BY model_name')
            configs = [dict(row) for row in cursor.fetchall()]

            # 获取可用的API key数量
            available_keys_count = len(self.get_available_gemini_keys())

            # 为每个配置添加总限制
            for config in configs:
                config['total_rpm_limit'] = config['single_api_rpm_limit'] * available_keys_count
                config['total_tpm_limit'] = config['single_api_tpm_limit'] * available_keys_count
                config['total_rpd_limit'] = config['single_api_rpd_limit'] * available_keys_count

                # 为了兼容原有代码，保留旧字段名
                config['rpm_limit'] = config['total_rpm_limit']
                config['tpm_limit'] = config['total_tpm_limit']
                config['rpd_limit'] = config['total_rpd_limit']

            return configs

    def update_model_config(self, model_name: str, **kwargs) -> bool:
        """更新模型配置"""
        allowed_fields = ['single_api_rpm_limit', 'single_api_tpm_limit', 'single_api_rpd_limit', 'status']
        fields = []
        values = []

        for field, value in kwargs.items():
            if field in allowed_fields:
                fields.append(f"{field} = ?")
                values.append(value)

        if not fields:
            return False

        values.append(model_name)
        query = f"UPDATE model_configs SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP WHERE model_name = ?"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0

    def is_thinking_model(self, model_name: str) -> bool:
        """检查模型是否支持思考功能"""
        return '2.5' in model_name

    # Gemini Key管理
    def add_gemini_key(self, key: str) -> bool:
        """添加Gemini Key"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO gemini_keys (key)
                    VALUES (?)
                ''', (key,))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False

    def update_gemini_key(self, key_id: int, **kwargs):
        """更新Gemini Key"""
        allowed_fields = ['status']
        fields = []
        values = []

        for field, value in kwargs.items():
            if field in allowed_fields:
                fields.append(f"{field} = ?")
                values.append(value)

        if not fields:
            return False

        values.append(key_id)
        query = f"UPDATE gemini_keys SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0

    def delete_gemini_key(self, key_id: int) -> bool:
        """删除Gemini Key"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM gemini_keys WHERE id = ?", (key_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_all_gemini_keys(self) -> List[Dict]:
        """获取所有Gemini Keys"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM gemini_keys ORDER BY id ASC")
            return [dict(row) for row in cursor.fetchall()]

    def get_available_gemini_keys(self) -> List[Dict]:
        """获取可用的Gemini Keys"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM gemini_keys 
                WHERE status = 1 
                ORDER BY id ASC
            ''')
            return [dict(row) for row in cursor.fetchall()]

    def toggle_gemini_key_status(self, key_id: int) -> bool:
        """切换Gemini Key状态"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE gemini_keys 
                SET status = CASE WHEN status = 1 THEN 0 ELSE 1 END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (key_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_gemini_key_by_id(self, key_id: int) -> Optional[Dict]:
        """根据ID获取Gemini Key"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM gemini_keys WHERE id = ?', (key_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_thinking_models(self) -> List[str]:
        """获取支持思考功能的模型列表"""
        return [model for model in self.get_supported_models() if self.is_thinking_model(model)]

    # 用户Key管理
    def generate_user_key(self, name: str = None) -> str:
        """生成用户Key，自动填补删除的ID"""
        prefix = "sk-"
        length = 48
        characters = string.ascii_letters + string.digits
        random_part = ''.join(secrets.choice(characters) for _ in range(length))
        key = prefix + random_part

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 查找最小的可用ID（填补空缺）
            cursor.execute('''
                WITH RECURSIVE seq(x) AS (
                    SELECT 1
                    UNION ALL
                    SELECT x + 1 FROM seq
                    WHERE x < (SELECT COALESCE(MAX(id), 0) + 1 FROM user_keys)
                )
                SELECT MIN(x) as next_id
                FROM seq
                WHERE x NOT IN (SELECT id FROM user_keys)
            ''')

            result = cursor.fetchone()
            if result and result['next_id']:
                next_id = result['next_id']
            else:
                # 如果没有空缺，则使用下一个最大ID
                cursor.execute('SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM user_keys')
                next_id = cursor.fetchone()['next_id']

            try:
                # 使用指定的ID插入
                cursor.execute('''
                    INSERT INTO user_keys (id, key, name) VALUES (?, ?, ?)
                ''', (next_id, key, name))
                conn.commit()
            except sqlite3.IntegrityError:
                # 如果ID冲突，则使用自动递增
                cursor.execute('''
                    INSERT INTO user_keys (key, name) VALUES (?, ?)
                ''', (key, name))
                conn.commit()

        return key

    def validate_user_key(self, key: str) -> Optional[Dict]:
        """验证用户Key"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM user_keys 
                WHERE key = ? AND status = 1
            ''', (key,))
            row = cursor.fetchone()

            if row:
                # 更新最后使用时间
                cursor.execute('''
                    UPDATE user_keys SET last_used = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (row['id'],))
                conn.commit()
                return dict(row)
            return None

    def get_all_user_keys(self) -> List[Dict]:
        """获取所有用户Keys"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_keys ORDER BY id ASC")
            return [dict(row) for row in cursor.fetchall()]

    def toggle_user_key_status(self, key_id: int) -> bool:
        """切换用户Key状态"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE user_keys 
                SET status = CASE WHEN status = 1 THEN 0 ELSE 1 END 
                WHERE id = ?
            ''', (key_id,))
            conn.commit()
            return cursor.rowcount > 0

    def delete_user_key(self, key_id: int) -> bool:
        """删除用户Key"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_keys WHERE id = ?", (key_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_user_key_by_id(self, key_id: int) -> Optional[Dict]:
        """根据ID获取用户Key"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM user_keys WHERE id = ?', (key_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_user_key(self, key_id: int, **kwargs) -> bool:
        """更新用户Key信息"""
        allowed_fields = ['name', 'status']
        fields = []
        values = []

        for field, value in kwargs.items():
            if field in allowed_fields:
                fields.append(f"{field} = ?")
                values.append(value)

        if not fields:
            return False

        values.append(key_id)
        query = f"UPDATE user_keys SET {', '.join(fields)} WHERE id = ?"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0

    def get_key_usage_stats(self, key_id: int, key_type: str = 'gemini', days: int = 7) -> Dict:
        """获取密钥的使用统计"""
        column = 'gemini_key_id' if key_type == 'gemini' else 'user_key_id'

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(tokens) as total_tokens,
                    DATE(timestamp) as date
                FROM usage_logs 
                WHERE {column} = ? 
                AND timestamp > datetime('now', '-{days} days')
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            ''', (key_id,))

            daily_stats = [dict(row) for row in cursor.fetchall()]

            # 总计
            cursor.execute(f'''
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(tokens) as total_tokens
                FROM usage_logs 
                WHERE {column} = ? 
                AND timestamp > datetime('now', '-{days} days')
            ''', (key_id,))

            total_stats = dict(cursor.fetchone())

            return {
                'daily_stats': daily_stats,
                'total_stats': total_stats
            }

    # 使用记录管理
    def log_usage(self, gemini_key_id: int, user_key_id: int, model_name: str, requests: int = 1, tokens: int = 0):
        """记录使用量（按模型）"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO usage_logs (gemini_key_id, user_key_id, model_name, requests, tokens)
                VALUES (?, ?, ?, ?, ?)
            ''', (gemini_key_id, user_key_id, model_name, requests, tokens))
            conn.commit()

    def get_usage_stats(self, model_name: str, time_window: str = 'minute') -> Dict[str, int]:
        """获取指定模型在指定时间窗口内的使用统计"""
        time_deltas = {
            'minute': timedelta(minutes=1),
            'day': timedelta(days=1)
        }

        if time_window not in time_deltas:
            raise ValueError(f"Invalid time window: {time_window}")

        cutoff_time = datetime.now() - time_deltas[time_window]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    COALESCE(SUM(requests), 0) as total_requests,
                    COALESCE(SUM(tokens), 0) as total_tokens
                FROM usage_logs
                WHERE model_name = ? AND timestamp > ?
            ''', (model_name, cutoff_time))

            row = cursor.fetchone()
            return {
                'requests': row['total_requests'],
                'tokens': row['total_tokens']
            }

    def get_all_usage_stats(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """获取所有模型的使用统计"""
        stats = {}
        models = self.get_supported_models()

        for model in models:
            stats[model] = {
                'minute': self.get_usage_stats(model, 'minute'),
                'day': self.get_usage_stats(model, 'day')
            }

        return stats

    def get_model_usage_rate(self, model_name: str) -> float:
        """获取模型使用率（基于RPM）"""
        stats = self.get_usage_stats(model_name, 'minute')
        model_config = self.get_model_config(model_name)

        if not model_config or model_config['rpm_limit'] == 0:
            return 0.0

        return stats['requests'] / model_config['rpm_limit']

    def get_database_stats(self) -> Dict:
        """获取数据库统计信息"""
        stats = {}

        try:
            # 获取数据库文件大小
            if os.path.exists(self.db_path):
                stats['database_size_mb'] = os.path.getsize(self.db_path) / 1024 / 1024
            else:
                stats['database_size_mb'] = 0

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # 获取各表的记录数
                tables = ['system_config', 'gemini_keys', 'model_configs', 'user_keys', 'usage_logs']
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    stats[f'{table}_count'] = cursor.fetchone()['count']

                # 获取最近的使用记录
                cursor.execute('''
                    SELECT COUNT(*) as recent_usage 
                    FROM usage_logs 
                    WHERE timestamp > datetime('now', '-1 hour')
                ''')
                stats['recent_usage_count'] = cursor.fetchone()['recent_usage']

        except Exception as e:
            stats['error'] = str(e)

        return stats

    def cleanup_old_logs(self, days: int = 30) -> int:
        """清理旧的使用日志"""
        cutoff_date = datetime.now() - timedelta(days=days)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM usage_logs 
                WHERE timestamp < ?
            ''', (cutoff_date,))
            deleted_count = cursor.rowcount
            conn.commit()

        return deleted_count

    def backup_database(self, backup_path: str = None) -> bool:
        """备份数据库"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"gemini_proxy_backup_{timestamp}.db"

        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            return True
        except Exception as e:
            print(f"Backup failed: {e}")
            return False

    def get_system_info(self) -> Dict:
        """获取系统信息"""
        return {
            'database_path': self.db_path,
            'database_exists': os.path.exists(self.db_path),
            'environment': 'render' if os.getenv('RENDER_EXTERNAL_URL') else 'local',
            'stats': self.get_database_stats()
        }