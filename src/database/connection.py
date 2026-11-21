"""Shared TiDB settings for tools and apps. The legacy schema is opt-in only."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
import re

import certifi
from dotenv import dotenv_values
from sqlalchemy import URL, create_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DATABASE = "fraud_detection_demo"
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class TiDBSettings:
    host: str
    port: int
    user: str
    password: str = field(repr=False)
    database: str
    tls: str = "verify"
    ca: str | None = None

    @classmethod
    def from_mapping(cls, values: Mapping, database=None):
        host = (values.get("DB_HOST") or "").strip()
        user = (values.get("DB_USER") or "").strip()
        password = values.get("DB_PASS") or ""
        name = database or values.get("DB_NAME") or ""
        tls = (values.get("DB_TLS") or "verify").strip().lower()
        if not host or not user:
            raise ValueError("Set DB_HOST and DB_USER")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name):
            raise ValueError("DB_NAME must be a valid project database name")
        if tls not in {"verify", "disable"}:
            raise ValueError("DB_TLS must be verify or disable")
        if tls == "disable" and host.lower() not in LOOPBACK_HOSTS:
            raise ValueError("DB_TLS=disable is only permitted for a loopback development server")
        if not password and not (tls == "disable" and host.lower() in LOOPBACK_HOSTS):
            raise ValueError("Set DB_PASS")
        try:
            port = int(values.get("DB_PORT") or 4000)
        except ValueError:
            raise ValueError("DB_PORT must be an integer") from None
        if not 1 <= port <= 65535:
            raise ValueError("DB_PORT must be between 1 and 65535")
        return cls(host, port, user, password, name, tls, values.get("SSL_CA") or None)

def load_demo_settings(*, env_file=None):
    file = PROJECT_ROOT / ".env" if env_file is None else Path(env_file)
    if env_file is not None and not file.is_file():
        raise ValueError("The selected --env-file does not exist")
    values = {**dotenv_values(file, interpolate=False), **os.environ}
    database = values.get("TIDB_DEMO_DATABASE") or DEMO_DATABASE
    if not database.endswith("_demo"):
        raise ValueError("TIDB_DEMO_DATABASE must end in _demo; the legacy schema is not a demo target")
    return TiDBSettings.from_mapping(values, database)


def open_engine(settings, *, without_database=False):
    url = URL.create(
        "mysql+pymysql",
        username=settings.user,
        password=settings.password,
        host=settings.host,
        port=settings.port,
        database=None if without_database else settings.database,
        query={"charset": "utf8mb4"},
    )
    args = {
        "connect_timeout": 8, "read_timeout": 30, "write_timeout": 30,
        "init_command": "SET time_zone = '+00:00'",
    }
    if settings.tls == "verify":
        ca = Path(settings.ca).expanduser() if settings.ca else Path(certifi.where())
        if not ca.is_file():
            raise ValueError("SSL_CA must point to a readable CA certificate file")
        args.update(ssl_ca=str(ca), ssl_verify_cert=True, ssl_verify_identity=True)
    else:
        args["ssl_disabled"] = True
    return create_engine(url, connect_args=args, pool_pre_ping=True, pool_recycle=900, hide_parameters=True)


def db_error_code(exc):
    args = getattr(getattr(exc, "orig", exc), "args", ())
    return args[0] if args and isinstance(args[0], int) else "unknown"
