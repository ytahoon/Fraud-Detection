"""Initialize the credential-free, loopback-only Docker demonstration."""

import time

from sqlalchemy.exc import OperationalError

from . import ieee_release, synthetic_release
from .database.connection import LOOPBACK_HOSTS, load_demo_settings, open_engine
from .database.tidb_admin import check, create_demo_database, migrate, seed_demo
from .database.tidb_scoring import score_demo


def wait_for_tidb(settings):
    server = open_engine(settings, without_database=True)
    try:
        deadline = time.monotonic() + 90
        while True:
            try:
                with server.connect() as conn:
                    version = conn.exec_driver_sql("SELECT VERSION()").scalar_one()
                    if "TiDB" not in version:
                        raise ValueError("The local demo expects TiDB")
                    return
            except OperationalError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
    finally:
        server.dispose()


def main():
    settings = load_demo_settings()
    if settings.host not in LOOPBACK_HOSTS or settings.database != "fraud_detection_local_demo" or settings.tls != "disable":
        raise ValueError("This command only initializes the isolated local Docker demo")
    releases = (synthetic_release.load_release(), ieee_release.load_release(ieee_release.FIXTURE_RELEASE))
    wait_for_tidb(settings)
    create_demo_database(settings)
    engine = open_engine(settings)
    try:
        migrate(engine)
        seed_demo(engine)
        for release in releases:
            result = score_demo(engine, release)
            if result.quarantined:
                raise ValueError(f"Local demo contains quarantined rows for {release.version}")
            print(f"{release.version}: {result.scored} new scores")
        check(engine)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
