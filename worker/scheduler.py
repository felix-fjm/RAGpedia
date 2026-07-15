"""
Weekly update scheduler — intended production entry point.

NOTE: This scheduler is NOT deployed, scheduled, or running anywhere.
      It exists as a portfolio artifact demonstrating the intended Monday
      03:00 UTC weekly cadence.  The development server (Hetzner) was
      decommissioned after the build was complete; no production cron
      infrastructure was ever set up.

      To trigger a one-off manual update (the correct approach for testing):
          python update.py [--dump-path /data/wiki_dump.json.gz]
"""

import logging
import time

import schedule

from update import run_update

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scheduler")


def main() -> None:
    logger.info(
        "Registering weekly update job: every Monday at 03:00 UTC.  "
        "(NOT deployed — portfolio repo only.  "
        "Use `python update.py` to run a manual update.)"
    )
    schedule.every().monday.at("03:00").do(run_update)

    while True:
        schedule.run_pending()
        time.sleep(60)  # check schedule once per minute


if __name__ == "__main__":
    main()
