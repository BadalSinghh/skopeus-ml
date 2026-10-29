"""Render Keep-Alive Ping Bot for Skopeus ML Backend.

Render's free tier automatically suspends web services after 15 minutes of inactivity.
This background service periodically pings the server's own public endpoint (e.g. every 10 minutes),
keeping the instance warm 24/7 and completely eliminating cold-start delays for web clients.
"""

import asyncio
import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger("skopeus.ping_bot")

DEFAULT_INTERVAL_SECONDS = 600  # 10 minutes (Render timeout is 15 minutes)


class RenderPingBot:
    """Async background task that keeps the Render deployment alive."""

    def __init__(
        self,
        target_url: Optional[str] = None,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    ):
        # Auto-detect Render external URL, fallback to environment variable
        self.target_url = (
            target_url
            or os.environ.get("RENDER_EXTERNAL_URL")
            or os.environ.get("PING_URL")
            or os.environ.get("BACKEND_URL")
        )
        self.interval_seconds = int(
            os.environ.get("PING_INTERVAL_SECONDS", str(interval_seconds))
        )
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self.start_time: float = time.time()
        self.ping_count: int = 0
        self.last_ping_time: Optional[float] = None
        self.last_ping_status: Optional[str] = None

    def get_ping_target(self) -> Optional[str]:
        if not self.target_url:
            return None
        url = self.target_url.rstrip("/")
        if not url.endswith("/api/ping"):
            url = f"{url}/api/ping"
        return url

    async def _ping_loop(self) -> None:
        """Periodic keepalive loop."""
        ping_target = self.get_ping_target()
        logger.info(
            "Render Ping Bot started. Target: %s, Interval: %ds",
            ping_target or "[NO EXTERNAL URL SET - LOCAL DEV]",
            self.interval_seconds,
        )

        # Initial delay of 60 seconds after server startup
        await asyncio.sleep(60)

        while self.is_running:
            target = self.get_ping_target()
            if target:
                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        resp = await client.get(target)
                        self.ping_count += 1
                        self.last_ping_time = time.time()
                        self.last_ping_status = f"HTTP {resp.status_code}"
                        logger.info(
                            "Render Keep-Alive ping #%d successful -> %s [%s]",
                            self.ping_count,
                            target,
                            self.last_ping_status,
                        )
                except Exception as e:
                    self.last_ping_status = f"FAILED: {e}"
                    logger.warning("Render Keep-Alive ping failed: %s", e)
            else:
                logger.debug("Render Ping Bot active but RENDER_EXTERNAL_URL is not configured (local development).")

            await asyncio.sleep(self.interval_seconds)

    def start(self) -> None:
        """Start the background ping bot."""
        if self.is_running:
            return
        self.is_running = True
        self._task = asyncio.create_task(self._ping_loop())

    def stop(self) -> None:
        """Stop the background ping bot."""
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("Render Ping Bot stopped.")

    def get_status(self) -> dict:
        """Return diagnostic status of the ping bot."""
        return {
            "enabled": self.is_running,
            "target_url": self.get_ping_target(),
            "interval_seconds": self.interval_seconds,
            "total_pings": self.ping_count,
            "last_ping_time": (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.last_ping_time))
                if self.last_ping_time
                else None
            ),
            "last_ping_status": self.last_ping_status,
            "uptime_seconds": round(time.time() - self.start_time, 1),
        }


# Singleton bot instance
ping_bot = RenderPingBot()
