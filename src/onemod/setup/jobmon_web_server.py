from time import sleep
from types import TracebackType
from typing import Any, Optional
import multiprocessing as mp
import os
import requests
import signal
import socket
import sys

from jobmon.core.configuration import JobmonConfig
from jobmon.server.web.api import get_app, JobmonConfig
from jobmon.server.web.models import init_db
from sqlalchemy import create_engine


# Setup local Jobmon web service
class WebServerProcess:
    """Context manager creates the Jobmon web server in a process and tears it down on exit."""

    def __init__(self, filepath: str) -> None:
        """Initializes the web server process.

        Args:
            ephemera: a dictionary containing the connection information for the database,
            specifically the database host, port, service account user, service account
            password, and database name
        """
        if sys.platform == "darwin":
            self.web_host = "127.0.0.1"
        else:
            self.web_host = socket.getfqdn()
        self.web_port = str(10_000 + os.getpid() % 30_000)
        self.filepath = filepath

        self.original_url = ""

    def __enter__(self):
        """Starts the web service process."""
        # jobmon_cli string
        database_uri = f"sqlite:///{self.filepath}"

        init_db(create_engine(database_uri))

        config = JobmonConfig(
            dict_config={"db": {"sqlalchemy_database_uri": database_uri}}
        )
        app = get_app(config)

        # Configure the client config temporarily
        # Save the original config
        self.original_url = config.get("http", "service_url")
        config.set(
            "http",
            "service_url",
            f"http://{self.web_host}:{self.web_port}",
        )
        config.write()

        # Bind the flask app and return self
        self.app = app
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        # Reset the configuration to the original value
        config = JobmonConfig()
        config.set(
            "http",
            "service_url",
            self.original_url,
        )


def start_web_service(filepath):

    with WebServerProcess(filepath) as server:
        app = server.app
        with app.app_context():
            app.run(host="0.0.0.0", port=server.web_port)



