import os

import fire
import socket
import sys

from jobmon.core.exceptions import ConfigError
from jobmon.server.web.api import get_app, JobmonConfig
from jobmon.server.web.models import init_db
from sqlalchemy import create_engine


# Setup local Jobmon web service
class WebServerProcess:
    """Context manager creates the Jobmon web server in a process and tears it down on exit."""

    def __init__(self, filepath: str) -> None:
        """Initializes the web server process.

        Runs on

        Args:
            filepath: path to the SQLlite database file backing up the service.
        """
        if sys.platform == "darwin":
            self.web_host = "127.0.0.1"
        else:
            self.web_host = socket.getfqdn()
        self.web_port = 10_000 + os.getpid() % 30_000
        self.filepath = filepath

    def start_web_service(self):
        """Starts the web service process."""
        database_uri = f"sqlite:///{self.filepath}"
        if not os.path.exists(self.filepath):
            open(self.filepath, 'a').close()  # Make an empty database file
            init_db(create_engine(database_uri))

        config = JobmonConfig(
            dict_config={"db": {"sqlalchemy_database_uri": database_uri}}
        )
        app = get_app(config)
        config.set(
            "http",
            "service_url",
            f"http://{self.web_host}:{self.web_port}",
        )
        config.write()

        # Run the app
        with app.app_context():
            app.run(host="0.0.0.0", port=self.web_port)


def start_web_service(filepath):

    config = JobmonConfig()
    try:
        original_url = config.get("http", "service_url")  # Save the config url to reset on exit
    except ConfigError:
        # OK if service URL not set, since it'll be set to an empty string later
        original_url = ""

    try:
        server = WebServerProcess(filepath=filepath)
        server.start_web_service()
    finally:
        # Allow the program to exit gracefully, and reset the config to its original value
        # Reset the configuration to the original value
        if original_url:
            config.set(
                "http",
                "service_url",
                original_url,
            )
            config.write()


def main():
    fire.Fire(start_web_service)
