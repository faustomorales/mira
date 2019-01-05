import shutil
import os

assets_directory = 'tests/assets'


def pytest_sessionstart(session):
    """ before session.main() is called. """
    if not os.path.exists(assets_directory):
        os.makedirs(assets_directory)


def pytest_sessionfinish(session, exitstatus):
    """ whole test run finishes. """
    shutil.rmtree(assets_directory)