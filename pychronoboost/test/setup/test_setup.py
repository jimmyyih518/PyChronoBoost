import subprocess
import venv
import os
import tempfile
import pytest


def test_setup_py():
    """
    Test that the package can be installed using setup.py in a virtual environment.
    """
    # Create a temporary directory for the virtual environment
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a virtual environment
        venv.create(temp_dir, with_pip=True)

        # Paths for the Python executable and pip within the virtual environment
        python_executable = os.path.join(temp_dir, "bin", "python")
        pip_executable = os.path.join(temp_dir, "bin", "pip")

        # Upgrade pip in the virtual environment
        subprocess.check_call(
            [python_executable, "-m", "pip", "install", "--upgrade", "pip"]
        )

        # Install the package in the virtual environment
        subprocess.check_call([pip_executable, "install", "--no-deps", "."])

        # Check if the package can be imported
        try:
            subprocess.check_call([python_executable, "-c", "import pychronoboost"])
        except subprocess.CalledProcessError:
            pytest.fail("Failed to import 'pychronoboost' after installation")
