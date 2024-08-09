import os
import subprocess
import sys
import tempfile
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SandboxRunner:
    def __init__(self):
        self.venv_dir = None
        self.temp_dir = None

    def create_sandbox_env(self):
        logger.info("Creating a sandbox environment...")
        self.temp_dir = tempfile.mkdtemp()
        self.venv_dir = os.path.join(self.temp_dir, "venv")

        try:
            subprocess.check_call([sys.executable, "-m", "venv", self.venv_dir])
            logger.info(f"Virtual environment created at: {self.venv_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            raise

    def install_packages(self, packages):
        pip_executable = os.path.join(self.venv_dir, "bin", "pip")

        for package in packages:
            logger.info(f"Installing package: {package}")
            try:
                subprocess.check_call([pip_executable, "install", package])
                logger.info(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install package {package}: {e}")
                raise

    def execute_file(self, file_path):
        python_executable = os.path.join(self.venv_dir, "bin", "python")

        try:
            logger.info(f"Executing Python file in sandboxed environment: {file_path}")
            result = subprocess.run(
                [python_executable, file_path],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Code executed successfully.")
            logger.info(f"Output:\n{result.stdout}")
            return result.stdout, None
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing code: {e}")
            logger.error(f"Error output:\n{e.stderr}")
            return None, e.stderr

    def cleanup_sandbox(self):
        if self.temp_dir:
            logger.info(f"Cleaning up sandbox environment at {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            logger.info("Sandbox environment cleaned up.")

    def run_sandboxed_file(self, file_path, packages=None, expected_output=None):
        if packages is None:
            packages = []

        try:
            # Step 1: Create the sandbox environment
            self.create_sandbox_env()

            # Step 2: Install the required packages
            self.install_packages(packages)

            # Step 3: Execute the code
            output, error = self.execute_file(file_path)

            # Step 4: Check for expected output
            if expected_output is not None and output is not None:
                if expected_output in output:
                    logger.info("The code produced the expected output.")
                else:
                    logger.warning("The code did not produce the expected output.")

            return output, error
        finally:
            # Step 5: Clean up the environment
            self.cleanup_sandbox()
