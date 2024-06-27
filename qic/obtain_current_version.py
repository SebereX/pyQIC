#!/usr/bin/env python3
"""
Obtain the information about the current installation of pyQIC.
"""
import subprocess
import os
import logging

logger = logging.getLogger(__name__)

def get_qic_info(verbose = False):
    # Package name
    package_name = 'qicna'

    # Get the package information using pip show
    result = subprocess.run(
        ['pip', 'show', package_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return

    # Extract the location of the package
    info_lines = result.stdout.splitlines()
    location_line = next((line for line in info_lines if line.startswith('Location:')), None)
    if location_line is None:
        print("Could not find the package location.")
        return

    package_location = location_line.split(' ', 1)[1]
    package_path = os.path.join(package_location)

    # Save the current directory
    original_dir = os.getcwd()

    try:
        # Change to the package directory
        os.chdir(package_path)

        # Get the current git branch
        branch_result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if branch_result.returncode != 0:
            print(f"Error getting git branch: {branch_result.stderr}")
            return
        branch = branch_result.stdout.strip()

        # Get the latest commit hash
        commit_result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if commit_result.returncode != 0:
            print(f"Error getting git commit: {commit_result.stderr}")
            return
        commit = commit_result.stdout.strip()

        # Get the repository URL
        remote_result = subprocess.run(
            ['git', 'config', '--get', 'remote.origin.url'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if remote_result.returncode != 0:
            print(f"Error getting git remote URL: {remote_result.stderr}")
            return
        remote_url = remote_result.stdout.strip()

        if verbose:
            # Print the git information
            print(f"Package: {package_name}")
            print(f"Location: {package_path}")
            print(f"Branch: {branch}")
            print(f"Commit: {commit}")
            print(f"Repository URL: {remote_url}")

        # Change back to the original directory
        os.chdir(original_dir)

        return remote_url, branch, commit

    except:
        raise ImportError('Unable to determine the current installation')

    finally:
        # Change back to the original directory
        os.chdir(original_dir)

