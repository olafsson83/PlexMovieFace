"""Main entry point. Run with no arguments for an interactive menu, or pass
flags for non-interactive/scheduled use.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"


def run_stage(script):
    print(f"\n=== {script} ===")
    result = subprocess.run([sys.executable, str(SRC / script)])
    if result.returncode != 0:
        print(f"\n{script} failed (exit code {result.returncode}).")
        return False
    return True


def full_pipeline():
    if not run_stage("discover_characters.py"):
        return
    print(
        "\nDiscovery complete. Review the crops in characters/, drop a source photo per "
        "character you want swapped into source_faces/, then run the swap stage."
    )


def menu():
    actions = {
        "1": ("Discover characters in the movie", lambda: run_stage("discover_characters.py")),
        "2": ("Swap movie (after you've named some characters)", lambda: run_stage("swap_movie.py")),
        "3": ("Full pipeline (discover, then pause for you to name characters)", full_pipeline),
    }
    while True:
        print("\nPlexMovieFace")
        for key, (label, _) in actions.items():
            print(f"  {key}) {label}")
        print("  0) Exit")

        choice = input("Choose an option: ").strip()
        if choice == "0":
            return
        action = actions.get(choice)
        if not action:
            print("Not a valid option, try again.")
            continue
        action[1]()


def main():
    parser = argparse.ArgumentParser(description="PlexMovieFace pipeline")
    parser.add_argument("--discover-only", action="store_true", help="Run only the character discovery stage")
    parser.add_argument("--swap-only", action="store_true", help="Run only the swap stage")
    args = parser.parse_args()

    if args.discover_only:
        run_stage("discover_characters.py")
    elif args.swap_only:
        run_stage("swap_movie.py")
    else:
        menu()


if __name__ == "__main__":
    main()
