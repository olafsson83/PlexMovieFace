"""Main entry point. Run with no arguments for an interactive menu, or pass
flags for non-interactive/scheduled use.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"


def run_stage(script, extra_args=None):
    print(f"\n=== {script} ===")
    result = subprocess.run([sys.executable, str(SRC / script), *(extra_args or [])])
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
        "4": ("Calibrate -- test timing on a short sample before a full run",
              lambda: run_stage("swap_movie.py", ["--calibrate"])),
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
    parser.add_argument("--calibrate", nargs="?", const="30", metavar="SECONDS",
                         help="Test timing on the first SECONDS (default 30) of the movie instead of a full run")
    parser.add_argument("--no-tracking", action="store_true",
                         help="Force full per-frame detection instead of optical-flow tracking (slower, for a correctness check)")
    args = parser.parse_args()

    swap_extra = []
    if args.calibrate is not None:
        swap_extra += ["--calibrate", args.calibrate]
    if args.no_tracking:
        swap_extra += ["--no-tracking"]

    if args.discover_only:
        run_stage("discover_characters.py")
    elif args.swap_only or swap_extra:
        run_stage("swap_movie.py", swap_extra)
    else:
        menu()


if __name__ == "__main__":
    main()
