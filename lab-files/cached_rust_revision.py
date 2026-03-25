import subprocess
from pathlib import Path

from lab import tools
from lab.cached_revision import CachedRevision


class CachedRustPlannerRevision(CachedRevision):
    """
    Cached revision for a Rust planner.

    Builds the planner using cargo and keeps only the compiled binary.
    """

    def __init__(self, revision_cache, repo, rev, build_options=None):
        build_cmd = ["cargo", "build", "--release"]
        if build_options:
            build_cmd += build_options

        super().__init__(
            revision_cache,
            repo,
            rev,
            build_cmd=build_cmd,
            exclude=[".git", "target/debug"],
        )

        self.build_options = build_options or []

    def _cleanup(self):
        """
        Keep only the release binary and remove unnecessary files.
        """
        target_dir = self.path / "target" / "release"

        # Remove everything except release binaries
        for path in self.path.glob("target/*"):
            if path != target_dir:
                tools.remove_path(path)

        # Optionally strip binaries (Linux/macOS)
        binaries = list(target_dir.glob("*"))
        binaries = [b for b in binaries if b.is_file() and b.stat().st_mode & 0o111]

        if binaries:
            try:
                subprocess.check_call(["strip"] + [str(b) for b in binaries])
            except Exception:
                pass  # stripping is optional

    def get_binary_path(self, binary_name="planners"):
        """
        Return path to compiled binary inside experiment.
        """
        return self.get_relative_exp_path(f"target/release/{binary_name}")