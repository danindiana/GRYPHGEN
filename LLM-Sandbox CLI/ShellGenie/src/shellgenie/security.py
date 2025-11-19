"""Security module for command validation and sandboxing."""

import os
import re
import shlex
from pathlib import Path
from typing import List, Optional, Set, Tuple

from loguru import logger

from shellgenie.models import CommandResponse, SecurityLevel


class SecurityValidator:
    """Validates commands for security risks."""

    # Dangerous commands that should be blocked or warned about
    CRITICAL_COMMANDS = {
        "rm -rf /",
        "dd if=/dev/zero",
        "mkfs",
        ":(){ :|:& };:",  # Fork bomb
        "> /dev/sda",
        "mv / /dev/null",
    }

    DANGEROUS_PATTERNS = [
        r"rm\s+-rf\s+/",  # Delete root
        r"rm\s+-rf\s+\*",  # Delete everything
        r"dd\s+if=/dev/(zero|random)\s+of=/dev/sd[a-z]",  # Overwrite disk
        r"mkfs\.",  # Format filesystem
        r"chmod\s+-R\s+777\s+/",  # Dangerous permissions
        r"chown\s+-R\s+.*\s+/",  # Change ownership of root
        r":\(\)\{.*\|\:&\}\;\:",  # Fork bomb
        r"wget.*\|\s*(bash|sh)",  # Download and execute
        r"curl.*\|\s*(bash|sh)",  # Download and execute
        r"sudo\s+su\s+-",  # Privilege escalation
        r"nc\s+-.*-e",  # Netcat reverse shell
        r"/dev/tcp/.*exec",  # Bash reverse shell
    ]

    MODERATE_PATTERNS = [
        r"sudo\s+",  # Sudo usage
        r"rm\s+",  # File deletion
        r"chmod\s+",  # Permission changes
        r"chown\s+",  # Ownership changes
        r"kill\s+-9",  # Force kill
        r"pkill\s+",  # Kill processes
        r"reboot",  # System reboot
        r"shutdown",  # System shutdown
        r"systemctl\s+stop",  # Stop services
        r"service\s+.*\s+stop",  # Stop services
    ]

    NETWORK_PATTERNS = [
        r"curl\s+",
        r"wget\s+",
        r"nc\s+",
        r"netcat\s+",
        r"telnet\s+",
        r"ssh\s+",
        r"scp\s+",
        r"rsync\s+",
    ]

    # Commands that are generally safe
    SAFE_COMMANDS = {
        "ls", "cat", "echo", "pwd", "cd", "grep", "find", "head", "tail",
        "wc", "sort", "uniq", "diff", "less", "more", "tree", "file",
        "stat", "which", "whereis", "man", "info", "help", "history",
        "date", "cal", "uptime", "whoami", "hostname", "uname",
    }

    def __init__(self, security_level: SecurityLevel = SecurityLevel.MODERATE):
        """Initialize the security validator.

        Args:
            security_level: The security level to enforce
        """
        self.security_level = security_level
        self.whitelist: Set[str] = set()
        self.blacklist: Set[str] = set()

    def validate_command(self, command: str) -> CommandResponse:
        """Validate a command for security risks.

        Args:
            command: The command to validate

        Returns:
            CommandResponse with risk assessment and warnings
        """
        command = command.strip()
        warnings: List[str] = []
        alternatives: List[str] = []
        risk_level = "low"

        # Check if disabled
        if self.security_level == SecurityLevel.DISABLED:
            return CommandResponse(
                command=command,
                risk_level="unknown",
                warnings=["Security checks are disabled!"],
            )

        # Check critical commands
        if self._is_critical_command(command):
            risk_level = "critical"
            warnings.append("CRITICAL: This command could destroy your system!")
            if self.security_level == SecurityLevel.STRICT:
                warnings.append("Command blocked by security policy.")
                return CommandResponse(
                    command="",
                    risk_level=risk_level,
                    warnings=warnings,
                )

        # Check dangerous patterns
        dangerous_matches = self._check_dangerous_patterns(command)
        if dangerous_matches:
            risk_level = "high"
            warnings.extend([f"Dangerous pattern detected: {match}" for match in dangerous_matches])
            if self.security_level == SecurityLevel.STRICT:
                warnings.append("Command blocked by security policy.")
                return CommandResponse(
                    command="",
                    risk_level=risk_level,
                    warnings=warnings,
                )

        # Check moderate patterns
        moderate_matches = self._check_moderate_patterns(command)
        if moderate_matches:
            if risk_level == "low":
                risk_level = "medium"
            warnings.extend([f"Potentially risky: {match}" for match in moderate_matches])

        # Check network patterns
        network_matches = self._check_network_patterns(command)
        if network_matches:
            warnings.extend([f"Network activity detected: {match}" for match in network_matches])

        # Check if command is in whitelist
        base_command = self._get_base_command(command)
        if base_command in self.whitelist:
            risk_level = "low"
            warnings = []  # Clear warnings for whitelisted commands

        # Check if command is in blacklist
        if base_command in self.blacklist:
            risk_level = "critical"
            warnings.append(f"Command '{base_command}' is blacklisted")
            if self.security_level in [SecurityLevel.STRICT, SecurityLevel.MODERATE]:
                warnings.append("Command blocked by security policy.")
                return CommandResponse(
                    command="",
                    risk_level=risk_level,
                    warnings=warnings,
                )

        # Generate safer alternatives
        if risk_level in ["high", "critical"]:
            alternatives = self._suggest_alternatives(command)

        # Add explanation
        explanation = self._generate_explanation(command)

        return CommandResponse(
            command=command,
            explanation=explanation,
            risk_level=risk_level,
            warnings=warnings,
            alternatives=alternatives,
            confidence=0.85,  # Base confidence
        )

    def _is_critical_command(self, command: str) -> bool:
        """Check if command is in critical commands list."""
        normalized = " ".join(command.split())
        return any(critical in normalized for critical in self.CRITICAL_COMMANDS)

    def _check_dangerous_patterns(self, command: str) -> List[str]:
        """Check for dangerous command patterns."""
        matches = []
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                matches.append(pattern)
        return matches

    def _check_moderate_patterns(self, command: str) -> List[str]:
        """Check for moderately risky patterns."""
        matches = []
        for pattern in self.MODERATE_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                matches.append(pattern)
        return matches

    def _check_network_patterns(self, command: str) -> List[str]:
        """Check for network-related patterns."""
        matches = []
        for pattern in self.NETWORK_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                matches.append(pattern)
        return matches

    def _get_base_command(self, command: str) -> str:
        """Extract the base command from a command string."""
        try:
            parts = shlex.split(command)
            if parts:
                # Remove sudo if present
                if parts[0] == "sudo" and len(parts) > 1:
                    return parts[1]
                return parts[0]
        except ValueError:
            # If shlex fails, fall back to simple split
            parts = command.split()
            if parts:
                return parts[0]
        return ""

    def _suggest_alternatives(self, command: str) -> List[str]:
        """Suggest safer alternatives for risky commands."""
        alternatives = []

        # rm alternatives
        if "rm -rf" in command:
            alternatives.append("# Consider using 'trash' or 'rm -i' for interactive deletion")
            alternatives.append("# Or move files to a trash directory: mkdir -p ~/.trash && mv <files> ~/.trash/")

        # sudo alternatives
        if command.startswith("sudo"):
            alternatives.append("# Review if sudo is necessary for this operation")
            alternatives.append("# Consider using capability-based permissions instead")

        # Network alternatives
        if re.search(r"wget.*\|\s*bash", command):
            alternatives.append("# Download first, review, then execute:")
            alternatives.append("wget <url> -O script.sh && less script.sh && bash script.sh")

        return alternatives

    def _generate_explanation(self, command: str) -> str:
        """Generate a brief explanation of what the command does."""
        base_cmd = self._get_base_command(command)

        explanations = {
            "ls": "List directory contents",
            "cd": "Change directory",
            "pwd": "Print working directory",
            "cat": "Display file contents",
            "grep": "Search for patterns in text",
            "find": "Search for files",
            "rm": "Remove files or directories",
            "cp": "Copy files or directories",
            "mv": "Move or rename files",
            "mkdir": "Create directories",
            "chmod": "Change file permissions",
            "chown": "Change file ownership",
            "sudo": "Execute command with elevated privileges",
        }

        return explanations.get(base_cmd, f"Execute '{base_cmd}' command")

    def add_to_whitelist(self, command: str) -> None:
        """Add a command to the whitelist."""
        self.whitelist.add(command)
        logger.info(f"Added '{command}' to whitelist")

    def add_to_blacklist(self, command: str) -> None:
        """Add a command to the blacklist."""
        self.blacklist.add(command)
        logger.info(f"Added '{command}' to blacklist")

    def set_security_level(self, level: SecurityLevel) -> None:
        """Set the security level."""
        self.security_level = level
        logger.info(f"Security level set to: {level.value}")


def sanitize_environment() -> dict:
    """Create a sanitized environment for command execution.

    Returns:
        Dictionary of safe environment variables
    """
    safe_env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/home/user"),
        "USER": os.environ.get("USER", "user"),
        "SHELL": os.environ.get("SHELL", "/bin/bash"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "TERM": os.environ.get("TERM", "xterm-256color"),
    }
    return safe_env


def validate_path(path: str, base_dir: Optional[str] = None) -> Tuple[bool, str]:
    """Validate that a path is safe and within allowed boundaries.

    Args:
        path: The path to validate
        base_dir: Optional base directory to restrict access to

    Returns:
        Tuple of (is_valid, reason)
    """
    try:
        resolved_path = Path(path).resolve()

        # Check if path exists
        if not resolved_path.exists():
            return False, "Path does not exist"

        # If base_dir is specified, ensure path is within it
        if base_dir:
            base_path = Path(base_dir).resolve()
            if not str(resolved_path).startswith(str(base_path)):
                return False, f"Path is outside allowed directory: {base_dir}"

        # Check for suspicious patterns
        path_str = str(resolved_path)
        if any(suspicious in path_str for suspicious in ["/dev/", "/proc/", "/sys/"]):
            return False, "Access to system directories is restricted"

        return True, "Path is valid"

    except Exception as e:
        return False, f"Path validation error: {str(e)}"
