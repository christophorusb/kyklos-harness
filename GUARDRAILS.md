# Default Safety & Execution Guardrails

Follow these operational boundaries on every task unless the user explicitly instructs otherwise:

1. **Zero Privilege Escalation:** Never run `sudo`, `su`, `doas`, `pkexec`, `dscl`, `security`, or `osascript` privilege prompts, and never use debugger or kernel attachment APIs (`task_for_pid`, `ptrace`, `dtrace`, `kauth`).
2. **Hands Off Host Credentials & Personal Data:** Never list, read, copy, or modify real host credential stores (`~/.ssh`, `~/.aws`, `~/.kube`, `~/.gnupg`, `~/.config`, `~/Library/Keychains`, `.env`, `.npmrc`, `.netrc`, `.git-credentials`) or personal folders outside the workspace.
3. **Isolated Sandboxes for Path & Home Tests:** Write throwaway diagnostic scripts to `.kyklos/scratch/` so project imports resolve. Whenever testing code that touches `$HOME`, configuration dirs, or credential paths, create an isolated temporary directory (`mktemp -d`) with synthetic test fixtures and pass it via `--home`, `HOME=...`, or dependency injection.
4. **Non-Hanging Background Processes:** In `bash`, any backgrounded process (`cmd &`) inherits open stdout/stderr pipes and will hang the tool call until timeout unless redirected. Always redirect both streams (`cmd > /tmp/run.log 2>&1 &`), capture `$!`, and clean up the specific PID (`kill "$PID"`) before finishing. Never run broad `killall` or `pkill` against common binary names.
5. **Minimal Host Recon & GUI Noise:** Avoid broad host enumeration (`ps aux`, unfiltered `lsof`, `launchctl list`, `crontab -l`) when a targeted check (`ps -p <pid>`) suffices. For GUI or menu-bar apps, prefer headless CLI flags (`--self-test`, `--status`) over launching interactive UI loops during automated verification.
6. **Safe File Hygiene:** Create directories holding sensitive or state data with mode `0700` and files with mode `0600`, using atomic temp-file rename and `O_NOFOLLOW` where supported.
