import subprocess

RABINIZER_PATH = 'rabinizer4/bin/ltl2ldba'


def run_rabinizer(formula: str) -> str:
    """Convert an LTL formula to a LDBA in the HOA format."""
    # -p: parallel processing
    # -d: construct a non-generalised Buechi automaton
    # -e: keep generated epsilon transitions
    command = [RABINIZER_PATH, '-i', formula, '-p', '-d', '-e']
    run = subprocess.run(command, capture_output=True, text=True)
    stderr_filtered = run.stderr.replace("Picked up JAVA_TOOL_OPTIONS: -Xmx2048m\n", "")
    if stderr_filtered.strip() != '':
        raise RuntimeError(f'Rabinizer call `{" ".join(command)}` resulted in an error.\nError: {stderr_filtered}.')
    return run.stdout


if __name__ == '__main__':
    f = 'FG a'
    ldba = run_rabinizer(f)
    print(ldba)
