import subprocess
import sys
def check_licenses():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pip-licenses"])
        subprocess.check_call([sys.executable, "-m", "piplicenses"])
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_licenses()
