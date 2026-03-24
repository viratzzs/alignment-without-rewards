import sys

try:
    from recipe.gkd import main_gkd
except ImportError as e:
    print(f"Warning: could not import VERL's main_gkd: {e}")
    def main_execution():
        print("VERL not found, terminating custom execution wrapper block.")
        sys.exit(1)
    main_gkd = type('obj', (object,), {'main': main_execution})

# The actual SubnetworkLogger has been natively injected into FSDP workers
# at /workspace/verl/verl/workers/fsdp_workers.py to execute directly on the Neural Network!
if __name__ == "__main__":
    main_gkd.main()
