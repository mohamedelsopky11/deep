#!/bin/bash
# Usage: ./scripts/run_inference.sh <image_path>
python -c "from src.inference import FireRiskSystem; sys = FireRiskSystem(); print(sys.run_inference('$1'))"
