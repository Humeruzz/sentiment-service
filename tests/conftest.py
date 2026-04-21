import os

# mlflow's opentelemetry-proto uses old-style protobuf generated code
# incompatible with protobuf>=4. The pure-Python impl handles it correctly.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
