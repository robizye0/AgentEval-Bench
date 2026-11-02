import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def test_cli_tracer_exports_otlp_when_endpoint_is_configured():
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    received = []

    class ExporterHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(
                (self.path, self.rfile.read(int(self.headers.get("Content-Length", 0))))
            )
            self.send_response(200)
            self.end_headers()

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), ExporterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    script = (
        "from agenteval.integration.langfuse_tracer import EvaluationTracer; "
        "t=EvaluationTracer('test'); t.record_result('case', True, 1.0, 12, 0.01, 4); "
        "t.record_tool_call('case', 'search', 12); assert t.flush()"
    )
    env = {**os.environ, "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint}
    try:
        subprocess.run([sys.executable, "-c", script], env=env, check=True, timeout=15)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert received
    assert all(path == "/v1/traces" and body for path, body in received)
