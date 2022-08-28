import base64
import json
import os
import socket
import sys

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

# Change module search path at runtime to find Python module.
sys.path.append(os.path.abspath(os.path.join("..", "python")))
from treesegmentation.ts_api import *
from integration_with_c import c_pipeline


class TSHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve files from the src/ directory
        dir = os.fspath(os.path.join(os.getcwd(), "src"))
        super().__init__(*args, **{**kwargs, "directory": dir})
    
    def do_POST(self):
        # Only run on the appropriate request
        if self.path != "/treeseg-run":
            self.send_response_only(200)
            self.end_headers()
            return
        
        # Parse JSON context object
        data_len = int(self.headers["Content-Length"])
        data = self.rfile.read(data_len)
        context = json.loads(data)

        # Run the pipeline locally
        # result = default_pipeline(obj)
        result = c_pipeline(context)

        # Load images from disk
        p_height = result["save_grid_path"]
        p_patch = result["save_patches_path"]
        p_hierarchy = result["save_partition_path"]
        with open(p_height, "rb") as height, open(p_patch, "rb") as patch, open(p_hierarchy, "rb") as hierarchy:
            # Convert image data to base 64 encoded string
            data_height = str(base64.b64encode(height.read()))[2:-1]
            data_patch = str(base64.b64encode(patch.read()))[2:-1]
            data_hierarchy = str(base64.b64encode(hierarchy.read()))[2:-1]

        # Send encoded image data to client
        resp = json.dumps({
            "elapsed_time": result["elapsed_time"],
            "data-grid-height": data_height,
            "data-grid-patch": data_patch,
            "data-grid-hierarchy": data_hierarchy
        })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(bytes(resp, "utf-8"))


def tsserver(port=8080):
    # ThreadingHTTPServer allows KeyboardInterrupts to occur asynchronously.
    with ThreadingHTTPServer(("localhost", port), TSHandler) as server:
        server_host, server_port = server.server_address
        print(f"Tree segmentation server started on {server_host}:{server_port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopping tree segmentation server")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        tsserver(int(sys.argv[1]))
    else:
        tsserver()
