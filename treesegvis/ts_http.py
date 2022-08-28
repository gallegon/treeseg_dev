import sys
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

sys.path.append(os.path.abspath(os.path.join("..", "python")))
from treesegmentation.ts_api import *

class TSHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        print("GET request!")
        super().do_GET()
    
    def do_POST(self):
        data_len = len(self.headers["Content-Length"])
        data = self.rfile(data_len)
        print("=== POST REQUEST")
        print("--- POST PATH")
        print(self.path)
        print("--- POST DATA")
        print(data)
        print("---")
        
        # result = default_pipeline(context)
        # print("Ran pipeline!")
        super().do_POST()


def tsserver(port=8080):
    with ThreadingHTTPServer(("localhost", port), TSHandler) as server:
        server_host, server_port = server.server_address
        print(f"Tree segmentation server started on {server_host}:{server_port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopping tree segmentation server")

if __name__ == "__main__":
    tsserver(8080)
