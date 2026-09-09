import http.server
import socketserver
import os
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

class ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

def run_server(port=PORT):
    for p in range(port, port + 50):
        try:
            with ThreadingServer(("", p), Handler) as httpd:
                url = f"http://localhost:{p}"
                print("=======================================================")
                print(f"[Dashboard Server] Started successfully")
                print(f"URL: {url}")
                print(f"Directory: {DIRECTORY}")
                print("Press Ctrl+C to terminate.")
                print("=======================================================", flush=True)
                httpd.serve_forever()
                break
        except OSError:
            continue

if __name__ == "__main__":
    run_server()
