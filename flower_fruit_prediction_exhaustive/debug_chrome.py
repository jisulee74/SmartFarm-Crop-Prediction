import subprocess
import time
import json
import urllib.request
import base64
import socket

# 1. Start Chrome
chrome_path = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
port = 9222
proc = subprocess.Popen([
    chrome_path,
    '--headless=new',
    f'--remote-debugging-port={port}',
    '--disable-gpu',
    '--no-sandbox',
    '--window-size=1600,1200',
    'https://jisulee74.github.io/SmartFarm-Crop-Prediction/flower_fruit_prediction/'
])

time.sleep(3)

try:
    res = urllib.request.urlopen(f'http://localhost:{port}/json')
    tabs = json.loads(res.read().decode())
    target_tab = [t for t in tabs if t.get('type') == 'page'][0]
    ws_url = target_tab['webSocketDebuggerUrl']
    print('Found Page Target:', ws_url)
    
    # Simple websocket client using standard socket
    # Or even simpler: use CDP HTTP endpoint /json/new or evaluate via devtools
    # Let's see if we can use CDP over websocket with a minimal frame implementation
except Exception as e:
    print('Error:', e)
finally:
    proc.terminate()
