import subprocess
import time
import json
import urllib.request
import socket
import os
import hashlib
import base64
import struct
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def make_ws_handshake(sock, host, port, path):
    key = base64.b64encode(os.urandom(16)).decode('utf-8')
    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: localhost:{port}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n\r\n"
    )
    sock.sendall(req.encode('utf-8'))
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += sock.recv(4096)
    if b"101" not in resp:
        print("Handshake response:", resp.decode('utf-8', errors='ignore'))
    assert b"101" in resp, "Handshake failed"

def send_frame(sock, message):
    data = message.encode('utf-8')
    length = len(data)
    frame = bytearray([0x81]) # FIN + text
    mask = os.urandom(4)
    if length <= 125:
        frame.append(0x80 | length)
    elif length <= 65535:
        frame.append(0x80 | 126)
        frame.extend(struct.pack("!H", length))
    else:
        frame.append(0x80 | 127)
        frame.extend(struct.pack("!Q", length))
    frame.extend(mask)
    masked_data = bytearray(b ^ mask[i % 4] for i, b in enumerate(data))
    frame.extend(masked_data)
    sock.sendall(frame)

def recv_frame(sock):
    header = sock.recv(2)
    if len(header) < 2:
        return None
    b1, b2 = header[0], header[1]
    length = b2 & 0x7F
    if length == 126:
        length = struct.unpack("!H", sock.recv(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", sock.recv(8))[0]
    data = bytearray()
    while len(data) < length:
        chunk = sock.recv(min(4096, length - len(data)))
        if not chunk:
            break
        data.extend(chunk)
    return data.decode('utf-8', errors='ignore')

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
    ws_path = ws_url.split(f':{port}')[1]

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('127.0.0.1', port))
    make_ws_handshake(sock, '127.0.0.1', port, ws_path)
    print("WebSocket connected to Chrome!")

    # Enable console & runtime
    send_frame(sock, json.dumps({"id": 1, "method": "Runtime.enable"}))
    send_frame(sock, json.dumps({"id": 2, "method": "Console.enable"}))

    # Click tab 3 (생육 예측 분석)
    click_js = """
    (async () => {
        const btn = document.querySelector('button[data-tab="tab-analysis"]');
        if (btn) btn.click();
        await new Promise(r => setTimeout(r, 2000));
        
        const perfPane = document.getElementById('anl-subtab-perf');
        const axPane = document.getElementById('anl-subtab-ax');
        const kpiPane = document.getElementById('anl-subtab-kpi');

        return {
            activeTab: document.querySelector('.primary-tab-btn.active')?.dataset.tab,
            tabAnalysisActive: document.getElementById('tab-analysis')?.classList.contains('active'),
            // Subtab 1: Performance
            perf: {
                summaryRmse: document.getElementById('anl_summary_rmse')?.textContent,
                summaryMae: document.getElementById('anl_summary_mae')?.textContent,
                summaryR2: document.getElementById('anl_summary_r2')?.textContent,
                summaryCcc: document.getElementById('anl_summary_ccc')?.textContent,
                targetDistTitle: document.querySelector('#anl_target_dist_section .anl-dist-title')?.textContent,
                distWarnVisible: document.getElementById('anl_dist_warning_card')?.style.display !== 'none',
                compCaption: document.getElementById('anl_comp_footer_caption')?.textContent,
                histCaption: document.getElementById('anl_hist_footer_caption')?.textContent,
                modelCompChart: !!document.querySelector('#anl_chart_model_comp canvas'),
                strataChart: !!document.querySelector('#anl_chart_strata canvas'),
                sseChart: !!document.querySelector('#anl_chart_sse canvas'),
                diagContent: document.getElementById('anl_diagnosis_content')?.textContent?.trim()?.substring(0, 80),
                baselineRows: document.querySelectorAll('#anl_baseline_tbody tr')?.length,
                strataRows: document.querySelectorAll('#anl_strata_tbody tr')?.length,
            },
            // Subtab 2: AX Roadmap
            ax: {
                hypListItems: document.querySelectorAll('#anl_hyp_list .anl-hyp-item')?.length,
                hypDetailContent: document.getElementById('anl_hyp_detail_panel')?.textContent?.trim()?.substring(0, 80),
                schemaTabs: document.querySelectorAll('#anl_schema_tabs button')?.length,
                schemaRows: document.querySelectorAll('#anl_schema_tbody tr')?.length,
            },
            // Subtab 3: Verification & KPI
            kpi: {
                pipelineArms: document.querySelectorAll('#anl_arms_container .anl-arm-card')?.length,
                armDetail: document.getElementById('anl_arm_detail_panel')?.textContent?.trim()?.substring(0, 80),
                targetKpiItems: document.querySelectorAll('#anl_target_kpi_grid .anl-target-kpi-item')?.length,
                kpiTableRows: document.querySelectorAll('#anl_kpi_tbody tr')?.length,
            },
            consoleErrors: window.__consoleErrors || [],
            pageTitle: document.title
        };
    })()
    """
    
    # Inject console error trap first
    send_frame(sock, json.dumps({
        "id": 3,
        "method": "Runtime.evaluate",
        "params": {
            "expression": "window.__consoleErrors = []; window.addEventListener('error', e => window.__consoleErrors.push(e.message));"
        }
    }))

    time.sleep(1)

    send_frame(sock, json.dumps({
        "id": 4,
        "method": "Runtime.evaluate",
        "params": {
            "expression": click_js,
            "awaitPromise": True,
            "returnByValue": True
        }
    }))

    # Read messages
    sock.settimeout(5.0)
    for _ in range(15):
        try:
            msg = recv_frame(sock)
            if not msg:
                continue
            data = json.loads(msg)
            if data.get('id') == 4:
                print("Result of tab-analysis inspection:")
                print(json.dumps(data.get('result', {}).get('result', {}).get('value', {}), indent=2, ensure_ascii=False))
                break
        except Exception as e:
            print("Recv err:", e)
            break

finally:
    try:
        sock.close()
    except:
        pass
    proc.terminate()
