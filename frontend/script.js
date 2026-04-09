const API_BASE = "http://localhost:8000";
const WS_BASE = "ws://localhost:8000";

// Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const selectedFilename = document.getElementById('selected-filename');
const processBtn = document.getElementById('process-btn');
const uploadSection = document.getElementById('upload-section');
const processingView = document.getElementById('processing-view');
const resultsSection = document.getElementById('results-section');
const liveImageContainer = document.getElementById('live-image-container');
const logFeed = document.getElementById('log-feed');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const resultsBody = document.getElementById('results-body');
const connectionStatus = document.getElementById('connection-status');
const totalResultsCount = document.getElementById('total-results-count');

let currentFile = null;
let resultsData = [];
let ws = null;

// Initialization
function init() {
    setupEventListeners();
    checkServerConnection();
}

function checkServerConnection() {
    fetch(`${API_BASE}/`)
        .then(() => {
            connectionStatus.classList.remove('disconnected');
            connectionStatus.classList.add('connected');
            connectionStatus.innerHTML = '<span class="dot"></span> Online';
        })
        .catch(() => {
            connectionStatus.classList.remove('connected');
            connectionStatus.classList.add('disconnected');
            connectionStatus.innerHTML = '<span class="dot"></span> Server Offline';
        });
}

function setupEventListeners() {
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelect(e.target.files[0]);
        }
    });

    processBtn.addEventListener('click', startAnalysis);

    document.getElementById('download-csv').addEventListener('click', () => downloadResults('csv'));
    document.getElementById('download-excel').addEventListener('click', () => downloadResults('excel'));
}

function handleFileSelect(file) {
    if (file.type !== 'application/pdf') {
        showToast("Please select a PDF file", "error");
        return;
    }
    currentFile = file;
    selectedFilename.textContent = file.name;
    fileInfo.classList.remove('hidden');
    processBtn.classList.remove('hidden');
    processBtn.disabled = false;
    showToast("File selected: " + file.name, "info");
}

async function startAnalysis() {
    if (!currentFile) return;

    // 1. Upload File
    addLog("Uploading file to server...", "info");
    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        const filename = data.filename;

        // 2. Clear UI for processing
        uploadSection.classList.add('hidden');
        processingView.classList.remove('hidden');
        resultsData = [];
        resultsBody.innerHTML = '';
        logFeed.innerHTML = '';
        liveImageContainer.innerHTML = '<div class="placeholder"><p>Initializing YOLO model...</p></div>';

        // 3. Connect WebSocket
        connectWebSocket(filename);

    } catch (error) {
        addLog(`Upload failed: ${error.message}`, "error");
        showToast("Failed to upload file", "error");
    }
}

function connectWebSocket(filename) {
    ws = new WebSocket(`${WS_BASE}/ws/process`);

    ws.onopen = () => {
        addLog("WebSocket connected. Starting detection pipeline...", "success");
        ws.send(JSON.stringify({ filename }));
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleWsMessage(msg);
    };

    ws.onerror = (error) => {
        addLog("WebSocket Error: " + error.message, "error");
    };

    ws.onclose = () => {
        addLog("Analysis stream closed.", "info");
    };
}

function handleWsMessage(msg) {
    switch (msg.status) {
        case 'starting':
            addLog(msg.message, 'info');
            break;
        case 'info':
            addLog(msg.message, 'info');
            break;
        case 'progress':
            const percent = Math.round((msg.page / msg.total_pages) * 100);
            updateProgress(percent, msg.message);
            addLog(msg.message, 'processing');
            break;
        case 'detection_image':
            updateLiveImage(`${API_BASE}${msg.image_url}`);
            addLog(`Page ${msg.page}: YOLO found ${msg.count} candidate areas.`, 'success');
            break;
        case 'extraction':
            resultsData.push(msg.data);
            addResultRow(msg.data);
            updateResultsCount();
            break;
        case 'complete':
            addLog(msg.message, 'success');
            showToast("Analysis Complete!", "success");
            resultsSection.classList.remove('hidden');
            updateProgress(100, "Done");
            break;
        case 'error':
            addLog(msg.message, 'error');
            showToast("Error: " + msg.message, "error");
            break;
    }
}

function addLog(text, type = 'info') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    const time = new Date().toLocaleTimeString([], { hour12: false });
    entry.innerHTML = `<span style="opacity: 0.5">[${time}]</span> ${text}`;
    logFeed.appendChild(entry);
    logFeed.scrollTop = logFeed.scrollHeight;
}

function updateProgress(percent, text) {
    progressFill.style.width = `${percent}%`;
    progressText.textContent = `${percent}%`;
}

function updateLiveImage(url) {
    liveImageContainer.innerHTML = `<img src="${url}?t=${Date.now()}" alt="Detection">`;
}

function addResultRow(data) {
    const row = document.createElement('tr');
    
    const confClass = data.confidence > 0.8 ? 'conf-high' : (data.confidence > 0.5 ? 'conf-med' : 'conf-low');
    
    row.innerHTML = `
        <td>${data.id}</td>
        <td>${data.project_name || '-'}</td>
        <td>${data.line_number || '-'}</td>
        <td>${data.diameter || '-'}</td>
        <td>${data.service_code || '-'}</td>
        <td>${data.piping_class || '-'}</td>
        <td>${data.bop_elevation || '-'}</td>
        <td><span class="conf-pill ${confClass}">${(data.confidence * 100).toFixed(1)}%</span></td>
        <td title="${data.text}">${truncate(data.text, 30)}</td>
    `;
    resultsBody.appendChild(row);
}

function updateResultsCount() {
    totalResultsCount.textContent = `Extracted ${resultsData.length} specifications across pages`;
}

function truncate(str, n) {
    return (str.length > n) ? str.substr(0, n - 1) + '&hellip;' : str;
}

async function downloadResults(format) {
    if (!currentFile) return;
    const url = `${API_BASE}/download/${format}/${currentFile.name}`;
    window.open(url, '_blank');
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.getElementById('toast-container').appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// Toast Styling (Dynamic injection)
const style = document.createElement('style');
style.textContent = `
    #toast-container { position: fixed; bottom: 20px; right: 20px; z-index: 1000; display: flex; flex-direction: column; gap: 10px; }
    .toast { padding: 12px 24px; border-radius: 8px; color: white; font-weight: 500; animation: slideIn 0.3s ease-out; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
    .toast-info { background: #2f81f7; }
    .toast-success { background: #3fb950; }
    .toast-error { background: #f85149; }
    @keyframes slideIn { from { transform: translateX(100%); } to { transform: translateX(0); } }
`;
document.head.appendChild(style);

init();
