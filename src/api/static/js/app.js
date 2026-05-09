// Chigma Frontend Logic — Production Dashboard
// ================================================

document.addEventListener('DOMContentLoaded', () => {

    // --- Navigation ---
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view-section');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const viewId = item.getAttribute('data-view');

            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');

            views.forEach(view => view.classList.remove('active'));
            document.getElementById(`view-${viewId}`).classList.add('active');
            lucide.createIcons();
        });
    });

    // --- Health Check Polling ---
    async function checkHealth() {
        const dot = document.querySelector('.pulse-dot');
        const text = document.getElementById('health-text');
        const gpuText = document.getElementById('metric-gpu');

        try {
            const res = await fetch('/api/v1/health');
            const data = await res.json();

            if (data.status === 'healthy') {
                dot.classList.remove('error');
                text.textContent = 'System Online';
                if (data.gpu) {
                    gpuText.textContent = `${data.gpu} (${data.gpu_memory_gb}GB)`;
                } else {
                    gpuText.textContent = 'CPU Inference Mode';
                }
            } else {
                throw new Error();
            }
        } catch (e) {
            dot.classList.add('error');
            text.textContent = 'API Offline';
            gpuText.textContent = 'Offline';
            document.getElementById('metric-status').textContent = 'Offline';
            document.getElementById('metric-status').style.color = 'var(--danger)';
            document.getElementById('metric-status').classList.remove('text-gradient');
        }
    }

    checkHealth();
    setInterval(checkHealth, 30000);

    // --- Sample Image Loader ---
    window.loadSample = async function(inputId, src, dropzoneId, previewId, contentId) {
        try {
            const res = await fetch(src);
            const blob = await res.blob();
            const file = new File([blob], 'sample.png', { type: 'image/png' });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            document.getElementById(inputId).files = dataTransfer.files;

            if (previewId) {
                const prev = document.getElementById(previewId);
                prev.src = src;
                prev.style.display = 'block';
                if (contentId) document.getElementById(contentId).style.display = 'none';
            } else {
                const dz = document.getElementById(dropzoneId);
                const icon = dz.querySelector('i');
                const label = dz.querySelector('p');
                if (icon) icon.style.display = 'none';
                if (label) label.textContent = 'sample.png (loaded)';
            }
        } catch (err) {
            console.error('Failed to load sample:', err);
        }
    };

    // --- File Dropzones ---
    function setupDropzone(zoneId, inputId, previewId, contentId) {
        const dropzone = document.getElementById(zoneId);
        const input = document.getElementById(inputId);
        if (!dropzone || !input) return;

        dropzone.addEventListener('click', () => input.click());

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                input.files = e.dataTransfer.files;
                triggerPreview();
            }
        });

        input.addEventListener('change', triggerPreview);

        function triggerPreview() {
            if (input.files && input.files[0] && previewId) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const img = document.getElementById(previewId);
                    img.src = e.target.result;
                    img.style.display = 'block';
                    if (contentId) {
                        document.getElementById(contentId).style.display = 'none';
                    }
                };
                reader.readAsDataURL(input.files[0]);
            } else if (input.files && input.files[0]) {
                const icon = dropzone.querySelector('i');
                const label = dropzone.querySelector('p');
                if (icon) icon.style.display = 'none';
                if (label) label.textContent = input.files[0].name;
            }
        }
    }

    setupDropzone('detection-dropzone', 'detection-file', null, null);
    setupDropzone('vlm-dropzone', 'vlm-file', 'vlm-preview', 'vlm-drop-content');

    // --- Slider Update ---
    const confSlider = document.getElementById('conf-slider');
    const confVal = document.getElementById('conf-val');
    if (confSlider && confVal) {
        confSlider.addEventListener('input', (e) => {
            confVal.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    // ============================================
    // DETECTION MODULE
    // ============================================
    const detForm = document.getElementById('detection-form');
    if (detForm) {
        detForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const fileInput = document.getElementById('detection-file');
            if (!fileInput.files.length) {
                alert('Please select an image first.');
                return;
            }

            const submitBtn = detForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i data-lucide="loader" class="spin"></i> Processing...';
            lucide.createIcons();
            submitBtn.disabled = true;

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            const conf = confSlider ? confSlider.value : '0.25';

            try {
                const res = await fetch(`/api/v1/detection/predict?conf=${conf}&iou=0.45`, {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                if (!res.ok) throw new Error(data.detail || 'API Error');

                drawDetectionResults(fileInput.files[0], data);

            } catch (error) {
                alert('Detection Error: ' + error.message);
            } finally {
                submitBtn.innerHTML = originalText;
                submitBtn.disabled = false;
                lucide.createIcons();
            }
        });
    }

    function drawDetectionResults(file, data) {
        const placeholder = document.querySelector('.placeholder-text');
        if (placeholder) placeholder.style.display = 'none';

        const imgEl = document.getElementById('detection-preview');
        const canvas = document.getElementById('detection-canvas');
        const ctx = canvas.getContext('2d');
        const stats = document.getElementById('detection-stats');

        const reader = new FileReader();
        reader.onload = (e) => {
            imgEl.onload = () => {
                canvas.width = imgEl.clientWidth;
                canvas.height = imgEl.clientHeight;

                const scaleX = canvas.width / data.image_shape[1];
                const scaleY = canvas.height / data.image_shape[0];

                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.lineWidth = 2;
                ctx.font = '14px Inter';

                data.detections.forEach(det => {
                    const [x1, y1, x2, y2] = det.bbox;
                    const cx1 = x1 * scaleX;
                    const cy1 = y1 * scaleY;
                    const w = (x2 - x1) * scaleX;
                    const h = (y2 - y1) * scaleY;

                    ctx.strokeStyle = '#ef4444';
                    ctx.fillStyle = 'rgba(239, 68, 68, 0.15)';
                    ctx.fillRect(cx1, cy1, w, h);
                    ctx.strokeRect(cx1, cy1, w, h);

                    const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
                    ctx.fillStyle = '#ef4444';
                    const textWidth = ctx.measureText(label).width;
                    ctx.fillRect(cx1, cy1 - 20, textWidth + 10, 20);
                    ctx.fillStyle = '#ffffff';
                    ctx.fillText(label, cx1 + 5, cy1 - 5);
                });

                stats.innerHTML = `
                    <strong>Results:</strong> Found ${data.num_detections} defects.<br>
                    <span style="color:var(--text-secondary)">Inference Latency: ${data.latency_ms.toFixed(1)} ms</span>
                `;
            };
            imgEl.src = e.target.result;
            imgEl.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    // ============================================
    // VLM COPILOT MODULE
    // ============================================
    const vlmForm = document.getElementById('vlm-form');
    const chatContainer = document.getElementById('vlm-chat');

    function appendChatMsg(sender, text, isHtml) {
        if (!chatContainer) return;
        isHtml = isHtml || false;
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-msg ${sender}-msg`;
        const icon = sender === 'system' ? 'bot' : 'user';
        msgDiv.innerHTML = `
            <i data-lucide="${icon}"></i>
            <div class="msg-bubble">${isHtml ? text : escapeHtml(text)}</div>
        `;
        chatContainer.appendChild(msgDiv);
        lucide.createIcons();
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    if (vlmForm) {
        vlmForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const fileInput = document.getElementById('vlm-file');
            const queryInput = document.getElementById('vlm-query');

            if (!fileInput.files.length) {
                alert('Please select an image for context.');
                return;
            }

            const queryText = queryInput.value;
            appendChatMsg('user', queryText);
            queryInput.value = '';

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);

            const typingId = 'typing-' + Date.now();
            const typingDiv = document.createElement('div');
            typingDiv.className = 'chat-msg system-msg';
            typingDiv.id = typingId;
            typingDiv.innerHTML = '<i data-lucide="bot"></i><div class="msg-bubble">Thinking... <i data-lucide="loader" class="spin"></i></div>';
            chatContainer.appendChild(typingDiv);
            lucide.createIcons();
            chatContainer.scrollTop = chatContainer.scrollHeight;

            try {
                const url = new URL(window.location.origin + '/api/v1/vlm/query');
                url.searchParams.append('query', queryText);

                const res = await fetch(url, {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                document.getElementById(typingId).remove();

                if (!res.ok) throw new Error(data.detail || 'API Error');

                appendChatMsg('system', data.response);

            } catch (error) {
                const el = document.getElementById(typingId);
                if (el) el.remove();
                appendChatMsg('system', 'Error: ' + error.message);
            }
        });
    }

    // --- VLM Report Button ---
    const vlmReportBtn = document.getElementById('vlm-report-btn');
    if (vlmReportBtn) {
        vlmReportBtn.addEventListener('click', async () => {
            const fileInput = document.getElementById('vlm-file');
            if (!fileInput.files.length) {
                alert('Please select an image first to generate a report.');
                return;
            }

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);

            appendChatMsg('user', 'Generate automated inspection report.');

            const typingId = 'typing-' + Date.now();
            const typingDiv = document.createElement('div');
            typingDiv.className = 'chat-msg system-msg';
            typingDiv.id = typingId;
            typingDiv.innerHTML = '<i data-lucide="bot"></i><div class="msg-bubble">Generating comprehensive report... <i data-lucide="loader" class="spin"></i></div>';
            chatContainer.appendChild(typingDiv);
            lucide.createIcons();

            try {
                const res = await fetch('/api/v1/vlm/report', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                document.getElementById(typingId).remove();

                if (!res.ok) throw new Error(data.detail || 'API Error');

                const badgeClass = data.pass_fail === 'PASS' ? 'badge-pass' : 'badge-fail';

                const reportHtml = `
                    <div class="report-card">
                        <span class="report-badge ${badgeClass}">${data.pass_fail}</span>
                        <br><strong>ID:</strong> ${data.report_id}
                        <br><strong>Severity:</strong> <span style="text-transform:capitalize">${data.assessment.severity || 'N/A'}</span>
                        <br><br><strong>Analysis:</strong><br>
                        ${data.assessment.raw_assessment}
                    </div>
                `;

                appendChatMsg('system', reportHtml, true);

            } catch (error) {
                const el = document.getElementById(typingId);
                if (el) el.remove();
                appendChatMsg('system', 'Error: ' + error.message);
            }
        });
    }

    // ============================================
    // DRONE AUTONOMY MODULE
    // ============================================
    const droneConsole = document.getElementById('drone-console');
    const droneForm = document.getElementById('drone-form');

    function logDrone(msg, isErr) {
        if (!droneConsole) return;
        isErr = isErr || false;
        const d = document.createElement('div');
        d.style.color = isErr ? 'var(--danger)' : '#a3b8cc';
        d.textContent = '> ' + msg;
        droneConsole.appendChild(d);
        droneConsole.scrollTop = droneConsole.scrollHeight;
    }

    if (droneForm) {
        droneForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const input = document.getElementById('drone-cmd');
            const cmd = input.value;
            input.value = '';
            logDrone('CMD: ' + cmd);

            try {
                const res = await fetch('/api/v1/drone/command/natural', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: cmd })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'API Error');
                logDrone('ACK: ' + data.message);
                logDrone('ACT: ' + data.parsed_action.toUpperCase());
            } catch (err) {
                logDrone('ERR: ' + err.message, true);
            }
        });
    }

    const btnHome = document.getElementById('btn-drone-home');
    if (btnHome) {
        btnHome.addEventListener('click', async () => {
            logDrone('CMD: RETURN TO HOME');
            try {
                const res = await fetch('/api/v1/drone/command/return-home', { method: 'POST' });
                const data = await res.json();
                logDrone('ACK: ' + data.message);
            } catch (err) {
                logDrone('ERR: ' + err.message, true);
            }
        });
    }

    const btnStop = document.getElementById('btn-drone-stop');
    if (btnStop) {
        btnStop.addEventListener('click', async () => {
            logDrone('CMD: EMERGENCY STOP');
            try {
                const res = await fetch('/api/v1/drone/command/emergency-stop', { method: 'POST' });
                const data = await res.json();
                logDrone('ACK: ' + data.message, true);
            } catch (err) {
                logDrone('ERR: ' + err.message, true);
            }
        });
    }

    // --- Telemetry Polling ---
    setInterval(async () => {
        const droneView = document.getElementById('view-drone');
        if (!droneView || !droneView.classList.contains('active')) return;
        try {
            const res = await fetch('/api/v1/drone/status');
            if (res.ok) {
                const data = await res.json();
                document.getElementById('drone-mode').textContent = data.mode;
                document.getElementById('drone-alt').textContent = data.altitude_m.toFixed(1);
                document.getElementById('drone-spd').textContent = data.speed_ms.toFixed(1);
                document.getElementById('drone-bat').textContent = data.battery_pct.toFixed(0);
            }
        } catch (e) { /* silently skip if offline */ }
    }, 2000);

});