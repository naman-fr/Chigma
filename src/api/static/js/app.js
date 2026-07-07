// Chigma DRDO Tactical Frontend Control System
// ===================================================

document.addEventListener('DOMContentLoaded', () => {
    // --- Authentication Clearance Store ---
    let jwtToken = localStorage.getItem('chigma_tactical_token') || null;
    let userRole = localStorage.getItem('chigma_tactical_role') || null;
    let username = localStorage.getItem('chigma_tactical_username') || null;

    // UI Elements
    const authGate = document.getElementById('auth-gate');
    const appMain = document.getElementById('app-main');
    const loginForm = document.getElementById('login-form');
    const userRoleEl = document.getElementById('user-role');
    const userNameEl = document.getElementById('user-name');
    const tacticalRoleDisplay = document.getElementById('tactical-role-display');
    const logoutBtn = document.getElementById('btn-logout');

    // Pre-populate demo credentials
    window.setDemoCreds = function(user, pass) {
        document.getElementById('username').value = user;
        document.getElementById('password').value = pass;
    };

    // Authenticated API Fetch wrapper
    async function authFetch(url, options = {}) {
        options.headers = options.headers || {};
        if (jwtToken) {
            options.headers['Authorization'] = `Bearer ${jwtToken}`;
        }
        
        try {
            const res = await fetch(url, options);
            if (res.status === 401) {
                // Revoke session if invalid credentials
                revokeClearance();
                throw new Error("Clearance session expired. Re-authorization required.");
            }
            return res;
        } catch (err) {
            console.error("Secure network request failed:", err);
            throw err;
        }
    }

    // Initialize Security Session
    function initializeSession() {
        if (jwtToken && userRole && username) {
            authGate.style.display = 'none';
            appMain.style.display = 'flex';
            
            // Format role display
            userRoleEl.textContent = userRole.toUpperCase();
            userNameEl.textContent = username === 'drdo_commander' ? 'COMMANDER (DRDO)' : 'OPERATOR (ARMY)';
            if (tacticalRoleDisplay) {
                tacticalRoleDisplay.textContent = userRole;
            }

            // Hide/Show Commander-only options
            const cmdOnlyElements = document.querySelectorAll('.commander-only');
            cmdOnlyElements.forEach(el => {
                if (userRole === 'Commander') {
                    el.style.display = 'block';
                } else {
                    el.style.display = 'none';
                }
            });

            checkHealth();
            lucide.createIcons();
        } else {
            authGate.style.display = 'flex';
            appMain.style.display = 'none';
        }
    }

    // Revoke Session
    function revokeClearance() {
        localStorage.removeItem('chigma_tactical_token');
        localStorage.removeItem('chigma_tactical_role');
        localStorage.removeItem('chigma_tactical_username');
        jwtToken = null;
        userRole = null;
        username = null;
        initializeSession();
    }

    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            revokeClearance();
        });
    }

    // Login Form Submit handler
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const u = document.getElementById('username').value;
            const p = document.getElementById('password').value;

            const submitBtn = loginForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i data-lucide="loader" class="spin"></i> AUTHORIZING...';
            lucide.createIcons();
            submitBtn.disabled = true;

            try {
                // URL Encoded form data for OAuth2 spec
                const bodyParams = new URLSearchParams();
                bodyParams.append('username', u);
                bodyParams.append('password', p);

                const res = await fetch('/api/v1/auth/token', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: bodyParams
                });

                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Access Denied');

                // Save parameters
                jwtToken = data.access_token;
                userRole = data.role;
                username = data.username;

                localStorage.setItem('chigma_tactical_token', jwtToken);
                localStorage.setItem('chigma_tactical_role', userRole);
                localStorage.setItem('chigma_tactical_username', username);

                initializeSession();
            } catch (err) {
                alert('Clearance Authorization Failure: ' + err.message);
            } finally {
                submitBtn.innerHTML = originalText;
                submitBtn.disabled = false;
                lucide.createIcons();
            }
        });
    }

    // Run session check on load
    initializeSession();


    // --- Navigation Views ---
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view-section');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const viewId = item.getAttribute('data-view');

            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');

            views.forEach(view => view.classList.remove('active'));
            document.getElementById(`view-${viewId}`).classList.add('active');
            
            if (viewId === 'audit') {
                loadAuditLogs();
            }

            lucide.createIcons();
        });
    });

    // --- Health Check Polling ---
    async function checkHealth() {
        if (!jwtToken) return;
        const dot = document.querySelector('.pulse-dot');
        const text = document.getElementById('health-text');
        const gpuText = document.getElementById('metric-gpu');

        try {
            const res = await authFetch('/api/v1/health');
            const data = await res.json();

            if (data.status === 'healthy') {
                dot.classList.remove('error');
                text.textContent = 'SYSTEM SECURE';
                if (data.gpu) {
                    gpuText.textContent = `${data.gpu} (${data.gpu_memory_gb}GB)`;
                } else {
                    gpuText.textContent = 'CPU Target Processor';
                }
            } else {
                throw new Error();
            }
        } catch (e) {
            dot.classList.add('error');
            text.textContent = 'TELEMETRY OFFLINE';
            if (gpuText) gpuText.textContent = 'OFFLINE';
            const mStatus = document.getElementById('metric-status');
            if (mStatus) {
                mStatus.textContent = 'OFFLINE / DISCONNECTED';
                mStatus.style.color = 'var(--danger)';
                mStatus.classList.remove('text-gradient');
            }
        }
    }

    // Set check interval
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
                if (label) label.textContent = 'sample1.png (target loaded)';
            }
        } catch (err) {
            console.error('Failed to load sample target:', err);
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
                if (label) label.textContent = input.files[0].name + " (loaded)";
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
                alert('Please select a target material first.');
                return;
            }

            const submitBtn = detForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i data-lucide="loader" class="spin"></i> DEPLOYING TARGET SCAN...';
            lucide.createIcons();
            submitBtn.disabled = true;

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            const conf = confSlider ? confSlider.value : '0.25';

            try {
                const res = await authFetch(`/api/v1/detection/predict?conf=${conf}&iou=0.45`, {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                if (!res.ok) throw new Error(data.detail || 'API Scan Error');

                drawDetectionResults(fileInput.files[0], data);

            } catch (error) {
                alert('Strategic Scan Error: ' + error.message);
            } finally {
                submitBtn.innerHTML = originalText;
                submitBtn.disabled = false;
                lucide.createIcons();
            }
        });
    }

    function drawDetectionResults(file, data) {
        const placeholder = document.querySelector('#view-detection .placeholder-text');
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
                ctx.font = 'bold 12px Share Tech Mono';

                data.detections.forEach(det => {
                    const [x1, y1, x2, y2] = det.bbox;
                    const cx1 = x1 * scaleX;
                    const cy1 = y1 * scaleY;
                    const w = (x2 - x1) * scaleX;
                    const h = (y2 - y1) * scaleY;

                    // Neon target border
                    ctx.strokeStyle = '#FF3131';
                    ctx.fillStyle = 'rgba(255, 49, 49, 0.15)';
                    ctx.fillRect(cx1, cy1, w, h);
                    ctx.strokeRect(cx1, cy1, w, h);

                    // Target indicator corners
                    ctx.fillStyle = '#FF3131';
                    ctx.fillRect(cx1 - 2, cy1 - 2, 8, 2);
                    ctx.fillRect(cx1 - 2, cy1 - 2, 2, 8);

                    const label = `${det.class_name.toUpperCase()} ${(det.confidence * 100).toFixed(0)}%`;
                    const textWidth = ctx.measureText(label).width;
                    ctx.fillRect(cx1, cy1 - 18, textWidth + 10, 18);
                    ctx.fillStyle = '#000000';
                    ctx.fillText(label, cx1 + 5, cy1 - 5);
                });

                stats.innerHTML = `
                    <strong>SCAN ANALYSIS:</strong> Detected ${data.num_detections} anomalies.<br>
                    <span class="term-green">AESA Processor Latency: ${data.latency_ms.toFixed(1)} ms</span>
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
        const icon = sender === 'system' ? 'shield-check' : 'user';
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
                alert('Please upload a target image to verify.');
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
            typingDiv.innerHTML = '<i data-lucide="loader" class="spin"></i><div class="msg-bubble">QUERYING DRDO COPILOT ENGINE...</div>';
            chatContainer.appendChild(typingDiv);
            lucide.createIcons();
            chatContainer.scrollTop = chatContainer.scrollHeight;

            try {
                const url = new URL(window.location.origin + '/api/v1/vlm/query');
                url.searchParams.append('query', queryText);

                const res = await authFetch(url, {
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
                appendChatMsg('system', 'TACTICAL COPILOT FAULT: ' + error.message);
            }
        });
    }

    // --- VLM Report Button ---
    const vlmReportBtn = document.getElementById('vlm-report-btn');
    if (vlmReportBtn) {
        vlmReportBtn.addEventListener('click', async () => {
            const fileInput = document.getElementById('vlm-file');
            if (!fileInput.files.length) {
                alert('Load a target material before compiling the report.');
                return;
            }

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);

            appendChatMsg('user', 'Generate tactical report analysis.');

            const typingId = 'typing-' + Date.now();
            const typingDiv = document.createElement('div');
            typingDiv.className = 'chat-msg system-msg';
            typingDiv.id = typingId;
            typingDiv.innerHTML = '<i data-lucide="loader" class="spin"></i><div class="msg-bubble">COMPILING DEFENSE REPORT...</div>';
            chatContainer.appendChild(typingDiv);
            lucide.createIcons();

            try {
                const res = await authFetch('/api/v1/vlm/report', {
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
                        <br><strong>TACTICAL REPORT ID:</strong> ${data.report_id}
                        <br><strong>DEFENSE SEVERITY:</strong> <span style="text-transform:capitalize">${data.assessment.severity || 'N/A'}</span>
                        <br><br><strong>ANALYSIS REASONING:</strong><br>
                        ${data.assessment.raw_assessment}
                    </div>
                `;

                appendChatMsg('system', reportHtml, true);

            } catch (error) {
                const el = document.getElementById(typingId);
                if (el) el.remove();
                appendChatMsg('system', 'REPORT FAULT: ' + error.message);
            }
        });
    }


    // ============================================
    // DRONE AUTONOMY MODULE
    // ============================================
    const droneConsole = document.getElementById('drone-console');
    const droneForm = document.getElementById('drone-form');

    function logDrone(msg, type) {
        if (!droneConsole) return;
        const d = document.createElement('div');
        if (type === 'err') {
            d.className = 'term-red';
        } else if (type === 'warn') {
            d.className = 'term-amber';
        } else {
            d.className = 'term-green';
        }
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
            logDrone('COMMAND AUTHORIZATION INITIATED: ' + cmd, 'warn');

            try {
                const res = await authFetch('/api/v1/drone/command/natural', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: cmd })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'API Flight Error');
                
                logDrone('ACK: ' + data.message, 'green');
                logDrone('TACTICAL ACTION: ' + data.parsed_action.toUpperCase(), 'green');
                
                if (data.target) logDrone('TARGET GRIP: ' + data.target, 'green');
                if (data.confidence) logDrone('AESA CONFIDENCE: ' + (data.confidence * 100).toFixed(0) + '%', 'green');
                
                // Show target coordinates if present
                if (data.parameters && data.parameters.distance_m) {
                    logDrone('ESTIMATED DISTANCE: ' + data.parameters.distance_m + 'm', 'green');
                }
            } catch (err) {
                logDrone('CRITICAL FLIGHT ERROR: ' + err.message, 'err');
            }
        });
    }

    const btnHome = document.getElementById('btn-drone-home');
    if (btnHome) {
        btnHome.addEventListener('click', async () => {
            logDrone('INITIATING EMERGENCY RETURN TO LAUNCH (RTL)...', 'warn');
            try {
                const res = await authFetch('/api/v1/drone/command/return-home', { method: 'POST' });
                const data = await res.json();
                logDrone('RTL ACK: ' + data.message, 'green');
            } catch (err) {
                logDrone('RTL FLIGHT FAULT: ' + err.message, 'err');
            }
        });
    }

    const btnStop = document.getElementById('btn-drone-stop');
    if (btnStop) {
        btnStop.addEventListener('click', async () => {
            logDrone('CRITICAL ENGINE INTERRUPT ENFORCED — BRAKE IN PLACE...', 'err');
            try {
                const res = await authFetch('/api/v1/drone/command/emergency-stop', { method: 'POST' });
                const data = await res.json();
                logDrone('STOP ACK: ' + data.message, 'err');
            } catch (err) {
                logDrone('INTERRUPT FAULT: ' + err.message, 'err');
            }
        });
    }

    // --- Dynamic Telemetry & Radar Coordinates Tracking ---
    const radarDrone = document.getElementById('radar-drone');
    const radarTarget = document.getElementById('radar-target');

    setInterval(async () => {
        const droneView = document.getElementById('view-drone');
        if (!droneView || !droneView.classList.contains('active')) return;
        
        try {
            const res = await authFetch('/api/v1/drone/status');
            if (res.ok) {
                const data = await res.json();
                document.getElementById('drone-mode').textContent = data.mode;
                document.getElementById('drone-alt').textContent = data.altitude_m.toFixed(1);
                document.getElementById('drone-spd').textContent = data.speed_ms.toFixed(1);
                document.getElementById('drone-bat').textContent = data.battery_pct.toFixed(1);

                // Update drone rotation heading indicator
                if (data.heading_deg !== undefined) {
                    radarDrone.querySelector('.drone-arrow').style.transform = `rotate(${data.heading_deg}deg)`;
                }

                // Map NED local coordinates (x, y) to the 16:9 radar dashboard screen
                // Simple scaling: (x, y) are local coordinate offsets. Let's map center to 50%, 50%.
                if (data.position && data.position.x !== undefined && data.position.y !== undefined) {
                    // Let's assume geofence is 100 meters, so max offset is 100
                    const clampOffset = (val) => Math.max(-100, Math.min(100, val));
                    const scaledX = 50 + (clampOffset(data.position.y) / 2); // screen X maps to local Y (east-west)
                    const scaledY = 50 - (clampOffset(data.position.x) / 2); // screen Y maps to local -X (north-south)
                    
                    radarDrone.style.left = `${scaledX}%`;
                    radarDrone.style.top = `${scaledY}%`;
                }
            }
        } catch (e) { /* silently skip */ }
    }, 1000);


    // ============================================
    // SECURITY AUDIT SYSTEM (Commander Only)
    // ============================================
    window.loadAuditLogs = async function() {
        if (userRole !== 'Commander') return;
        const tbody = document.getElementById('audit-tbody');
        if (!tbody) return;

        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center;"><i data-lucide="loader" class="spin"></i> PULLING AUDIT LOG REGISTRY...</td></tr>';
        lucide.createIcons();

        try {
            const res = await authFetch('/api/v1/auth/audit');
            const data = await res.json();
            
            if (!res.ok) throw new Error(data.detail || 'Failed to download audit data');

            if (data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" style="text-align: center;">Audit database registry empty.</td></tr>';
                return;
            }

            tbody.innerHTML = '';
            data.forEach(log => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${log.timestamp}</td>
                    <td><strong>${log.username}</strong></td>
                    <td><span style="font-family:monospace; color:var(--accent-primary);">${log.action}</span></td>
                    <td>${log.details}</td>
                    <td><span class="audit-severity-${log.severity}">${log.severity}</span></td>
                `;
                tbody.appendChild(tr);
            });
        } catch (err) {
            tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color:var(--danger)">SECURITY FAULT: ${err.message}</td></tr>`;
        }
    };

});