// TaikoNation Studio JavaScript Application

class StateManager {
    constructor() {
        this.state = {
            training: { active: false, progress: 0, metrics: {} },
            generation: { active: false, progress: 0 },
            charts: [],
            models: {},
            logs: [],
            system: { status: 'initializing', message: 'Connecting...' }
        };
        this.listeners = {};
    }

    setState(key, value) {
        this.state[key] = { ...this.state[key], ...value };
        this.notifyListeners(key);
    }

    getState(key) {
        return this.state[key];
    }

    subscribe(key, callback) {
        if (!this.listeners[key]) {
            this.listeners[key] = [];
        }
        this.listeners[key].push(callback);
    }

    notifyListeners(key) {
        if (this.listeners[key]) {
            this.listeners[key].forEach(callback => callback(this.state[key]));
        }
    }
}

class TaikoNationApp {
    constructor() {
        this.currentPage = 'dashboard';
        this.apiBase = '/api';
        this.wsConnection = null;
        this.stateManager = new StateManager();
        this.uploadedAudioFilename = null;
        
        this.init();
    }

    async init() {
        console.log('Initializing TaikoNation Studio...');
        
        this.stateManager.subscribe('system', this.updateSystemStatus.bind(this));
        this.stateManager.subscribe('training', this.renderTrainingProgress.bind(this));
        this.stateManager.subscribe('generation', this.renderGenerationProgress.bind(this));

        this.setupEventListeners();
        await this.connectToBackend();
        await this.loadDashboardData();
        this.setupWebSocket();
        
        console.log('TaikoNation Studio initialized successfully');
    }

    setupEventListeners() {
        // File upload handlers
        const audioFileInput = document.getElementById('audioFile');
        const uploadZone = document.querySelector('.upload-zone');
        
        if (audioFileInput && uploadZone) {
            uploadZone.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
            uploadZone.addEventListener('drop', this.handleFileDrop.bind(this));
            audioFileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }

        // Form submissions
        const forms = ['trainingForm', 'generationForm', 'evaluationForm', 'configForm'];
        forms.forEach(formId => {
            const form = document.getElementById(formId);
            if (form) {
                form.addEventListener('submit', (e) => {
                    e.preventDefault();
                    this.handleFormSubmit(formId);
                });
            }
        });

        // Real-time slider updates for evaluation
        const sliders = document.querySelectorAll('.rating-slider');
        sliders.forEach(slider => {
            slider.addEventListener('input', this.updateSliderValue.bind(this));
        });

        // Nav buttons (semantic, accessible)
        const navButtons = document.querySelectorAll('.nav-item[data-page]');
        navButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const page = btn.getAttribute('data-page');
                showPage(page);
            });
            btn.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    const page = btn.getAttribute('data-page');
                    showPage(page);
                }
            });
        });

        // Tab buttons
        const tabButtons = document.querySelectorAll('.tab-button');
        tabButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const onclick = btn.getAttribute('onclick') || '';
                const m = onclick.match(/showTab\('(.*)'\)/);
                if (m && m[1]) showTab(m[1]);
            });
        });
    }

    async connectToBackend() {
        try {
            const response = await fetch(`${this.apiBase}/status`);
            if (response.ok) {
                const status = await response.json();
                this.stateManager.setState('system', { status: 'connected', message: 'System Ready' });
                console.log('Backend connection established:', status);
            } else {
                throw new Error('Backend not responding');
            }
        } catch (error) {
            console.error('Failed to connect to backend:', error);
            this.stateManager.setState('system', { status: 'error', message: 'Backend Unavailable' });
        }
    }

    setupWebSocket() {
        try {
            // Use Socket.IO client (index.html includes socket.io script)
            if (typeof io === 'undefined') {
                console.warn('Socket.IO client not found; real-time updates will be unavailable');
                return;
            }

            this.socket = io();

            this.socket.on('connect', () => {
                console.log('Socket.IO connected');
            });

            this.socket.on('training_progress', (data) => {
                this.stateManager.setState('training', { active: true, progress: data.progress, metrics: data.metrics });
                if (data.progress >= 100) {
                    this.stateManager.setState('training', { active: false });
                    this.addSystemLog('success', 'Training completed successfully');
                }
            });

            this.socket.on('generation_progress', (data) => {
                this.stateManager.setState('generation', { active: true, progress: data.progress });
                if (data.progress >= 100) {
                    this.stateManager.setState('generation', { active: false });
                    this.addSystemLog('success', 'Chart generated successfully');
                }
            });

            this.socket.on('system_log', (log) => {
                // server sends a log object {timestamp, level, message}
                this.addSystemLog(log.level || 'info', log.message, log.timestamp);
            });

            this.socket.on('chart_generated', (payload) => {
                if (payload && payload.chart) this.addGeneratedChart(payload.chart);
            });

            this.socket.on('disconnect', (reason) => {
                console.log('Socket.IO disconnected:', reason);
            });
        } catch (error) {
            console.error('Failed to establish Socket.IO connection:', error);
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'training_progress':
                this.updateTrainingProgress(data.progress, data.metrics);
                break;
            case 'generation_progress':
                this.updateGenerationProgress(data.progress);
                break;
            case 'system_log':
                this.addSystemLog(data.level, data.message);
                break;
            case 'model_update':
                this.refreshModelGrid();
                break;
            case 'chart_generated':
                this.addGeneratedChart(data.chart);
                break;
            default:
                console.log('Unknown WebSocket message:', data);
        }
    }

    updateSystemStatus({ status, message }) {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        
        if (statusDot && statusText) {
            statusDot.className = `status-dot ${status}`;
            statusText.textContent = message;
        }
    }

    async loadDashboardData() {
        try {
            const response = await fetch(`${this.apiBase}/dashboard`);
            if (response.ok) {
                const data = await response.json();
                this.updateDashboardMetrics(data.metrics);
                this.updateSystemLogs(data.recent_logs);
            }
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
        }
    }

    updateDashboardMetrics(metrics) {
        const elements = {
            'activeModels': metrics.active_models || '-',
            'generatedCharts': metrics.generated_charts || '-',
            'bestAccuracy': metrics.best_accuracy ? `${metrics.best_accuracy}%` : '-',
            'avgRating': metrics.avg_rating || '-'
        };

        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
    }

    updateSystemLogs(logs) {
        const logsContainer = document.getElementById('systemLogs');
        if (logsContainer) {
            logsContainer.innerHTML = '';
            logs.forEach(log => {
                this.addSystemLog(log.level, log.message, log.timestamp);
            });
        }
    }

    addSystemLog(level, message, timestamp = null) {
        const logsContainer = document.getElementById('systemLogs');
        if (!logsContainer) return;

        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        
        const time = timestamp || new Date().toLocaleTimeString();
        logEntry.innerHTML = `
            <span class="log-timestamp">[${time}]</span>
            <span class="log-level-${level}">${this.getLogIcon(level)} ${message}</span>
        `;
        
        logsContainer.insertBefore(logEntry, logsContainer.firstChild);
        
        // Keep only the last 50 log entries
        while (logsContainer.children.length > 50) {
            logsContainer.removeChild(logsContainer.lastChild);
        }
    }

    getLogIcon(level) {
        const icons = {
            'info': 'ℹ️',
            'success': '✓',
            'warning': '⚠️',
            'error': '❌'
        };
        return icons[level] || 'ℹ️';
    }

    // File handling methods
    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
    }

    handleFileDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        this.processAudioFiles(files);
    }

    handleFileSelect(e) {
        const files = e.target.files;
        this.processAudioFiles(files);
    }

    async processAudioFiles(files) {
        for (let file of files) {
            if (this.isValidAudioFile(file)) {
                await this.uploadAudioFile(file);
            } else {
                this.addSystemLog('warning', `Skipped invalid file: ${file.name}`);
            }
        }
    }

    isValidAudioFile(file) {
         const validTypes = ['audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/ogg', 'audio/flac', 'audio/x-wav'];
         const validExtensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac'];
         const fileType = (file.type || '').toLowerCase();

         return validTypes.includes(fileType) || 
             validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    }

    async uploadAudioFile(file) {
        const formData = new FormData();
        formData.append('audio', file);
        
        try {
            this.addSystemLog('info', `Uploading ${file.name}...`);
            
            const response = await fetch(`${this.apiBase}/upload-audio`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                this.addSystemLog('success', `Audio processed: ${file.name}`);
                this.uploadedAudioFilename = result.filename;
                
                // Auto-fill form fields if on generation page
                if (this.currentPage === 'generation') {
                    this.updateGenerationForm(result);
                }
            } else {
                throw new Error(`Upload failed: ${response.statusText}`);
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.addSystemLog('error', `Failed to upload ${file.name}: ${error.message}`);
        }
    }

    updateGenerationForm(audioData) {
        const form = document.getElementById('generationForm');
        if (!form) return;

        // Update form fields with detected audio information
        const titleInput = form.querySelector('[name="title"]');
        const bpmInput = form.querySelector('[name="bpm"]');
        
        if (titleInput && audioData.title) {
            titleInput.value = audioData.title;
        }
        
        if (bpmInput && audioData.detected_bpm) {
            bpmInput.value = audioData.detected_bpm;
        }
    }

    // Training methods
    async startTraining() {
        const form = document.getElementById('trainingForm');
        const formData = new FormData(form);
        
        try {
            this.stateManager.setState('training', { active: true, progress: 0, metrics: {} });
            
            const response = await fetch(`${this.apiBase}/start-training`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                this.addSystemLog('success', 'Training started successfully');
            } else {
                throw new Error(`Training failed: ${response.statusText}`);
            }
        } catch (error) {
            console.error('Training error:', error);
            this.addSystemLog('error', `Training failed: ${error.message}`);
            this.stateManager.setState('training', { active: false });
        }
    }

    showTrainingProgress() {
        const progressSection = document.getElementById('trainingProgress');
        if (progressSection) {
            progressSection.style.display = 'block';
        }
    }

    hideTrainingProgress() {
        const progressSection = document.getElementById('trainingProgress');
        if (progressSection) {
            progressSection.style.display = 'none';
        }
    }

    renderTrainingProgress({ active, progress, metrics }) {
        if (!active) {
            this.hideTrainingProgress();
            return;
        }
        this.showTrainingProgress();

        const progressBar = document.getElementById('trainingProgressBar');
        const progressText = document.getElementById('trainingProgressText');
        
        if (progressBar) progressBar.style.width = `${progress}%`;
        if (progressText) progressText.textContent = `${progress}% Complete`;
        
        if (metrics) {
            this.updateTrainingMetrics(metrics);
        }
    }

    updateTrainingMetrics(metrics) {
        const container = document.getElementById('trainingMetrics');
        if (!container) return;

        container.innerHTML = '';
        
        Object.entries(metrics).forEach(([key, value]) => {
            const metricDiv = document.createElement('div');
            metricDiv.className = 'training-metric';
            metricDiv.innerHTML = `
                <div class="training-metric-value">${value}</div>
                <div class="training-metric-label">${this.formatMetricLabel(key)}</div>
            `;
            container.appendChild(metricDiv);
        });
    }

    formatMetricLabel(key) {
        return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    async stopTraining() {
        try {
            const response = await fetch(`${this.apiBase}/stop-training`, {
                method: 'POST'
            });
            
            if (response.ok) {
                this.stateManager.setState('training', { active: false, progress: 0, metrics: {} });
                this.addSystemLog('info', 'Training stopped by user');
            }
        } catch (error) {
            console.error('Failed to stop training:', error);
        }
    }

    // Generation methods
    async generateChart() {
        const form = document.getElementById('generationForm');
        if (!this.uploadedAudioFilename) {
            this.addSystemLog('error', 'Please upload an audio file first.');
            return;
        }

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        data.audio_filename = this.uploadedAudioFilename;

        try {
            this.stateManager.setState('generation', { active: true, progress: 0 });
            
            const response = await fetch(`${this.apiBase}/generate-chart`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });
            
            if (response.ok) {
                this.addSystemLog('success', 'Chart generation started');
            } else {
                throw new Error(`Generation failed: ${response.statusText}`);
            }
        } catch (error) {
            console.error('Generation error:', error);
            this.addSystemLog('error', `Chart generation failed: ${error.message}`);
            this.stateManager.setState('generation', { active: false });
        }
    }

    showGenerationProgress() {
        const progressBar = document.getElementById('generationProgress');
        if (progressBar) {
            progressBar.style.display = 'block';
        }
    }

    hideGenerationProgress() {
        const progressBar = document.getElementById('generationProgress');
        if (progressBar) {
            progressBar.style.display = 'none';
        }
    }

    renderGenerationProgress({ active, progress }) {
        if (!active) {
            this.hideGenerationProgress();
            return;
        }
        this.showGenerationProgress();

        const progressBar = document.getElementById('generationProgressBar');
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
    }

    // Evaluation methods
    async startEvaluation() {
        try {
            const response = await fetch(`${this.apiBase}/get-chart-for-evaluation`);
            
            if (response.ok) {
                const chart = await response.json();
                this.showEvaluationForm(chart);
            } else {
                throw new Error('No charts available for evaluation');
            }
        } catch (error) {
            console.error('Evaluation error:', error);
            this.addSystemLog('error', 'Failed to start evaluation session');
        }
    }

    showEvaluationForm(chart) {
        const form = document.getElementById('evaluationForm');
        const titleElement = document.getElementById('evaluationChartTitle');
        
        if (form) {
            form.style.display = 'block';
            form.dataset.chartId = chart.id;
        }
        
        if (titleElement) {
            titleElement.textContent = `Rate: ${chart.title} - ${chart.difficulty}`;
        }
    }

    async submitEvaluation() {
        const form = document.getElementById('evaluationForm');
        const formData = new FormData(form);
        const chartId = form.dataset.chartId;
        
        formData.append('chart_id', chartId);
        
        try {
            const response = await fetch(`${this.apiBase}/submit-evaluation`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                this.addSystemLog('success', 'Evaluation submitted successfully');
                this.startEvaluation(); // Load next chart
            } else {
                throw new Error('Failed to submit evaluation');
            }
        } catch (error) {
            console.error('Evaluation submission error:', error);
            this.addSystemLog('error', 'Failed to submit evaluation');
        }
    }

    skipEvaluation() {
        this.startEvaluation(); // Load next chart
    }

    updateSliderValue(e) {
        const slider = e.target;
        const value = slider.value;
        
        // Update visual feedback for slider
        const percent = ((value - slider.min) / (slider.max - slider.min)) * 100;
        slider.style.background = `linear-gradient(to right, var(--accent-primary) 0%, var(--accent-primary) ${percent}%, var(--bg-tertiary) ${percent}%, var(--bg-tertiary) 100%)`;
    }

    async startComparativeEvaluation() {
        try {
            const response = await fetch(`${this.apiBase}/charts`);
            if (response.ok) {
                const data = await response.json();
                if (data.charts.length >= 2) {
                    const chartA = data.charts[Math.floor(Math.random() * data.charts.length)];
                    let chartB = data.charts[Math.floor(Math.random() * data.charts.length)];
                    while (chartB.id === chartA.id) {
                        chartB = data.charts[Math.floor(Math.random() * data.charts.length)];
                    }
                    this.showComparativeEvaluation(chartA, chartB);
                } else {
                    this.addSystemLog('warning', 'Not enough charts for comparative evaluation');
                }
            }
        } catch (error) {
            console.error('Failed to load charts for comparative evaluation:', error);
        }
    }

    showComparativeEvaluation(chartA, chartB) {
        document.getElementById('comparativeEvaluation').style.display = 'block';
        document.getElementById('chartA').innerText = chartA.title;
        document.getElementById('chartB').innerText = chartB.title;
        document.getElementById('chartA').dataset.id = chartA.id;
        document.getElementById('chartB').dataset.id = chartB.id;
    }

    async submitComparativeEvaluation(preference) {
        const chartA_id = document.getElementById('chartA').dataset.id;
        const chartB_id = document.getElementById('chartB').dataset.id;
        const formData = new FormData();
        formData.append('chartA_id', chartA_id);
        formData.append('chartB_id', chartB_id);
        formData.append('preference', preference);

        try {
            const response = await fetch(`${this.apiBase}/submit-comparative-evaluation`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                this.addSystemLog('success', 'Comparative evaluation submitted successfully');
                this.startComparativeEvaluation(); // Load next pair
            } else {
                throw new Error('Failed to submit comparative evaluation');
            }
        } catch (error) {
            console.error('Comparative evaluation submission error:', error);
            this.addSystemLog('error', 'Failed to submit comparative evaluation');
        }
    }

    // Configuration methods
    async loadConfig() {
        try {
            const response = await fetch(`${this.apiBase}/config`);
            
            if (response.ok) {
                const config = await response.json();
                this.populateConfigForm(config);
                this.addSystemLog('success', 'Configuration loaded');
            }
        } catch (error) {
            console.error('Config load error:', error);
            this.addSystemLog('error', 'Failed to load configuration');
        }
    }

    populateConfigForm(config) {
        const form = document.getElementById('trainingForm');
        if (!form) return;

        Object.entries(config).forEach(([key, value]) => {
            const input = form.querySelector(`[name="${key}"]`);
            if (input) {
                input.value = value;
            }
        });
    }

    async saveConfig() {
        const form = document.getElementById('trainingForm');
        const formData = new FormData(form);
        
        try {
            const response = await fetch(`${this.apiBase}/save-config`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                this.addSystemLog('success', 'Configuration saved');
            } else {
                throw new Error('Failed to save configuration');
            }
        } catch (error) {
            console.error('Config save error:', error);
            this.addSystemLog('error', 'Failed to save configuration');
        }
    }

    // Utility methods
    formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        
        if (hours > 0) {
            return `${hours}h ${minutes}m ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }

    async refreshStatus() {
        await this.loadDashboardData();
        this.addSystemLog('info', 'Status refreshed');
    }

    async loadResearchData() {
        try {
            const response = await fetch(`${this.apiBase}/research/experiments`);
            if (response.ok) {
                const data = await response.json();
                this.updateResearchDashboard(data);
            }
        } catch (error) {
            console.error('Failed to load research data:', error);
        }
    }

    updateResearchDashboard(data) {
        const { experiments, model_comparisons, pattern_evolution, human_feedback_stats } = data;
        document.getElementById('experimentHistory').innerText = JSON.stringify(experiments, null, 2);
        document.getElementById('modelComparisons').innerText = JSON.stringify(model_comparisons, null, 2);
        document.getElementById('patternEvolution').innerText = JSON.stringify(pattern_evolution, null, 2);
        document.getElementById('humanFeedbackStats').innerText = JSON.stringify(human_feedback_stats, null, 2);
    }
}

// Helper stubs for events/UI pieces not yet implemented
TaikoNationApp.prototype.addGeneratedChart = function(chart) {
    // Add chart to UI library grid if present
    try {
        const grid = document.getElementById('chartsGrid');
        if (!grid) return;

        const card = document.createElement('div');
        card.className = 'chart-card';
        card.innerHTML = `
            <div class="chart-preview"><div class="chart-waveform">${chart.title}</div></div>
            <div class="chart-info">
                <div class="chart-title">${chart.title}</div>
                <div class="chart-meta">${chart.artist} • ${chart.bpm} BPM</div>
                <div class="chart-actions">
                    <a href="/api/download-chart?id=${chart.id}" class="btn btn-secondary" download>Download .osu</a>
                    <a href="/editor?id=${chart.id}" class="btn btn-secondary">Edit</a>
                </div>
            </div>
        `;
        grid.prepend(card);
    } catch (e) {
        console.error('Failed to add generated chart to UI', e);
    }
};

TaikoNationApp.prototype.refreshModelGrid = function() {
    // Basic refresh: fetch /api/models and update modelGrid
    fetch(`${this.apiBase}/models`).then(r => r.json()).then(data => {
        const grid = document.getElementById('modelGrid');
        if (!grid) return;
        grid.innerHTML = '';
        const models = data.models || {};
        Object.values(models).forEach(m => {
            const card = document.createElement('div');
            card.className = 'model-card';
            card.innerHTML = `
                <div class="model-header"><div class="model-name">${m.name}</div><div class="model-type">${m.type}</div></div>
                <div class="model-stats"><div class="stat-item"><div class="stat-value">${m.accuracy || '-'}</div><div class="stat-label">Accuracy</div></div></div>
            `;
            grid.appendChild(card);
        });
    }).catch(err => console.error('Failed to refresh model grid', err));
};

// Page and tab navigation functions (global scope for onclick handlers)
function showPage(pageId) {
    // Hide all pages
    const pages = document.querySelectorAll('.page-content');
    pages.forEach(page => page.classList.add('hidden'));
    
    // Show selected page
    const targetPage = document.getElementById(pageId + '-page');
    if (targetPage) {
        targetPage.classList.remove('hidden');
    }
    
    // Update navigation
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        const onclickAttr = item.getAttribute('onclick') || '';
        if (onclickAttr.includes(`'${pageId}'`) || onclickAttr.includes(`"${pageId}"`)) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
    
    // Update page title
    const titles = {
        'dashboard': 'Dashboard',
        'training': 'Model Training',
        'generation': 'Chart Generation',
        'library': 'Chart Library',
        'evaluation': 'Human Evaluation',
        'config': 'Configuration',
        'research': 'Research'
    };
    
    const pageTitle = document.getElementById('pageTitle');
    if (pageTitle) {
        pageTitle.textContent = titles[pageId] || 'TaikoNation';
    }
    
    // Update current page in app instance
    if (window.taikoApp) {
        window.taikoApp.currentPage = pageId;
        if (pageId === 'research') {
            window.taikoApp.loadResearchData();
        }
    }
}

function showTab(tabId) {
    // Hide all tab contents
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.classList.remove('active'));
    
    // Show selected tab
    const targetTab = document.getElementById(tabId);
    if (targetTab) {
        targetTab.classList.add('active');
    }
    
    // Update tab buttons
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        const onclickAttr = button.getAttribute('onclick') || '';
        if (onclickAttr.includes(`'${tabId}'`) || onclickAttr.includes(`"${tabId}"`)) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
}

// Taiko drum interaction
function playSound(type) {
    const drum = event.target;
    drum.classList.add('pulse');
    
    // Remove animation class after animation completes
    setTimeout(() => {
        drum.classList.remove('pulse');
    }, 300);
    
    // In a real implementation, you would play actual taiko sounds here
    console.log(`Playing ${type} sound`);
    
    if (window.taikoApp) {
        window.taikoApp.addSystemLog('info', `Played ${type} drum sound`);
    }
}

// Global function wrappers for onclick handlers
function startTraining() {
    if (window.taikoApp) {
        window.taikoApp.startTraining();
    }
}

function stopTraining() {
    if (window.taikoApp) {
        window.taikoApp.stopTraining();
    }
}

function generateChart() {
    if (window.taikoApp) {
        window.taikoApp.generateChart();
    }
}

function startEvaluation() {
    if (window.taikoApp) {
        window.taikoApp.startEvaluation();
    }
}

function submitEvaluation() {
    if (window.taikoApp) {
        window.taikoApp.submitEvaluation();
    }
}

function skipEvaluation() {
    if (window.taikoApp) {
        window.taikoApp.skipEvaluation();
    }
}

function startComparativeEvaluation() {
    if (window.taikoApp) {
        window.taikoApp.startComparativeEvaluation();
    }
}

function submitComparativeEvaluation(preference) {
    if (window.taikoApp) {
        window.taikoApp.submitComparativeEvaluation(preference);
    }
}

function loadConfig() {
    if (window.taikoApp) {
        window.taikoApp.loadConfig();
    }
}

function saveConfig() {
    if (window.taikoApp) {
        window.taikoApp.saveConfig();
    }
}

function refreshStatus() {
    if (window.taikoApp) {
        window.taikoApp.refreshStatus();
    }
}

function exportResearchDataset() {
    if (window.taikoApp) {
        window.open('/api/research/export-dataset');
    }
}

// Backwards-compatible wrapper used by index.html "Export All" button
function exportAllCharts() {
    // Reuse the research dataset export endpoint for a full export
    if (window.taikoApp) {
        window.open('/api/research/export-dataset');
    } else {
        window.open('/api/research/export-dataset');
    }
}

// (Initialization moved to the end of this file to ensure all classes are defined)
class TaikoNationUI {
  constructor() {
    this.socket = null;
    this.currentUpload = null;
    this.activeGenerations = new Map();
    this.charts = [];
    this.filters = {
      search: '',
      difficulties: [],
      sort: 'recent'
    };

    this.init();
  }

  async init() {
    this.setupSocket();
    this.setupEventListeners();
    this.setupUploadFlow();
    this.loadInitialData();
    this.startHeartbeat();
  }

  setupSocket() {
    this.socket = io({
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5
    });

    // Connection status
    this.socket.on('connect', () => {
      this.updateConnectionStatus('connected');
      this.showToast('Connected to TaikoNation Studio', 'success');
    });

    this.socket.on('disconnect', () => {
      this.updateConnectionStatus('disconnected');
      this.showToast('Connection lost. Reconnecting...', 'warning');
    });

    // Real-time progress updates
    this.socket.on('generation_progress', (data) => {
      this.updateGenerationProgress(data.progress);
      if (data.progress === 100) {
        this.showToast('Chart generation completed!', 'success');
      }
    });

    this.socket.on('training_progress', (data) => {
      this.updateTrainingProgress(data.progress, data.metrics);
    });

    // System logs
    this.socket.on('system_log', (log) => {
      this.addActivityLog(log);
    });

    // Chart generation completion
    this.socket.on('chart_generated', (data) => {
      this.onChartGenerated(data.chart);
    });
  }

  setupUploadFlow() {
    const uploadZone = document.getElementById('upload-zone');
    const audioInput = document.getElementById('audio-input');

    // Drag & drop handling
    uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
      uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadZone.classList.remove('drag-over');
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        this.handleFileUpload(files[0]);
      }
    });

    // Click to upload
    uploadZone.addEventListener('click', () => {
      audioInput.click();
    });

    audioInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        this.handleFileUpload(e.target.files[0]);
      }
    });
  }

  async handleFileUpload(file) {
    // Validate file
    const allowedTypes = ['audio/mp3', 'audio/wav', 'audio/ogg', 'audio/flac', 'audio/m4a', 'audio/aac'];
    if (!allowedTypes.includes(file.type)) {
      this.showToast('Unsupported file type', 'error');
      return;
    }

    if (file.size > 100 * 1024 * 1024) { // 100MB limit
      this.showToast('File too large (max 100MB)', 'error');
      return;
    }

    // Show upload progress
    this.showUploadProgress(true);

    try {
      const formData = new FormData();
      formData.append('audio', file);

      // Add auth token if available
      const token = localStorage.getItem('taikonation_token');
      const headers = {};
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const response = await fetch('/api/upload-audio', {
        method: 'POST',
        headers,
        body: formData
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Upload failed');
      }

      const result = await response.json();
      this.currentUpload = result;

      // Update UI with file info
      this.populateFileInfo(result);
      this.nextUploadStep();

      this.showToast('Audio processed successfully!', 'success');

    } catch (error) {
      this.showToast(`Upload failed: ${error.message}`, 'error');
    } finally {
      this.showUploadProgress(false);
    }
  }

  populateFileInfo(data) {
    document.getElementById('detected-title').textContent = data.title || 'Unknown';
    document.getElementById('detected-duration').textContent =
      data.duration ? `${Math.round(data.duration)}s` : 'Unknown';
    document.getElementById('detected-bpm').textContent = data.detected_bpm || 'Unknown';
    document.getElementById('features-status').textContent =
      data.features_extracted ? '✅ Extracted' : '❌ Failed';

    // Pre-fill form
    document.getElementById('chart-title').value = data.title || '';
    document.getElementById('chart-bpm').value = data.detected_bpm || 120;

    document.getElementById('file-info').style.display = 'block';
  }

  async generateChart() {
    if (!this.currentUpload) {
      this.showToast('Please upload an audio file first', 'error');
      return;
    }

    const chartData = {
      title: document.getElementById('chart-title').value || 'Untitled',
      artist: document.getElementById('chart-artist').value || 'Unknown Artist',
      bpm: parseInt(document.getElementById('chart-bpm').value) || 120,
      difficulty: document.querySelector('.difficulty-btn.selected')?.dataset.difficulty || 'oni',
      genre: 'electronic', // Could be made configurable
      pattern_style: 'balanced',
      audio_filename: this.currentUpload.filename,
      npy_filename: this.currentUpload.npy_filename
    };

    try {
      const token = localStorage.getItem('taikonation_token');
      const headers = {
        'Content-Type': 'application/json'
      };
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const response = await fetch('/api/generate-chart', {
        method: 'POST',
        headers,
        body: JSON.stringify(chartData)
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Generation failed');
      }

      // Move to generation step
      this.nextUploadStep();
      this.showGenerationProgress(true);

    } catch (error) {
      this.showToast(`Generation failed: ${error.message}`, 'error');
    }
  }

  onChartGenerated(chart) {
    this.charts.unshift(chart);
    this.renderCharts();
    this.showGenerationProgress(false);

    // Show success with download option
    this.showToast(
      `Chart "${chart.title}" generated successfully!`,
      'success',
      {
        action: {
          text: 'Download',
          callback: () => this.downloadChart(chart.id)
        }
      }
    );

    // Reset upload flow
    this.resetUploadFlow();
  }

  async downloadChart(chartId) {
    try {
      const token = localStorage.getItem('taikonation_token');
      const url = new URL('/api/download-chart', window.location.origin);
      url.searchParams.set('id', chartId);
      if (token) {
        url.searchParams.set('api_token', token);
      }

      // Create temporary download link
      const link = document.createElement('a');
      link.href = url.toString();
      link.download = ''; // Let browser determine filename
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      this.showToast('Download started', 'info');

    } catch (error) {
      this.showToast(`Download failed: ${error.message}`, 'error');
    }
  }

  renderCharts() {
    const grid = document.getElementById('charts-grid');
    const empty = document.getElementById('charts-empty');
    const template = document.getElementById('chart-card-template');

    if (this.charts.length === 0) {
      grid.style.display = 'none';
      empty.style.display = 'block';
      return;
    }

    grid.style.display = 'grid';
    empty.style.display = 'none';

    // Filter and sort charts
    let filteredCharts = this.charts.filter(chart => {
      const matchesSearch = chart.title.toLowerCase().includes(this.filters.search.toLowerCase()) ||
                           chart.artist.toLowerCase().includes(this.filters.search.toLowerCase());
      const matchesDifficulty = this.filters.difficulties.length === 0 ||
                               this.filters.difficulties.includes(chart.difficulty);
      return matchesSearch && matchesDifficulty;
    });

    // Sort charts
    filteredCharts.sort((a, b) => {
      switch (this.filters.sort) {
        case 'recent':
          return new Date(b.created_at) - new Date(a.created_at);
        case 'rating':
          return (b.rating || 0) - (a.rating || 0);
        case 'title':
          return a.title.localeCompare(b.title);
        case 'difficulty':
          const diffOrder = ['kantan', 'futsuu', 'muzukashii', 'oni', 'ura'];
          return diffOrder.indexOf(a.difficulty) - diffOrder.indexOf(b.difficulty);
        default:
          return 0;
      }
    });

    // Clear grid
    grid.innerHTML = '';

    // Render charts
    filteredCharts.forEach(chart => {
      const card = template.content.cloneNode(true);

      // Populate card data
      card.querySelector('.chart-card').dataset.chartId = chart.id;
      card.querySelector('.chart-title').textContent = chart.title;
      card.querySelector('.chart-artist').textContent = chart.artist;
      card.querySelector('.chart-bpm').textContent = `${chart.bpm} BPM`;
      card.querySelector('.chart-rating').textContent = (chart.rating || 0).toFixed(1);
      card.querySelector('.chart-created').textContent = this.formatDate(chart.created_at);

      // Set difficulty styling
      const difficultyColor = card.querySelector('.difficulty-color');
      const difficultyText = card.querySelector('.difficulty-text');
      difficultyColor.className = `difficulty-color ${chart.difficulty}`;
      difficultyText.textContent = this.capitalizeDifficulty(chart.difficulty);

      // Generate waveform visualization (placeholder)
      this.generateWaveform(card.querySelector('.chart-waveform'), chart);

      // Add event listeners
      card.querySelector('.download-btn').addEventListener('click', () => {
        this.downloadChart(chart.id);
      });

      card.querySelector('.preview-btn').addEventListener('click', () => {
        this.previewChart(chart);
      });

      card.querySelector('.rate-btn').addEventListener('click', () => {
        this.showRatingModal(chart);
      });

      grid.appendChild(card);
    });
  }

    // Minimal event listeners to wire UI controls referenced in init()
    setupEventListeners() {
        // Generation button
        const genBtn = document.getElementById('generate-btn') || document.querySelector('.generate-btn');
        if (genBtn) genBtn.addEventListener('click', () => this.generateChart());

        // Search/filter inputs
        const search = document.getElementById('chart-search');
        if (search) {
            search.addEventListener('input', (e) => {
                this.filters.search = e.target.value || '';
                this.renderCharts();
            });
        }

        const sortSelect = document.getElementById('chart-sort');
        if (sortSelect) {
            sortSelect.addEventListener('change', (e) => {
                this.filters.sort = e.target.value;
                this.renderCharts();
            });
        }

        // Difficulty filters (checkboxes)
        const diffCheckboxes = document.querySelectorAll('.difficulty-filter');
        if (diffCheckboxes.length > 0) {
            diffCheckboxes.forEach(cb => cb.addEventListener('change', () => {
                this.filters.difficulties = Array.from(diffCheckboxes).filter(c => c.checked).map(c => c.value);
                this.renderCharts();
            }));
        }
    }

    // Load some initial data for the UI (non-blocking)
    async loadInitialData() {
        try {
            const resp = await fetch('/api/charts');
            if (resp.ok) {
                const data = await resp.json();
                this.charts = data.charts || [];
                this.renderCharts();
            }
        } catch (e) {
            // Not critical - just log locally
            console.warn('Failed to load initial charts:', e);
        }
    }

    // Heartbeat to keep socket alive / refresh small UI bits
    startHeartbeat() {
        if (!this.socket) return;
        // Send a lightweight ping every 30s if socket is connected
        this._heartbeatInterval = setInterval(() => {
            try {
                if (this.socket && this.socket.connected) {
                    this.socket.emit('heartbeat', { ts: Date.now() });
                }
            } catch (e) {
                // ignore
            }
        }, 30000);
    }

    // Connection / progress helpers used by socket callbacks
    updateConnectionStatus(status) {
        const dot = document.getElementById('statusDot') || document.getElementById('connection-dot');
        const text = document.getElementById('statusText') || document.getElementById('connection-status');
        if (dot) {
            dot.className = `status-dot ${status}`;
        }
        if (text) {
            text.textContent = status === 'connected' ? 'Connected' : (status === 'disconnected' ? 'Disconnected' : status);
        }
    }

    updateGenerationProgress(progress) {
        const bar = document.getElementById('generationProgressBar') || document.querySelector('.generation-progress .progress-bar');
        if (bar) {
            bar.style.width = `${progress}%`;
        }
    }

    updateTrainingProgress(progress, metrics = {}) {
        const bar = document.getElementById('trainingProgressBar') || document.querySelector('.training-progress .progress-bar');
        if (bar) bar.style.width = `${progress}%`;
        // Brief activity log
        if (progress % 10 === 0) {
            this.addActivityLog({ level: 'info', message: `Training ${progress}%`, timestamp: new Date().toLocaleTimeString() });
        }
    }

    // Simple upload-step helpers used by upload flow
    showUploadProgress(show) {
        const container = document.getElementById('uploadProgress') || document.querySelector('.upload-progress');
        if (!container) return;
        container.style.display = show ? 'block' : 'none';
        const bar = container.querySelector('.progress-bar') || container;
        if (!show && bar) bar.style.width = '0%';
    }

    nextUploadStep() {
        // Move to next step if steps container present
        const steps = document.querySelectorAll('.upload-step');
        if (!steps || steps.length === 0) return;
        for (let i = 0; i < steps.length; i++) {
            if (steps[i].classList.contains('active')) {
                steps[i].classList.remove('active');
                if (steps[i+1]) steps[i+1].classList.add('active');
                break;
            }
        }
    }

    resetUploadFlow() {
        const steps = document.querySelectorAll('.upload-step');
        if (!steps || steps.length === 0) return;
        steps.forEach((s, idx) => {
            s.classList.toggle('active', idx === 0);
        });
        this.currentUpload = null;
        // hide progress
        this.showUploadProgress(false);
    }

    // Toggle generation progress UI
    showGenerationProgress(show) {
        const container = document.getElementById('generationProgress') || document.querySelector('.generation-progress');
        if (!container) return;
        container.style.display = show ? 'block' : 'none';
        const bar = container.querySelector('.progress-bar') || container;
        if (!show && bar) bar.style.width = '0%';
    }

    // Remove a toast element
    removeToast(toast) {
        if (!toast) return;
        try {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(20px)';
            setTimeout(() => { if (toast.parentNode) toast.parentNode.removeChild(toast); }, 300);
        } catch (e) {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        }
    }

    // Lightweight waveform placeholder renderer
    generateWaveform(container, chart) {
        if (!container) return;
        container.textContent = chart.title ? `▂▁▃▂ ${chart.title}` : '▂▁▃▂ waveform';
    }

    // Preview chart placeholder
    previewChart(chart) {
        // For now open the download endpoint in a new tab as a quick preview
        if (!chart || !chart.id) return;
        const url = `/api/download-chart?id=${chart.id}`;
        window.open(url, '_blank');
    }

    // Simple rating modal stub (could be replaced by real modal)
    showRatingModal(chart) {
        const rating = prompt(`Rate chart "${chart.title}" (1-5):`, '4');
        if (rating) {
            fetch('/api/submit-rating', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ chart_id: chart.id, rating: parseInt(rating) })
            }).then(r => {
                if (r.ok) this.showToast('Rating submitted', 'success');
                else this.showToast('Failed to submit rating', 'error');
            }).catch(() => this.showToast('Failed to submit rating', 'error'));
        }
    }

  showToast(message, type = 'info', options = {}) {
        let container = document.getElementById('toast-container');

        // If there's no container in the DOM yet, create one so callers don't fail
        if (!container) {
            try {
                container = document.createElement('div');
                container.id = 'toast-container';
                container.className = 'toast-container';
                document.body.appendChild(container);
            } catch (e) {
                // If DOM isn't ready or append fails, fallback to console
                console.warn('Unable to create toast container:', e);
                console.log(`[TOAST ${type.toUpperCase()}] ${message}`);
                return;
            }
        }

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <div class="toast-icon">
                    ${this.getToastIcon(type)}
                </div>
                <div class="toast-message">${message}</div>
                ${options.action ? `
                    <button class="toast-action">${options.action.text}</button>
                ` : ''}
                <button class="toast-close">×</button>
            </div>
        `;

        // Event listeners (guard in case markup differs)
        const closeBtn = toast.querySelector('.toast-close');
        if (closeBtn) closeBtn.addEventListener('click', () => this.removeToast(toast));

        if (options.action) {
            const actionBtn = toast.querySelector('.toast-action');
            if (actionBtn) {
                actionBtn.addEventListener('click', () => {
                    try { options.action.callback(); } catch (e) { console.error('Toast action error', e); }
                    this.removeToast(toast);
                });
            }
        }

        container.appendChild(toast);

        // Auto-remove after delay
        setTimeout(() => {
            if (toast.parentNode) this.removeToast(toast);
        }, options.duration || 5000);

        // Animate in
        requestAnimationFrame(() => {
            toast.style.transform = 'translateX(0)';
            toast.style.opacity = '1';
        });
  }

  addActivityLog(log) {
    const list = document.getElementById('activity-list');
    const template = document.getElementById('activity-item-template');

    const item = template.content.cloneNode(true);

    // Set icon based on log level
    const iconElement = item.querySelector('.activity-icon-inner');
    iconElement.className = `activity-icon-inner icon-${this.getLogIcon(log.level)}`;

    // Set content
    item.querySelector('.activity-message').textContent = log.message;
    item.querySelector('.activity-timestamp').textContent = log.timestamp;

    // Add to top of list
    list.insertBefore(item, list.firstChild);

    // Limit to 50 items
    while (list.children.length > 50) {
      list.removeChild(list.lastChild);
    }

    // Scroll to top
    list.scrollTop = 0;
  }

  // Utility methods
  formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  }

  capitalizeDifficulty(difficulty) {
    const names = {
      'kantan': 'Kantan',
      'futsuu': 'Futsuu',
      'muzukashii': 'Muzukashii',
      'oni': 'Oni',
      'ura': 'Ura Oni'
    };
    return names[difficulty] || difficulty;
  }

  getToastIcon(type) {
    const icons = {
      success: '✅',
      error: '❌',
      warning: '⚠️',
      info: 'ℹ️'
    };
    return icons[type] || icons.info;
  }

  getLogIcon(level) {
    const icons = {
      success: 'check-circle',
      error: 'x-circle',
      warning: 'alert-triangle',
      info: 'info'
    };
    return icons[level] || icons.info;
  }
}
// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.taikoApp = new TaikoNationApp();
    window.taikoUI = new TaikoNationUI();
});