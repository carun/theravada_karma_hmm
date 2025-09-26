/**
 * Main application JavaScript for Theravada Karma HMM Web UI
 * Handles WebSocket communication and global state management
 */

class KarmaApp {
    constructor() {
        this.sessionId = null;
        this.websocket = null;
        this.isConnected = false;
        this.karmaModel = null;

        // Continuous simulation state
        this.isRunning = false;
        this.simulationInterval = null;
        this.simulationSpeed = 1000; // milliseconds between steps
        this.autoActionProbability = 0.3; // chance of random action per step
        this.totalSteps = 0;
        this.startTime = null;

        this.init();
    }

    async init() {
        // Generate session ID
        await this.createSession();

        // Initialize WebSocket connection
        this.connectWebSocket();

        // Initialize UI components
        this.initializeUI();

        // Bind event listeners
        this.bindEvents();
    }

    async createSession() {
        try {
            const response = await fetch('/api/session');
            const data = await response.json();
            this.sessionId = data.session_id;
            console.log('Session created:', this.sessionId);
        } catch (error) {
            console.error('Failed to create session:', error);
            this.showNotification('Failed to create session', 'error');
        }
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;

        this.websocket = new WebSocket(wsUrl);

        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.updateConnectionStatus('connected', 'Connected');
            this.enableControls();
        };

        this.websocket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
        };

        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus('disconnected', 'Disconnected');
            this.disableControls();

            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        };

        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus('disconnected', 'Connection Error');
        };
    }

    handleMessage(message) {
        console.log('Received message:', message);

        switch (message.type) {
            case 'simulation_initialized':
                this.handleSimulationInitialized(message.data);
                break;
            case 'meditation_added':
                this.handleMeditationAdded(message.data);
                break;
            case 'action_performed':
                this.handleActionPerformed(message.data);
                break;
            case 'time_advanced':
                this.handleTimeAdvanced(message.data);
                break;
            case 'visualization_data':
                this.handleVisualizationData(message.data);
                break;
            case 'simulation_reset':
                this.handleSimulationReset(message.data);
                break;
            case 'continuous_simulation_started':
                this.handleContinuousSimulationStarted(message.data);
                break;
            case 'continuous_simulation_stopped':
                this.handleContinuousSimulationStopped(message.data);
                break;
            case 'error':
                this.handleError(message.data);
                break;
            default:
                console.warn('Unknown message type:', message.type);
        }
    }

    sendMessage(type, data = {}) {
        if (this.websocket && this.isConnected) {
            const message = { type, data };
            this.websocket.send(JSON.stringify(message));
        } else {
            console.error('WebSocket not connected');
            this.showNotification('Connection lost. Please wait for reconnection.', 'warning');
        }
    }

    handleSimulationInitialized(data) {
        this.karmaModel = data;
        document.getElementById('current-time').textContent = data.current_time;
        this.showNotification('Simulation initialized successfully! You can now use quick actions or add meditation practices.', 'success');
        this.logActivity('Simulation', 'Initialized', data.current_time);

        // Enable dependent controls
        document.getElementById('add-meditation-btn').disabled = false;
        document.getElementById('perform-action-btn').disabled = false;
        document.getElementById('advance-time-btn').disabled = false;

        // Show toggle simulation button and hide initialize button
        document.getElementById('init-simulation-btn').style.display = 'none';
        document.getElementById('toggle-simulation-btn').style.display = 'block';

        // Request initial visualization data for all tabs
        this.requestVisualizationData('evolution');
        this.requestVisualizationData('network');
        this.requestVisualizationData('patterns');

        // Show initial state info
        this.updateStateDisplay({
            path_stage: 'WORLDLING',
            karmic_balance: 0.0,
            active_seeds_count: 0
        });
    }

    handleMeditationAdded(data) {
        this.showNotification(`${data.practice_type} meditation added`, 'success');
        this.logActivity('Meditation', `Added ${data.practice_type}`, new Date().toLocaleTimeString());
        document.getElementById('practice-count').textContent = data.total_practices;
    }

    handleActionPerformed(data) {
        this.updateStateDisplay(data.state);
        document.getElementById('current-time').textContent = data.current_time;
        this.showNotification(`Action performed (${data.seeds_created} seeds created)`, 'success');
        this.logActivity('Action', `Created ${data.seeds_created} karmic seeds`, data.current_time);

        // Refresh visualizations
        this.requestVisualizationData('evolution');
    }

    handleTimeAdvanced(data) {
        this.updateStateDisplay(data.state);
        document.getElementById('current-time').textContent = data.current_time;
        this.showNotification(`Advanced ${data.steps} time step(s)`, 'info');
        this.logActivity('Time', `Advanced ${data.steps} step(s)`, data.current_time);

        // Refresh visualizations
        this.requestVisualizationData('evolution');
    }

    handleVisualizationData(data) {
        // Pass to charts module
        if (window.Charts) {
            window.Charts.updateChart(data.viz_type, data.chart_data);
        }
    }

    handleSimulationReset(data) {
        this.karmaModel = null;
        this.isRunning = false;
        this.stopContinuousSimulation();
        this.showNotification('Simulation reset', 'info');
        this.clearActivityLog();
        this.resetStateDisplay();
        this.removeRunningVisualEffects();

        // Reset UI to initial state
        document.getElementById('init-simulation-btn').style.display = 'block';
        document.getElementById('toggle-simulation-btn').style.display = 'none';

        // Disable dependent controls
        document.getElementById('add-meditation-btn').disabled = true;
        document.getElementById('perform-action-btn').disabled = true;
        document.getElementById('advance-time-btn').disabled = true;
    }

    handleError(data) {
        console.error('Server error:', data.message);
        this.showNotification(data.message, 'error');
    }

    handleContinuousSimulationStarted(data) {
        this.isRunning = true;
        this.startTime = Date.now();
        this.totalSteps = 0;
        this.startContinuousSimulation();
        this.updateSimulationControls();
        this.addRunningVisualEffects();
        this.showNotification('Continuous simulation started', 'success');
        this.logActivity('Simulation', 'Started continuous mode', new Date().toLocaleTimeString());
    }

    handleContinuousSimulationStopped(data) {
        this.isRunning = false;
        this.stopContinuousSimulation();
        this.updateSimulationControls();
        this.removeRunningVisualEffects();
        this.showNotification('Continuous simulation stopped', 'info');
        this.logActivity('Simulation', `Stopped after ${this.totalSteps} steps`, new Date().toLocaleTimeString());
    }

    updateConnectionStatus(status, text) {
        const indicator = document.getElementById('connection-status');
        const statusText = document.getElementById('connection-text');

        indicator.className = `status-indicator ${status}`;
        statusText.textContent = text;
    }

    updateStateDisplay(state) {
        if (state) {
            document.getElementById('path-stage').textContent = state.path_stage || 'Not Started';
            document.getElementById('path-stage').className = `badge bg-${this.getPathStageColor(state.path_stage)} ms-2`;
            document.getElementById('karmic-balance').textContent = (state.karmic_balance || 0).toFixed(2);
            document.getElementById('active-seeds').textContent = state.active_seeds_count || 0;
        }
    }

    resetStateDisplay() {
        document.getElementById('path-stage').textContent = 'Not Started';
        document.getElementById('path-stage').className = 'badge bg-secondary ms-2';
        document.getElementById('karmic-balance').textContent = '0.00';
        document.getElementById('active-seeds').textContent = '0';
        document.getElementById('practice-count').textContent = '0';
        document.getElementById('current-time').textContent = 'Not initialized';
    }

    getPathStageColor(stage) {
        const colorMap = {
            'WORLDLING': 'secondary',
            'STREAM_ENTERER': 'info',
            'ONCE_RETURNER': 'primary',
            'NON_RETURNER': 'warning',
            'ARAHAT': 'success'
        };
        return colorMap[stage] || 'secondary';
    }

    logActivity(type, action, time) {
        const log = document.getElementById('activity-log');
        const item = document.createElement('div');
        item.className = 'activity-item fade-in';

        const typeClass = type.toLowerCase();
        item.innerHTML = `
            <span class="action-type ${typeClass}">${type}</span>
            ${action}
            <span class="timestamp">${time}</span>
        `;

        // Add to top of log
        const firstChild = log.firstElementChild;
        if (firstChild && firstChild.classList.contains('text-muted')) {
            log.removeChild(firstChild);
        }

        log.insertBefore(item, log.firstChild);

        // Limit to 50 items
        while (log.children.length > 50) {
            log.removeChild(log.lastChild);
        }
    }

    clearActivityLog() {
        const log = document.getElementById('activity-log');
        log.innerHTML = '<div class="p-3 text-muted text-center">No activities yet</div>';
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;

        document.body.appendChild(notification);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }

    requestVisualizationData(type) {
        this.sendMessage('get_visualization_data', { type });
    }

    enableControls() {
        document.getElementById('init-simulation-btn').disabled = false;
        document.getElementById('reset-btn').disabled = false;
    }

    disableControls() {
        const buttons = document.querySelectorAll('button');
        buttons.forEach(btn => {
            if (!btn.id.includes('tab')) {
                btn.disabled = true;
            }
        });
    }

    initializeUI() {
        // Initialize range input displays
        this.updateRangeDisplays();
    }

    bindEvents() {
        // Range input updates
        document.addEventListener('input', (e) => {
            if (e.target.type === 'range') {
                const display = e.target.parentElement.querySelector('small');
                if (display) {
                    display.textContent = e.target.value;
                }
            }
        });

        // Tab change events for visualizations
        document.addEventListener('shown.bs.tab', (e) => {
            const tabId = e.target.getAttribute('data-bs-target');
            if (tabId === '#evolution-panel') {
                this.requestVisualizationData('evolution');
            } else if (tabId === '#network-panel') {
                this.requestVisualizationData('network');
            } else if (tabId === '#patterns-panel') {
                this.requestVisualizationData('patterns');
            }
        });
    }

    updateRangeDisplays() {
        document.querySelectorAll('input[type="range"]').forEach(range => {
            const display = range.parentElement.querySelector('small');
            if (display) {
                display.textContent = range.value;
            }
        });
    }

    startContinuousSimulation() {
        if (this.simulationInterval) {
            clearInterval(this.simulationInterval);
        }

        this.simulationInterval = setInterval(() => {
            this.performSimulationStep();
        }, this.simulationSpeed);
    }

    stopContinuousSimulation() {
        if (this.simulationInterval) {
            clearInterval(this.simulationInterval);
            this.simulationInterval = null;
        }
    }

    performSimulationStep() {
        this.totalSteps++;

        // Randomly decide if an action should occur this step
        if (Math.random() < this.autoActionProbability) {
            this.performRandomAction();
        }

        // Always advance time
        this.sendMessage('advance_time', {
            steps: 1,
            context: { stress_level: 0.5 }
        });

        // Update statistics display
        this.updateSimulationStats();
    }

    performRandomAction() {
        // Context-sensitive action selection based on simulation time and karma balance
        const currentTime = this.totalSteps;
        const stressLevel = parseFloat(document.getElementById('stress-level')?.value || 0.5);

        // Define action probabilities based on context
        let actionWeights = {
            anger: 0.15 + stressLevel * 0.2,     // Higher when stressed
            greed: 0.20 + (currentTime % 20) * 0.01, // Periodic desires
            delusion: 0.10 + Math.sin(currentTime * 0.1) * 0.05, // Cyclical confusion
            compassion: 0.25 - stressLevel * 0.1,  // Lower when stressed
            meditation: 0.20 + (this.totalSteps > 50 ? 0.1 : 0), // Increases with practice
            generosity: 0.10
        };

        // Adjust based on meditation practices (if any exist)
        const practiceCount = parseInt(document.getElementById('practice-count')?.textContent || 0);
        if (practiceCount > 0) {
            actionWeights.meditation += 0.15;
            actionWeights.compassion += 0.1;
            actionWeights.anger *= 0.7;
            actionWeights.greed *= 0.8;
        }

        // Select action based on weighted probabilities
        const actions = [
            { type: 'anger', weight: actionWeights.anger },
            { type: 'greed', weight: actionWeights.greed },
            { type: 'delusion', weight: actionWeights.delusion },
            { type: 'compassion', weight: actionWeights.compassion },
            { type: 'meditation', weight: actionWeights.meditation },
            { type: 'generosity', weight: actionWeights.generosity }
        ];

        const totalWeight = actions.reduce((sum, action) => sum + action.weight, 0);
        let random = Math.random() * totalWeight;
        let selectedAction = actions[0];

        for (const action of actions) {
            random -= action.weight;
            if (random <= 0) {
                selectedAction = action;
                break;
            }
        }

        // Generate action parameters based on type
        const actionConfigs = {
            anger: {
                intention_strength: 0.5 + Math.random() * 0.4 + stressLevel * 0.2,
                active_kilesas: {
                    anger: 0.6 + Math.random() * 0.4,
                    hatred: Math.random() * 0.6,
                    aversion: Math.random() * 0.5
                },
                wholesome: false
            },
            greed: {
                intention_strength: 0.4 + Math.random() * 0.5,
                active_kilesas: {
                    greed: 0.7 + Math.random() * 0.3,
                    craving: Math.random() * 0.8,
                    attachment: Math.random() * 0.6
                },
                wholesome: false
            },
            delusion: {
                intention_strength: 0.3 + Math.random() * 0.4,
                active_kilesas: {
                    delusion: 0.5 + Math.random() * 0.5,
                    ignorance: Math.random() * 0.7,
                    wrong_view: Math.random() * 0.4
                },
                wholesome: false
            },
            compassion: {
                intention_strength: 0.6 + Math.random() * 0.4,
                active_kilesas: {},
                wholesome: true
            },
            meditation: {
                intention_strength: 0.7 + Math.random() * 0.3,
                active_kilesas: {},
                wholesome: true
            },
            generosity: {
                intention_strength: 0.5 + Math.random() * 0.4,
                active_kilesas: {},
                wholesome: true
            }
        };

        this.sendMessage('perform_action', actionConfigs[selectedAction.type]);
    }

    updateSimulationControls() {
        const toggleBtn = document.getElementById('toggle-simulation-btn');

        if (this.isRunning) {
            if (toggleBtn) {
                toggleBtn.innerHTML = '<i class="fas fa-stop me-2"></i>Stop Simulation';
                toggleBtn.className = 'btn btn-danger';
                toggleBtn.classList.add('btn-running');
            }
        } else {
            if (toggleBtn) {
                toggleBtn.innerHTML = '<i class="fas fa-play me-2"></i>Start Continuous Simulation';
                toggleBtn.className = 'btn btn-success';
                toggleBtn.classList.remove('btn-running');
            }
        }
    }

    updateSimulationStats() {
        if (this.startTime) {
            const elapsed = (Date.now() - this.startTime) / 1000; // seconds
            const stepsPerSecond = this.totalSteps / elapsed;

            const statsElement = document.getElementById('simulation-stats');
            if (statsElement) {
                statsElement.innerHTML = `
                    <small class="text-muted">
                        Steps: ${this.totalSteps} |
                        Elapsed: ${elapsed.toFixed(1)}s |
                        Speed: ${stepsPerSecond.toFixed(1)} steps/s
                    </small>
                `;
            }
        }
    }

    setSimulationSpeed(speed) {
        this.simulationSpeed = speed;
        if (this.isRunning) {
            this.startContinuousSimulation(); // Restart with new speed
        }
    }

    setAutoActionProbability(probability) {
        this.autoActionProbability = probability;
    }

    addRunningVisualEffects() {
        document.body.classList.add('simulation-running');
    }

    removeRunningVisualEffects() {
        document.body.classList.remove('simulation-running');
    }
}

// Quick action functions
function quickAction(type) {
    if (!window.app || !window.app.isConnected) {
        window.app.showNotification('Not connected to server', 'warning');
        return;
    }

    if (!window.app.karmaModel) {
        window.app.showNotification('Please initialize simulation first', 'warning');
        return;
    }

    const actions = {
        anger: {
            intention_strength: 0.8,
            active_kilesas: { anger: 0.9, hatred: 0.6, aversion: 0.7 },
            wholesome: false
        },
        greed: {
            intention_strength: 0.7,
            active_kilesas: { greed: 0.8, craving: 0.6, attachment: 0.5 },
            wholesome: false
        },
        compassion: {
            intention_strength: 0.8,
            active_kilesas: {},
            wholesome: true
        },
        meditation: {
            intention_strength: 0.9,
            active_kilesas: {},
            wholesome: true
        }
    };

    if (actions[type]) {
        window.app.sendMessage('perform_action', actions[type]);
        window.app.showNotification(`Performing ${type} action...`, 'info');
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new KarmaApp();
});
