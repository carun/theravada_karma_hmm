/**
 * Control panel handlers for Theravada Karma HMM Web UI
 * Manages simulation configuration, meditation practices, and action generation
 */

class Controls {
    constructor(app) {
        this.app = app;
        this.availableKilesas = [];
        this.init();
    }

    async init() {
        await this.loadKilesas();
        this.setupKilesaGrid();
        this.bindEvents();
    }

    async loadKilesas() {
        try {
            const response = await fetch('/api/kilesas');
            const data = await response.json();
            this.availableKilesas = data.kilesas;
        } catch (error) {
            console.error('Failed to load kilesas:', error);
            // Fallback list
            this.availableKilesas = [
                'anger', 'hatred', 'greed', 'delusion', 'conceit', 'envy',
                'jealousy', 'arrogance', 'vanity', 'hostility', 'resentment',
                'attachment', 'craving', 'aversion', 'ignorance', 'pride'
            ];
        }
    }

    setupKilesaGrid() {
        const container = document.getElementById('kilesa-checkboxes');
        container.innerHTML = '';

        this.availableKilesas.forEach(kilesa => {
            const item = document.createElement('div');
            item.className = 'kilesa-item';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.className = 'kilesa-checkbox form-check-input';
            checkbox.id = `kilesa-${kilesa}`;
            checkbox.value = kilesa;

            const label = document.createElement('label');
            label.className = 'kilesa-label';
            label.setAttribute('for', checkbox.id);
            label.textContent = kilesa.replace('_', ' ');

            const slider = document.createElement('input');
            slider.type = 'range';
            slider.className = 'kilesa-slider form-range';
            slider.min = '0';
            slider.max = '1';
            slider.step = '0.1';
            slider.value = '0.5';
            slider.disabled = true;

            // Enable/disable slider based on checkbox
            checkbox.addEventListener('change', () => {
                slider.disabled = !checkbox.checked;
                item.classList.toggle('active', checkbox.checked);
            });

            item.appendChild(checkbox);
            item.appendChild(label);
            item.appendChild(slider);
            container.appendChild(item);
        });
    }

    getActiveKilesas() {
        const kilesas = {};
        document.querySelectorAll('.kilesa-checkbox:checked').forEach(checkbox => {
            const slider = checkbox.parentElement.querySelector('.kilesa-slider');
            kilesas[checkbox.value] = parseFloat(slider.value);
        });
        return kilesas;
    }

    bindEvents() {
        // Initialize simulation
        document.getElementById('init-simulation-btn').addEventListener('click', () => {
            this.initializeSimulation();
        });

        // Add meditation practice
        document.getElementById('add-meditation-btn').addEventListener('click', () => {
            this.addMeditationPractice();
        });

        // Perform action
        document.getElementById('perform-action-btn').addEventListener('click', () => {
            this.performAction();
        });

        // Advance time
        document.getElementById('advance-time-btn').addEventListener('click', () => {
            this.advanceTime();
        });

        // Reset simulation
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetSimulation();
        });

        // Toggle continuous simulation
        document.getElementById('toggle-simulation-btn').addEventListener('click', () => {
            if (this.app.isRunning) {
                this.stopContinuousSimulation();
            } else {
                this.startContinuousSimulation();
            }
        });

        // Simulation speed control
        document.getElementById('simulation-speed').addEventListener('input', (e) => {
            const speed = parseInt(e.target.value);
            this.app.setSimulationSpeed(speed);
            this.updateSpeedDisplay(speed);
        });

        // Auto action rate control
        document.getElementById('auto-action-rate').addEventListener('input', (e) => {
            const rate = parseFloat(e.target.value);
            this.app.setAutoActionProbability(rate);
        });

        // Wholesome action toggle
        document.getElementById('wholesome-action').addEventListener('change', (e) => {
            const actionBtn = document.getElementById('perform-action-btn');
            if (e.target.checked) {
                actionBtn.innerHTML = '<i class="fas fa-heart me-2"></i>Perform Wholesome Action';
                actionBtn.className = 'btn btn-success w-100';
            } else {
                actionBtn.innerHTML = '<i class="fas fa-play me-2"></i>Perform Action';
                actionBtn.className = 'btn btn-warning w-100';
            }
        });
    }

    initializeSimulation() {
        const timeUnit = document.getElementById('time-unit').value;
        const timeScaleFactor = parseFloat(document.getElementById('time-scale').value);

        const config = {
            time_unit: timeUnit,
            time_scale_factor: timeScaleFactor
        };

        this.app.sendMessage('init_simulation', config);

        // Show loading state
        const btn = document.getElementById('init-simulation-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Initializing...';
        btn.disabled = true;

        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }, 2000);
    }

    addMeditationPractice() {
        const practiceConfig = {
            practice_type: document.getElementById('meditation-type').value,
            daily_duration: parseFloat(document.getElementById('daily-duration').value),
            consistency: parseFloat(document.getElementById('consistency').value),
            quality: parseFloat(document.getElementById('quality').value),
            years_practiced: parseFloat(document.getElementById('years-practiced').value),
            teacher_guidance: parseFloat(document.getElementById('teacher-guidance').value),
            retreat_hours: parseInt(document.getElementById('retreat-hours').value)
        };

        // Validation
        if (practiceConfig.daily_duration <= 0) {
            this.app.showNotification('Daily duration must be greater than 0', 'warning');
            return;
        }

        if (practiceConfig.years_practiced <= 0) {
            this.app.showNotification('Years practiced must be greater than 0', 'warning');
            return;
        }

        this.app.sendMessage('add_meditation', practiceConfig);

        // Show loading state
        const btn = document.getElementById('add-meditation-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Adding...';
        btn.disabled = true;

        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }, 1000);
    }

    performAction() {
        const activeKilesas = this.getActiveKilesas();
        const intentionStrength = parseFloat(document.getElementById('intention-strength').value);
        const wholesome = document.getElementById('wholesome-action').checked;

        const actionConfig = {
            intention_strength: intentionStrength,
            active_kilesas: activeKilesas,
            wholesome: wholesome
        };

        // Validation
        if (intentionStrength <= 0) {
            this.app.showNotification('Intention strength must be greater than 0', 'warning');
            return;
        }

        if (!wholesome && Object.keys(activeKilesas).length === 0) {
            this.app.showNotification('Please select at least one kilesa for unwholesome actions', 'warning');
            return;
        }

        this.app.sendMessage('perform_action', actionConfig);

        // Show loading state
        const btn = document.getElementById('perform-action-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        btn.disabled = true;

        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }, 1000);
    }

    advanceTime() {
        const timeSteps = parseInt(document.getElementById('time-steps').value);
        const stressLevel = parseFloat(document.getElementById('stress-level').value);

        const timeConfig = {
            steps: timeSteps,
            context: {
                stress_level: stressLevel
            }
        };

        // Validation
        if (timeSteps <= 0) {
            this.app.showNotification('Time steps must be greater than 0', 'warning');
            return;
        }

        this.app.sendMessage('advance_time', timeConfig);

        // Show loading state
        const btn = document.getElementById('advance-time-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Advancing...';
        btn.disabled = true;

        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }, 1000);
    }

    resetSimulation() {
        if (confirm('Are you sure you want to reset the simulation? All progress will be lost.')) {
            this.app.sendMessage('reset_simulation');

            // Reset form values to defaults
            this.resetFormValues();
        }
    }

    resetFormValues() {
        // Reset simulation config
        document.getElementById('time-unit').value = 'MONTHS';
        document.getElementById('time-scale').value = '1.0';

        // Reset meditation config
        document.getElementById('meditation-type').value = 'VIPASSANA';
        document.getElementById('daily-duration').value = '1.0';
        document.getElementById('consistency').value = '0.8';
        document.getElementById('quality').value = '0.7';
        document.getElementById('years-practiced').value = '1';
        document.getElementById('teacher-guidance').value = '0.5';
        document.getElementById('retreat-hours').value = '0';

        // Reset action config
        document.getElementById('wholesome-action').checked = false;
        document.getElementById('intention-strength').value = '0.7';

        // Reset kilesa selections
        document.querySelectorAll('.kilesa-checkbox').forEach(checkbox => {
            checkbox.checked = false;
        });
        document.querySelectorAll('.kilesa-slider').forEach(slider => {
            slider.value = '0.5';
            slider.disabled = true;
        });
        document.querySelectorAll('.kilesa-item').forEach(item => {
            item.classList.remove('active');
        });

        // Reset time control
        document.getElementById('time-steps').value = '1';
        document.getElementById('stress-level').value = '0.5';

        // Update range displays
        this.updateRangeDisplays();

        // Reset action button appearance
        const actionBtn = document.getElementById('perform-action-btn');
        actionBtn.innerHTML = '<i class="fas fa-play me-2"></i>Perform Action';
        actionBtn.className = 'btn btn-warning w-100';
    }

    updateRangeDisplays() {
        document.querySelectorAll('input[type="range"]').forEach(range => {
            const display = range.parentElement.querySelector('small');
            if (display) {
                display.textContent = range.value;
            }
        });
    }

    // Preset configurations
    loadPreset(presetName) {
        const presets = {
            beginner: {
                meditation_type: 'VIPASSANA',
                daily_duration: 0.5,
                consistency: 0.6,
                quality: 0.5,
                years_practiced: 0.5,
                teacher_guidance: 0.8,
                retreat_hours: 0
            },
            intermediate: {
                meditation_type: 'VIPASSANA',
                daily_duration: 1.0,
                consistency: 0.8,
                quality: 0.7,
                years_practiced: 2,
                teacher_guidance: 0.6,
                retreat_hours: 20
            },
            advanced: {
                meditation_type: 'VIPASSANA',
                daily_duration: 2.0,
                consistency: 0.9,
                quality: 0.8,
                years_practiced: 10,
                teacher_guidance: 0.4,
                retreat_hours: 200
            }
        };

        const preset = presets[presetName];
        if (preset) {
            Object.keys(preset).forEach(key => {
                const element = document.getElementById(key.replace('_', '-'));
                if (element) {
                    element.value = preset[key];
                }
            });
            this.updateRangeDisplays();
        }
    }

    startContinuousSimulation() {
        if (!this.app.karmaModel) {
            this.app.showNotification('Please initialize simulation first', 'warning');
            return;
        }

        this.app.sendMessage('start_continuous_simulation', {
            speed: parseInt(document.getElementById('simulation-speed').value),
            auto_action_rate: parseFloat(document.getElementById('auto-action-rate').value)
        });

        // Show loading state
        const btn = document.getElementById('toggle-simulation-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Starting...';
        btn.disabled = true;

        setTimeout(() => {
            btn.disabled = false;
        }, 1000);
    }

    stopContinuousSimulation() {
        this.app.sendMessage('stop_continuous_simulation', {});

        // Show loading state
        const btn = document.getElementById('toggle-simulation-btn');
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Stopping...';
        btn.disabled = true;

        setTimeout(() => {
            btn.disabled = false;
        }, 1000);
    }

    updateSpeedDisplay(speed) {
        const speedText = (speed / 1000).toFixed(1) + 's';
        const label = document.querySelector('label[for="simulation-speed"]');
        if (label) {
            label.innerHTML = `Simulation Speed <small class="text-muted">(${speedText} per step)</small>`;
        }
    }
}

// Initialize controls when app is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait for app to be initialized
    setTimeout(() => {
        if (window.app) {
            window.controls = new Controls(window.app);
        }
    }, 100);
});
