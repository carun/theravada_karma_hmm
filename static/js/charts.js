/**
 * Chart visualization handlers for Theravada Karma HMM Web UI
 * Manages Plotly.js charts for evolution, network, and pattern visualizations
 */

class Charts {
    constructor() {
        this.charts = {
            evolution: null,
            network: null,
            patterns: null
        };

        this.init();
    }

    init() {
        this.initializeCharts();
    }

    initializeCharts() {
        // Initialize empty charts
        this.initEvolutionChart();
        this.initNetworkChart();
        this.initPatternsChart();
    }

    initEvolutionChart() {
        const layout = {
            title: {
                text: 'Karmic Evolution Over Time',
                font: { size: 16 }
            },
            xaxis: {
                title: 'Time',
                showgrid: true,
                gridcolor: '#f0f0f0'
            },
            yaxis: {
                title: 'Karmic Value',
                showgrid: true,
                gridcolor: '#f0f0f0'
            },
            legend: {
                x: 0,
                y: 1,
                bgcolor: 'rgba(255,255,255,0.8)'
            },
            margin: { l: 50, r: 20, t: 50, b: 50 },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            hovermode: 'x unified'
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            displaylogo: false
        };

        // Empty initial data
        const data = [{
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'No data available',
            line: { color: '#cccccc' }
        }];

        Plotly.newPlot('evolution-chart', data, layout, config);
        this.charts.evolution = true;
    }

    initNetworkChart() {
        const layout = {
            title: {
                text: 'Kilesa Network Relationships',
                font: { size: 16 }
            },
            showlegend: false,
            margin: { l: 20, r: 20, t: 50, b: 20 },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            xaxis: { visible: false },
            yaxis: { visible: false },
            hovermode: 'closest'
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d'],
            displaylogo: false
        };

        // Empty initial data
        const data = [{
            x: [0],
            y: [0],
            type: 'scatter',
            mode: 'markers+text',
            text: ['No active kilesas'],
            textposition: 'middle center',
            marker: { size: 20, color: '#cccccc' }
        }];

        Plotly.newPlot('network-chart', data, layout, config);
        this.charts.network = true;
    }

    initPatternsChart() {
        const layout = {
            title: {
                text: 'Karmic Seed Patterns',
                font: { size: 16 }
            },
            margin: { l: 50, r: 20, t: 50, b: 100 },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            xaxis: {
                title: 'Seed Type',
                tickangle: -45
            },
            yaxis: {
                title: 'Count',
                showgrid: true,
                gridcolor: '#f0f0f0'
            }
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            displaylogo: false
        };

        // Empty initial data
        const data = [{
            x: ['No data'],
            y: [0],
            type: 'bar',
            marker: { color: '#cccccc' }
        }];

        Plotly.newPlot('patterns-chart', data, layout, config);
        this.charts.patterns = true;
    }

    updateChart(vizType, chartData) {
        switch (vizType) {
            case 'evolution':
                this.updateEvolutionChart(chartData);
                break;
            case 'network':
                this.updateNetworkChart(chartData);
                break;
            case 'patterns':
                this.updatePatternsChart(chartData);
                break;
            default:
                console.warn('Unknown visualization type:', vizType);
        }
    }

    updateEvolutionChart(data) {
        if (!data || data.message) {
            // No data available
            const emptyData = [{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines+markers',
                name: data?.message || 'No data available',
                line: { color: '#cccccc' }
            }];

            Plotly.react('evolution-chart', emptyData);
            return;
        }

        const traces = [];

        // Wholesome karma trace
        if (data.wholesome && data.wholesome.length > 0) {
            traces.push({
                x: data.time,
                y: data.wholesome,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Wholesome Karma',
                line: { color: '#27ae60', width: 2 },
                marker: { size: 6 }
            });
        }

        // Unwholesome karma trace
        if (data.unwholesome && data.unwholesome.length > 0) {
            traces.push({
                x: data.time,
                y: data.unwholesome,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Unwholesome Karma',
                line: { color: '#e74c3c', width: 2 },
                marker: { size: 6 }
            });
        }

        // Active seeds trace
        if (data.active_seeds && data.active_seeds.length > 0) {
            traces.push({
                x: data.time,
                y: data.active_seeds,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Active Seeds',
                line: { color: '#3498db', width: 2 },
                marker: { size: 6 },
                yaxis: 'y2'
            });
        }

        // Karmic balance trace
        if (data.karmic_balance && data.karmic_balance.length > 0) {
            traces.push({
                x: data.time,
                y: data.karmic_balance,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Karmic Balance',
                line: { color: '#f39c12', width: 2 },
                marker: { size: 4 }
            });
        }

        // Meditation effectiveness trace
        if (data.meditation_effectiveness && data.meditation_effectiveness.length > 0) {
            traces.push({
                x: data.time,
                y: data.meditation_effectiveness,
                type: 'scatter',
                mode: 'lines',
                name: 'Meditation Effectiveness',
                line: { color: '#9b59b6', width: 2, dash: 'dot' },
                yaxis: 'y2'
            });
        }

        // Kilesa suppression rate trace
        if (data.kilesa_suppression_rate && data.kilesa_suppression_rate.length > 0) {
            traces.push({
                x: data.time,
                y: data.kilesa_suppression_rate,
                type: 'scatter',
                mode: 'lines',
                name: 'Kilesa Suppression Rate',
                line: { color: '#16a085', width: 1, dash: 'dash' },
                yaxis: 'y2'
            });
        }

        const layout = {
            title: {
                text: 'Karmic Evolution Over Time',
                font: { size: 16 }
            },
            xaxis: {
                title: data.time_label || 'Time',
                showgrid: true,
                gridcolor: '#f0f0f0'
            },
            yaxis: {
                title: 'Karmic Value',
                showgrid: true,
                gridcolor: '#f0f0f0'
            },
            yaxis2: {
                title: 'Active Seeds Count',
                overlaying: 'y',
                side: 'right',
                showgrid: false
            },
            legend: {
                x: 0,
                y: 1,
                bgcolor: 'rgba(255,255,255,0.8)'
            },
            margin: { l: 50, r: 60, t: 50, b: 50 },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            hovermode: 'x unified'
        };

        Plotly.react('evolution-chart', traces, layout);
    }

    updateNetworkChart(data) {
        if (!data || !data.nodes || data.nodes.length === 0) {
            // No nodes available
            const emptyData = [{
                x: [0],
                y: [0],
                type: 'scatter',
                mode: 'markers+text',
                text: ['No active kilesas'],
                textposition: 'middle center',
                marker: { size: 20, color: '#cccccc' }
            }];

            Plotly.react('network-chart', emptyData);
            return;
        }

        // Create network layout using force-directed positioning
        const nodes = data.nodes;
        const edges = data.edges || [];

        // Simple circular layout for nodes
        const nodePositions = this.calculateNodePositions(nodes);

        const traces = [];

        // Add edges first (so they appear behind nodes)
        if (edges.length > 0) {
            const edgeTraces = edges.map(edge => {
                const fromNode = nodes.find(n => n.id === edge.from);
                const toNode = nodes.find(n => n.id === edge.to);
                const fromPos = nodePositions[edge.from];
                const toPos = nodePositions[edge.to];

                if (fromPos && toPos) {
                    return {
                        x: [fromPos.x, toPos.x, null],
                        y: [fromPos.y, toPos.y, null],
                        type: 'scatter',
                        mode: 'lines',
                        line: {
                            color: edge.color || '#cccccc',
                            width: Math.max(1, edge.value * 5)
                        },
                        hoverinfo: 'text',
                        text: edge.title,
                        showlegend: false
                    };
                }
                return null;
            }).filter(Boolean);

            traces.push(...edgeTraces);
        }

        // Add nodes
        const nodeTrace = {
            x: nodes.map(n => nodePositions[n.id].x),
            y: nodes.map(n => nodePositions[n.id].y),
            type: 'scatter',
            mode: 'markers+text',
            text: nodes.map(n => n.label),
            textposition: 'middle center',
            textfont: { size: 10, color: 'white' },
            marker: {
                size: nodes.map(n => Math.max(20, n.value * 50)),
                color: nodes.map(n => n.color || '#3498db'),
                line: { color: 'white', width: 2 }
            },
            hoverinfo: 'text',
            hovertext: nodes.map(n => `${n.label}: ${n.value.toFixed(2)}`),
            showlegend: false
        };

        traces.push(nodeTrace);

        const layout = {
            title: {
                text: 'Kilesa Network Relationships',
                font: { size: 16 }
            },
            showlegend: false,
            margin: { l: 20, r: 20, t: 50, b: 20 },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            xaxis: {
                visible: false,
                range: [-1.2, 1.2]
            },
            yaxis: {
                visible: false,
                range: [-1.2, 1.2]
            },
            hovermode: 'closest'
        };

        Plotly.react('network-chart', traces, layout);
    }

    updatePatternsChart(data) {
        if (!data || !data.patterns) {
            // No pattern data
            const emptyData = [{
                x: ['No data'],
                y: [0],
                type: 'bar',
                marker: { color: '#cccccc' }
            }];

            Plotly.react('patterns-chart', emptyData);
            return;
        }

        const patterns = data.patterns;
        const categories = Object.keys(patterns);
        const values = Object.values(patterns);

        const trace = {
            x: categories,
            y: values,
            type: 'bar',
            marker: {
                color: values.map(v => {
                    if (v > 50) return '#e74c3c';  // High - red
                    if (v > 20) return '#f39c12';  // Medium - orange
                    return '#27ae60';              // Low - green
                }),
                line: { color: 'white', width: 1 }
            },
            hovertemplate: '<b>%{x}</b><br>Count: %{y}<extra></extra>'
        };

        const layout = {
            title: {
                text: 'Karmic Seed Patterns',
                font: { size: 16 }
            },
            margin: { l: 50, r: 20, t: 50, b: 100 },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            xaxis: {
                title: 'Seed Type',
                tickangle: -45
            },
            yaxis: {
                title: 'Count',
                showgrid: true,
                gridcolor: '#f0f0f0'
            }
        };

        Plotly.react('patterns-chart', [trace], layout);
    }

    calculateNodePositions(nodes) {
        const positions = {};
        const numNodes = nodes.length;

        if (numNodes === 1) {
            positions[nodes[0].id] = { x: 0, y: 0 };
            return positions;
        }

        // Arrange nodes in a circle
        nodes.forEach((node, i) => {
            const angle = (2 * Math.PI * i) / numNodes;
            const radius = Math.min(0.8, 0.3 + numNodes * 0.1);

            positions[node.id] = {
                x: radius * Math.cos(angle),
                y: radius * Math.sin(angle)
            };
        });

        return positions;
    }

    resizeCharts() {
        // Trigger resize for all charts
        Object.keys(this.charts).forEach(chartType => {
            if (this.charts[chartType]) {
                const chartId = `${chartType}-chart`;
                if (document.getElementById(chartId)) {
                    Plotly.Plots.resize(chartId);
                }
            }
        });
    }
}

// Handle window resize
window.addEventListener('resize', () => {
    if (window.Charts) {
        window.Charts.resizeCharts();
    }
});

// Initialize charts when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.Charts = new Charts();
});
