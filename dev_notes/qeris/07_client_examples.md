# Client Examples

## Python MCP Client

### Basic Connection and Experiment Execution

```python
from mcp import Client
import asyncio

async def run_simple_experiment():
    client = Client()
    await client.connect('http://localhost:8765')
    
    # List available experiments
    experiments = await client.call_tool('list_experiments')
    print(f"Available experiments: {[e['name'] for e in experiments['experiments']]}")
    
    # Get detailed info
    info = await client.call_tool('get_experiment_info', {
        'experiment_name': 'RabiExperiment'
    })
    print(f"Parameters: {info['parameters']}")
    
    # Run experiment
    exp_id = await client.call_tool('run_experiment', {
        'experiment_name': 'RabiExperiment',
        'qubits': ['q0'],
        'parameters': {
            'start': 0,
            'stop': 1,
            'step': 0.05,
            'shots': 1024
        }
    })
    
    # Monitor progress
    while True:
        status = await client.call_tool('get_status')
        print(f"Progress: {status['experiment']['progress'] * 100:.1f}%")
        if status['experiment']['state'] == 'completed':
            break
        await asyncio.sleep(1)
    
    # Get results
    results = await client.call_tool('get_results', {
        'experiment_id': exp_id,
        'format': 'processed'
    })
    return results

asyncio.run(run_simple_experiment())
```

### Real-time Data Monitoring

```python
import asyncio
from mcp import Client
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class QERISMonitor:
    def __init__(self, server_url='http://localhost:8765'):
        self.client = Client()
        self.server_url = server_url
        self.data_x = []
        self.data_y = []
        
    async def connect(self):
        await self.client.connect(self.server_url)
        
    async def run_and_monitor(self, experiment_type, **params):
        # Set up live plot
        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'b-')
        
        def update_plot(frame):
            line.set_data(self.data_x, self.data_y)
            ax.relim()
            ax.autoscale_view()
            return line,
        
        # Start animation
        ani = FuncAnimation(fig, update_plot, interval=500, blit=True)
        plt.show(block=False)
        
        # Subscribe to live data
        async def collect_data():
            async for resource in self.client.read_resource('qeris://live_data'):
                data = json.loads(resource.text)
                if data['type'] == 'data_point':
                    self.data_x.extend(data['data']['x'])
                    self.data_y.extend(data['data']['y'])
        
        # Start data collection
        data_task = asyncio.create_task(collect_data())
        
        # Run experiment
        exp_id = await self.client.call_tool('run_experiment', {
            'type': experiment_type,
            **params
        })
        
        # Wait for completion
        async for resource in self.client.read_resource('qeris://experiment_status'):
            status = json.loads(resource.text)
            if status['experiment']['state'] == 'completed':
                break
                
        data_task.cancel()
        return exp_id

# Usage
monitor = QERISMonitor()
await monitor.connect()
await monitor.run_and_monitor('rabi', qubits=['q0'], parameters={
    'start': 0, 'stop': 5, 'step': 0.1
})
```

## Web Client

### HTML Interface

```html
<!-- qeris_client.html -->
<!DOCTYPE html>
<html>
<head>
    <title>QERIS Monitor</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: monospace; margin: 20px; }
        .status { margin: 10px 0; }
        #plot { width: 800px; height: 400px; }
        pre { background: #f0f0f0; padding: 10px; }
        button { margin: 5px; padding: 10px; }
    </style>
</head>
<body>
    <h1>Quantum Experiment Monitor</h1>
    <div class="status">Status: <span id="connection">Disconnected</span></div>
    
    <div class="controls">
        <select id="experiment-select" onchange="updateExperimentInfo()">
            <option value="">Select experiment...</option>
        </select>
        <button onclick="runSelectedExperiment()">Run</button>
        <button onclick="stopExperiment()">Stop</button>
        <button onclick="resetHardware('full')">Reset Hardware</button>
        <button onclick="resetServer(true)">Reset Server</button>
    </div>
    
    <div id="experiment-info" style="display: none;">
        <h3>Experiment Details</h3>
        <pre id="experiment-details"></pre>
    </div>
    
    <div id="plot"></div>
    <pre id="status">Ready</pre>
    
    <script src="qeris_client.js"></script>
</body>
</html>
```

### JavaScript MCP Client

```javascript
// qeris_client.js
const MCP_URL = 'http://localhost:8765';
let experimentId = null;
let plotData = { x: [], y: [] };

// MCP client implementation
class MCPClient {
    async callTool(name, params = {}) {
        const response = await fetch(`${MCP_URL}/tools/${name}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        return response.json();
    }
    
    async subscribeResource(uri, callback) {
        // Subscribe to MCP resource updates
        const eventSource = new EventSource(`${MCP_URL}/resources/subscribe?uri=${uri}`);
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            callback(data);
        };
        return eventSource;
    }
}

const client = new MCPClient();

// Subscribe to live data
client.subscribeResource('qeris://live_data', (data) => {
    const parsed = JSON.parse(data.text);
    if (parsed.type === 'data_point') {
        plotData.x = plotData.x.concat(parsed.data.x);
        plotData.y = plotData.y.concat(parsed.data.y);
        
        Plotly.newPlot('plot', [{
            x: plotData.x,
            y: plotData.y,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Experiment Data'
        }], {
            title: 'Live Experiment Data',
            xaxis: { title: parsed.data.dimensions?.[0] || 'X' },
            yaxis: { title: parsed.data.dimensions?.[1] || 'Y' }
        });
    }
});

// Subscribe to status updates
client.subscribeResource('qeris://experiment_status', (data) => {
    const status = JSON.parse(data.text);
    document.getElementById('status').textContent = JSON.stringify(status, null, 2);
    document.getElementById('connection').textContent = 
        `${status.experiment.state} (${(status.experiment.progress * 100).toFixed(1)}%)`;
});

// Load experiments on startup
async function loadExperiments() {
    const experiments = await client.callTool('list_experiments');
    const select = document.getElementById('experiment-select');
    
    experiments.experiments.forEach(exp => {
        const option = document.createElement('option');
        option.value = exp.name;
        option.textContent = `${exp.name} (${exp.category})`;
        select.appendChild(option);
    });
}

async function updateExperimentInfo() {
    const select = document.getElementById('experiment-select');
    const expName = select.value;
    
    if (!expName) {
        document.getElementById('experiment-info').style.display = 'none';
        return;
    }
    
    const info = await client.callTool('get_experiment_info', {
        experiment_name: expName
    });
    
    document.getElementById('experiment-details').textContent = 
        JSON.stringify(info, null, 2);
    document.getElementById('experiment-info').style.display = 'block';
}

async function runSelectedExperiment() {
    const select = document.getElementById('experiment-select');
    const expName = select.value;
    
    if (!expName) {
        alert('Please select an experiment');
        return;
    }
    
    // Get experiment info to use example parameters
    const info = await client.callTool('get_experiment_info', {
        experiment_name: expName
    });
    
    plotData = { x: [], y: [] };
    const result = await client.callTool('run_experiment', {
        experiment_name: expName,
        qubits: ['q0'],
        parameters: info.example.parameters
    });
    experimentId = result.experiment_id;
}

async function stopExperiment() {
    if (experimentId) {
        await client.callTool('stop_experiment', { experiment_id: experimentId });
    }
}

async function resetHardware(resetType) {
    if (confirm(`Are you sure you want to reset hardware (${resetType})?`)) {
        const result = await client.callTool('reset_hardware', { 
            reset_type: resetType 
        });
        alert(`Hardware reset: ${result.status}\n${JSON.stringify(result.details, null, 2)}`);
    }
}

async function resetServer(keepHardware) {
    if (confirm('Are you sure you want to reset the MCP server?')) {
        const result = await client.callTool('reset_server', { 
            keep_hardware_state: keepHardware 
        });
        alert(`Server reset: ${result.status}\n${JSON.stringify(result.details, null, 2)}`);
        // Reload experiments after server reset
        loadExperiments();
    }
}

// Initialize
loadExperiments();
```

## Command Line Client

```bash
#!/usr/bin/env python3
# qeris_cli.py

import click
import asyncio
import json
from mcp import Client

@click.group()
@click.option('--server', default='http://localhost:8765', help='QERIS server URL')
@click.pass_context
def cli(ctx, server):
    """QERIS Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['server'] = server

@cli.command()
@click.pass_context
def list_experiments(ctx):
    """List all available experiments"""
    async def _list():
        client = Client()
        await client.connect(ctx.obj['server'])
        result = await client.call_tool('list_experiments')
        for exp in result['experiments']:
            click.echo(f"{exp['name']} ({exp['category']}): {exp['description']}")
    
    asyncio.run(_list())

@cli.command()
@click.argument('experiment')
@click.argument('qubits', nargs=-1, required=True)
@click.option('--params', help='JSON parameters')
@click.pass_context
def run(ctx, experiment, qubits, params):
    """Run an experiment"""
    async def _run():
        client = Client()
        await client.connect(ctx.obj['server'])
        
        parameters = json.loads(params) if params else {}
        
        exp_id = await client.call_tool('run_experiment', {
            'experiment_name': experiment,
            'qubits': list(qubits),
            'parameters': parameters
        })
        
        click.echo(f"Started experiment: {exp_id}")
        
        # Monitor progress
        with click.progressbar(length=100, label='Running experiment') as bar:
            last_progress = 0
            while True:
                status = await client.call_tool('get_status')
                progress = int(status['experiment']['progress'] * 100)
                bar.update(progress - last_progress)
                last_progress = progress
                
                if status['experiment']['state'] == 'completed':
                    break
                await asyncio.sleep(0.5)
        
        # Get results
        results = await client.call_tool('get_results', {
            'experiment_id': exp_id,
            'format': 'processed'
        })
        click.echo(json.dumps(results, indent=2))
    
    asyncio.run(_run())

@cli.command()
@click.pass_context
def status(ctx):
    """Show current experiment status"""
    async def _status():
        client = Client()
        await client.connect(ctx.obj['server'])
        result = await client.call_tool('get_status')
        click.echo(json.dumps(result, indent=2))
    
    asyncio.run(_status())

@cli.command()
@click.argument('reset_type', type=click.Choice(['full', 'qubits', 'instruments', 'calibrations']))
@click.pass_context
def reset(ctx, reset_type):
    """Reset hardware"""
    if click.confirm(f'Reset hardware ({reset_type})?'):
        async def _reset():
            client = Client()
            await client.connect(ctx.obj['server'])
            result = await client.call_tool('reset_hardware', {
                'reset_type': reset_type
            })
            click.echo(json.dumps(result, indent=2))
        
        asyncio.run(_reset())

if __name__ == '__main__':
    cli()
```

Usage:
```bash
# List experiments
./qeris_cli.py list-experiments

# Run Rabi experiment
./qeris_cli.py run RabiExperiment q0 --params '{"start":0,"stop":1,"step":0.05}'

# Check status
./qeris_cli.py status

# Reset hardware
./qeris_cli.py reset full
```