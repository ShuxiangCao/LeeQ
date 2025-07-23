# Real-time Monitoring

## Overview

QERIS uses MCP resources to provide real-time data streaming without requiring WebSocket implementation. This enables live monitoring of experiments, parameter updates, and system status.

## MCP Resources

### Live Data Resource

Streams real-time experiment data:

```python
@self.mcp_server.resource("qeris://live_data")
async def live_data_resource() -> AsyncGenerator[Resource, None]:
    """Stream live experiment data"""
    while True:
        data = await self.adapter.get_live_data()
        yield Resource(
            uri="qeris://live_data",
            name="Live Experiment Data",
            description="Real-time data from running experiment",
            mimeType="application/json",
            text=json.dumps(data)
        )
        await asyncio.sleep(0.5)  # Update interval
```

### Status Resource

Provides experiment status updates:

```python
@self.mcp_server.resource("qeris://experiment_status")
async def status_resource() -> AsyncGenerator[Resource, None]:
    """Stream experiment status updates"""
    while True:
        status = await self.adapter.get_status()
        yield Resource(
            uri="qeris://experiment_status",
            name="Experiment Status",
            description="Current experiment status and progress",
            mimeType="application/json",
            text=json.dumps(status)
        )
        await asyncio.sleep(1.0)  # Status update interval
```

## Client Implementation

### Python Client with Live Plotting

```python
import asyncio
import json
import matplotlib.pyplot as plt
from mcp import Client
import numpy as np

class LiveExperimentMonitor:
    def __init__(self):
        self.client = Client()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'b-')
        self.data_x = []
        self.data_y = []
        
    async def connect(self, server_url='http://localhost:8765'):
        await self.client.connect(server_url)
        
    async def monitor_experiment(self):
        """Monitor running experiment with live plot"""
        plt.ion()  # Interactive mode
        self.fig.show()
        
        # Subscribe to live data
        async for resource in self.client.read_resource('qeris://live_data'):
            data = json.loads(resource.text)
            
            if data['type'] == 'data_point':
                # Append new data
                self.data_x.extend(data['data']['x'])
                self.data_y.extend(data['data']['y'])
                
                # Update plot
                self.line.set_data(self.data_x, self.data_y)
                self.ax.relim()
                self.ax.autoscale_view()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            
            elif data['type'] == 'status_update':
                print(f"Status: {data['message']}")
            
            elif data['type'] == 'error':
                print(f"Error: {data['error']}")
                break
    
    async def run_with_monitoring(self, experiment_name, qubits, parameters):
        """Run experiment and monitor simultaneously"""
        # Start monitoring task
        monitor_task = asyncio.create_task(self.monitor_experiment())
        
        # Run experiment
        exp_id = await self.client.call_tool('run_experiment', {
            'experiment_name': experiment_name,
            'qubits': qubits,
            'parameters': parameters
        })
        
        print(f"Started experiment: {exp_id}")
        
        # Wait for completion
        async for resource in self.client.read_resource('qeris://experiment_status'):
            status = json.loads(resource.text)
            if status['experiment']['state'] == 'completed':
                print("Experiment completed!")
                monitor_task.cancel()
                break
        
        # Get final results
        results = await self.client.call_tool('get_results', {
            'experiment_id': exp_id,
            'format': 'processed'
        })
        
        return results

# Usage
async def main():
    monitor = LiveExperimentMonitor()
    await monitor.connect()
    
    results = await monitor.run_with_monitoring(
        'RabiExperiment',
        ['q0'],
        {'start': 0, 'stop': 1, 'step': 0.05, 'shots': 1024}
    )
    
    print(f"Rabi frequency: {results['fit_params']['frequency']} MHz")

asyncio.run(main())
```

### Web-based Monitoring

```javascript
// Real-time monitoring in browser
class QERISMonitor {
    constructor(serverUrl = 'http://localhost:8765') {
        this.serverUrl = serverUrl;
        this.plotData = { x: [], y: [] };
        this.subscriptions = {};
    }
    
    subscribeToLiveData() {
        const eventSource = new EventSource(
            `${this.serverUrl}/resources/subscribe?uri=qeris://live_data`
        );
        
        eventSource.onmessage = (event) => {
            const resource = JSON.parse(event.data);
            const data = JSON.parse(resource.text);
            
            if (data.type === 'data_point') {
                this.updatePlot(data.data);
            }
        };
        
        this.subscriptions.liveData = eventSource;
    }
    
    subscribeToStatus() {
        const eventSource = new EventSource(
            `${this.serverUrl}/resources/subscribe?uri=qeris://experiment_status`
        );
        
        eventSource.onmessage = (event) => {
            const resource = JSON.parse(event.data);
            const status = JSON.parse(resource.text);
            this.updateStatus(status);
        };
        
        this.subscriptions.status = eventSource;
    }
    
    updatePlot(newData) {
        // Append new data
        this.plotData.x = this.plotData.x.concat(newData.x);
        this.plotData.y = this.plotData.y.concat(newData.y);
        
        // Update Plotly chart
        Plotly.react('plot', [{
            x: this.plotData.x,
            y: this.plotData.y,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Experiment Data'
        }], {
            title: 'Live Experiment Data',
            xaxis: { title: newData.dimensions?.[0] || 'X' },
            yaxis: { title: newData.dimensions?.[1] || 'Y' }
        });
    }
    
    updateStatus(status) {
        // Update progress bar
        const progress = status.experiment.progress * 100;
        document.getElementById('progress').style.width = `${progress}%`;
        document.getElementById('progress-text').textContent = `${progress.toFixed(1)}%`;
        
        // Update status text
        document.getElementById('status').textContent = status.experiment.state;
        
        // Update details
        document.getElementById('details').textContent = JSON.stringify(status, null, 2);
    }
    
    async startMonitoring() {
        // Clear previous data
        this.plotData = { x: [], y: [] };
        
        // Subscribe to resources
        this.subscribeToLiveData();
        this.subscribeToStatus();
    }
    
    stopMonitoring() {
        // Close all subscriptions
        Object.values(this.subscriptions).forEach(source => source.close());
        this.subscriptions = {};
    }
}
```

## Parameter Monitoring

### Tracking Parameter Changes

```python
async def monitor_parameter_drift():
    """Monitor qubit parameters during long experiments"""
    client = Client()
    await client.connect('http://localhost:8765')
    
    parameter_history = []
    drift_threshold = 0.001  # 1 MHz for frequency
    
    async for resource in client.read_resource('qeris://qubit_parameters'):
        params = json.loads(resource.text)
        
        # Track specific parameters
        for qubit_name, qubit_data in params.items():
            if qubit_name == 'device':
                continue
                
            current_freq = qubit_data['parameters'].get('frequency', {}).get('value')
            
            if current_freq:
                parameter_history.append({
                    'timestamp': resource.timestamp,
                    'qubit': qubit_name,
                    'frequency': current_freq
                })
                
                # Check for drift
                if len(parameter_history) > 10:
                    initial_freq = parameter_history[-10]['frequency']
                    drift = abs(current_freq - initial_freq)
                    
                    if drift > drift_threshold * 1e9:  # Convert to Hz
                        print(f"WARNING: {qubit_name} frequency drift: {drift/1e6:.3f} MHz")
                        
                        # Take corrective action
                        await client.call_tool('run_experiment', {
                            'experiment_name': 'QubitSpectroscopy',
                            'qubits': [qubit_name],
                            'parameters': {
                                'center': current_freq,
                                'span': 50e6,
                                'points': 101
                            }
                        })
```

### Multi-Resource Monitoring

```python
async def comprehensive_monitoring():
    """Monitor multiple resources simultaneously"""
    client = Client()
    await client.connect('http://localhost:8765')
    
    async def monitor_data():
        async for resource in client.read_resource('qeris://live_data'):
            data = json.loads(resource.text)
            # Process live data
            yield ('data', data)
    
    async def monitor_status():
        async for resource in client.read_resource('qeris://experiment_status'):
            status = json.loads(resource.text)
            # Process status updates
            yield ('status', status)
    
    async def monitor_parameters():
        async for resource in client.read_resource('qeris://qubit_parameters'):
            params = json.loads(resource.text)
            # Process parameter updates
            yield ('params', params)
    
    # Combine all monitors
    async def merge_monitors():
        monitors = [
            monitor_data(),
            monitor_status(),
            monitor_parameters()
        ]
        
        # Create tasks for each monitor
        tasks = [asyncio.create_task(anext(m)) for m in monitors]
        
        while tasks:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            for task in done:
                try:
                    result_type, result_data = task.result()
                    
                    # Process based on type
                    if result_type == 'data':
                        print(f"New data point: {len(result_data['data']['x'])} points")
                    elif result_type == 'status':
                        print(f"Status: {result_data['experiment']['state']}")
                    elif result_type == 'params':
                        print(f"Parameters updated")
                    
                    # Schedule next iteration for this monitor
                    monitor_idx = tasks.index(task)
                    tasks[monitor_idx] = asyncio.create_task(anext(monitors[monitor_idx]))
                    
                except StopAsyncIteration:
                    # This monitor is done
                    tasks.remove(task)
    
    await merge_monitors()
```

## Advanced Monitoring Features

### Adaptive Sampling

```python
class AdaptiveMonitor:
    """Monitor that adjusts sampling based on data rate"""
    
    def __init__(self, client):
        self.client = client
        self.sample_interval = 0.5  # Start with 500ms
        self.last_data_time = time.time()
        self.data_rate_history = []
        
    async def adaptive_monitoring(self):
        while True:
            # Get current data
            data = await self.get_current_data()
            
            # Calculate data rate
            current_time = time.time()
            time_delta = current_time - self.last_data_time
            
            if data and 'points_collected' in data:
                data_rate = data['points_collected'] / time_delta
                self.data_rate_history.append(data_rate)
                
                # Adjust sampling interval
                if len(self.data_rate_history) > 5:
                    avg_rate = np.mean(self.data_rate_history[-5:])
                    
                    if avg_rate > 100:  # High data rate
                        self.sample_interval = max(0.1, self.sample_interval * 0.9)
                    elif avg_rate < 10:  # Low data rate
                        self.sample_interval = min(2.0, self.sample_interval * 1.1)
            
            self.last_data_time = current_time
            await asyncio.sleep(self.sample_interval)
```

### Event-based Monitoring

```python
async def event_based_monitoring():
    """React to specific events during monitoring"""
    client = Client()
    await client.connect('http://localhost:8765')
    
    event_handlers = {
        'experiment_started': lambda d: print(f"Experiment {d['id']} started"),
        'experiment_completed': lambda d: print(f"Experiment {d['id']} completed"),
        'error_occurred': lambda d: print(f"ERROR: {d['message']}"),
        'milestone_reached': lambda d: print(f"Milestone: {d['description']}"),
        'parameter_drift': lambda d: print(f"Drift detected: {d['parameter']} on {d['qubit']}")
    }
    
    async for resource in client.read_resource('qeris://live_data'):
        data = json.loads(resource.text)
        
        # Check for events
        if 'event' in data:
            event_type = data['event']['type']
            if event_type in event_handlers:
                event_handlers[event_type](data['event'])
```

## Best Practices

1. **Handle connection failures gracefully** - Implement reconnection logic
2. **Buffer data for performance** - Don't update UI on every data point
3. **Use appropriate update intervals** - Balance between responsiveness and load
4. **Implement data decimation** - For long experiments, downsample old data
5. **Monitor resource usage** - Cancel subscriptions when not needed
6. **Provide offline capability** - Cache data locally for analysis
7. **Use typed data structures** - Validate incoming data format