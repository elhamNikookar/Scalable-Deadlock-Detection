"""
Advanced Visualization and Analysis Tools
For dl² methodology applied to 100 philosophers problem
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
from typing import List, Dict, Tuple, Any
import json
from pathlib import Path


class DL2VisualizationSuite:
    """
    Comprehensive visualization suite for dl² methodology
    """
    
    def __init__(self, system_data: Dict[str, Any]):
        self.system_data = system_data
        self.figures = {}
        
    def create_deadlock_detection_dashboard(self, save_path: str = None):
        """Create comprehensive deadlock detection dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Communication Graph', 'Deadlock Events Timeline', 
                          'Operation Types Distribution', 'Process States',
                          'Communication Patterns', 'Performance Metrics'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "indicator"}]]
        )
        
        # 1. Communication Graph (simplified)
        if 'communication_graph' in self.system_data:
            graph_data = self.system_data['communication_graph']
            # Add nodes
            for node_id, node_info in graph_data.get('nodes', {}).items():
                fig.add_trace(
                    go.Scatter(x=[node_info.get('x', 0)], y=[node_info.get('y', 0)],
                              mode='markers+text', text=[node_id], name='Nodes'),
                    row=1, col=1
                )
        
        # 2. Deadlock Events Timeline
        if 'deadlock_events' in self.system_data:
            events = self.system_data['deadlock_events']
            timestamps = [event[0] for event in events]
            fig.add_trace(
                go.Scatter(x=timestamps, y=[1] * len(timestamps),
                          mode='markers', name='Deadlock Events', marker=dict(color='red')),
                row=1, col=2
            )
        
        # 3. Operation Types Distribution
        if 'operation_types' in self.system_data:
            op_types = self.system_data['operation_types']
            fig.add_trace(
                go.Bar(x=list(op_types.keys()), y=list(op_types.values()),
                      name='Operation Types'),
                row=2, col=1
            )
        
        # 4. Process States
        if 'process_states' in self.system_data:
            states = self.system_data['process_states']
            fig.add_trace(
                go.Bar(x=list(states.keys()), y=list(states.values()),
                      name='Process States'),
                row=2, col=2
            )
        
        # 5. Communication Patterns Heatmap
        if 'communication_patterns' in self.system_data:
            patterns = self.system_data['communication_patterns']
            # Create heatmap data
            heatmap_data = np.random.rand(10, 10)  # Placeholder
            fig.add_trace(
                go.Heatmap(z=heatmap_data, name='Communication Patterns'),
                row=3, col=1
            )
        
        # 6. Performance Metrics
        if 'performance_metrics' in self.system_data:
            metrics = self.system_data['performance_metrics']
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=metrics.get('accuracy', 0.85),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Detection Accuracy"},
                    delta={'reference': 0.8},
                    gauge={'axis': {'range': [None, 1]},
                          'bar': {'color': "darkblue"},
                          'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                   {'range': [0.5, 0.8], 'color': "gray"}],
                          'threshold': {'line': {'color': "red", 'width': 4},
                                      'thickness': 0.75, 'value': 0.9}}
                ),
                row=3, col=2
            )
        
        fig.update_layout(height=1200, title_text="dl² Deadlock Detection Dashboard")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_philosophers_network_visualization(self, philosophers_data: Dict[str, Any], save_path: str = None):
        """Create interactive philosophers network visualization"""
        fig = go.Figure()
        
        # Extract philosopher positions and states
        positions = philosophers_data.get('positions', {})
        states = philosophers_data.get('states', {})
        communications = philosophers_data.get('communications', [])
        
        # Color mapping for states
        state_colors = {
            'thinking': 'blue',
            'hungry': 'orange', 
            'eating': 'green',
            'waiting': 'red',
            'communicating': 'purple'
        }
        
        # Add philosopher nodes
        for phil_id, pos in positions.items():
            state = states.get(phil_id, 'thinking')
            color = state_colors.get(state, 'gray')
            
            fig.add_trace(go.Scatter(
                x=[pos[0]], y=[pos[1]],
                mode='markers+text',
                marker=dict(size=20, color=color, opacity=0.8),
                text=[f'P{phil_id}'],
                textposition="middle center",
                name=f'Philosopher {phil_id}',
                hovertemplate=f'<b>Philosopher {phil_id}</b><br>State: {state}<extra></extra>'
            ))
        
        # Add communication edges
        for comm in communications:
            source_pos = positions.get(comm['source'], [0, 0])
            target_pos = positions.get(comm['target'], [0, 0])
            
            fig.add_trace(go.Scatter(
                x=[source_pos[0], target_pos[0]], 
                y=[source_pos[1], target_pos[1]],
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                opacity=0.5,
                showlegend=False,
                hovertemplate=f"Communication: {comm.get('type', 'unknown')}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Interactive Philosophers Communication Network",
            xaxis=dict(showgrid=True, zeroline=False),
            yaxis=dict(showgrid=True, zeroline=False),
            showlegend=True,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_performance_analysis_chart(self, performance_data: Dict[str, Any], save_path: str = None):
        """Create performance analysis chart"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Detection Time vs Problem Size', 'Accuracy Comparison',
                          'Communication Overhead', 'Deadlock Resolution Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Detection Time vs Problem Size
        problem_sizes = performance_data.get('problem_sizes', [10, 20, 50, 100])
        detection_times = performance_data.get('detection_times', [0.1, 0.3, 0.8, 2.1])
        
        fig.add_trace(
            go.Scatter(x=problem_sizes, y=detection_times, mode='lines+markers',
                      name='Detection Time', line=dict(color='blue')),
            row=1, col=1
        )
        
        # 2. Accuracy Comparison
        methods = performance_data.get('methods', ['dl²', 'Cycle Detection', 'Resource Analysis'])
        accuracies = performance_data.get('accuracies', [0.95, 0.88, 0.82])
        
        fig.add_trace(
            go.Bar(x=methods, y=accuracies, name='Accuracy', marker_color='green'),
            row=1, col=2
        )
        
        # 3. Communication Overhead
        overhead_data = performance_data.get('communication_overhead', np.random.rand(20))
        fig.add_trace(
            go.Scatter(y=overhead_data, mode='lines', name='Communication Overhead',
                      line=dict(color='orange')),
            row=2, col=1
        )
        
        # 4. Deadlock Resolution Time
        resolution_times = performance_data.get('resolution_times', [0.05, 0.1, 0.15, 0.2])
        fig.add_trace(
            go.Scatter(x=problem_sizes, y=resolution_times, mode='lines+markers',
                      name='Resolution Time', line=dict(color='red')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Performance Analysis")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_communication_pattern_analysis(self, pattern_data: Dict[str, Any], save_path: str = None):
        """Create communication pattern analysis visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Communication Types', 'Pattern Frequency',
                          'Deadlock Risk by Pattern', 'Temporal Patterns'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Communication Types Pie Chart
        comm_types = pattern_data.get('communication_types', {})
        fig.add_trace(
            go.Pie(labels=list(comm_types.keys()), values=list(comm_types.values()),
                  name="Communication Types"),
            row=1, col=1
        )
        
        # 2. Pattern Frequency
        patterns = pattern_data.get('patterns', {})
        fig.add_trace(
            go.Bar(x=list(patterns.keys()), y=list(patterns.values()),
                  name='Pattern Frequency', marker_color='lightblue'),
            row=1, col=2
        )
        
        # 3. Deadlock Risk by Pattern
        risk_data = pattern_data.get('deadlock_risk', {})
        fig.add_trace(
            go.Bar(x=list(risk_data.keys()), y=list(risk_data.values()),
                  name='Deadlock Risk', marker_color='red'),
            row=2, col=1
        )
        
        # 4. Temporal Patterns
        temporal_data = pattern_data.get('temporal_patterns', [])
        if temporal_data:
            timestamps = [item['timestamp'] for item in temporal_data]
            counts = [item['count'] for item in temporal_data]
            fig.add_trace(
                go.Scatter(x=timestamps, y=counts, mode='lines+markers',
                          name='Temporal Patterns', line=dict(color='purple')),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Communication Pattern Analysis")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


class DL2AnalysisTools:
    """
    Analysis tools for dl² methodology
    """
    
    def __init__(self, system_data: Dict[str, Any]):
        self.system_data = system_data
        self.analysis_results = {}
    
    def analyze_deadlock_patterns(self) -> Dict[str, Any]:
        """Analyze deadlock patterns in the system"""
        analysis = {
            'total_deadlocks': 0,
            'deadlock_frequency': 0.0,
            'common_patterns': [],
            'risk_factors': [],
            'resolution_success_rate': 0.0
        }
        
        # Analyze deadlock events
        if 'deadlock_events' in self.system_data:
            events = self.system_data['deadlock_events']
            analysis['total_deadlocks'] = len(events)
            
            if events:
                # Calculate frequency
                time_span = events[-1][0] - events[0][0] if len(events) > 1 else 1.0
                analysis['deadlock_frequency'] = len(events) / time_span
        
        # Analyze communication patterns
        if 'communication_patterns' in self.system_data:
            patterns = self.system_data['communication_patterns']
            # Find common patterns
            pattern_counts = {}
            for pattern in patterns:
                pattern_key = f"{pattern.get('type', 'unknown')}_{len(pattern.get('participants', []))}"
                pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1
            
            analysis['common_patterns'] = sorted(pattern_counts.items(), 
                                               key=lambda x: x[1], reverse=True)[:5]
        
        return analysis
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        metrics = {
            'detection_accuracy': 0.0,
            'false_positive_rate': 0.0,
            'false_negative_rate': 0.0,
            'average_detection_time': 0.0,
            'throughput': 0.0
        }
        
        # Calculate detection accuracy
        if 'detection_results' in self.system_data:
            results = self.system_data['detection_results']
            true_positives = results.get('true_positives', 0)
            false_positives = results.get('false_positives', 0)
            true_negatives = results.get('true_negatives', 0)
            false_negatives = results.get('false_negatives', 0)
            
            total = true_positives + false_positives + true_negatives + false_negatives
            if total > 0:
                metrics['detection_accuracy'] = (true_positives + true_negatives) / total
                metrics['false_positive_rate'] = false_positives / (false_positives + true_negatives)
                metrics['false_negative_rate'] = false_negatives / (false_negatives + true_positives)
        
        # Calculate average detection time
        if 'detection_times' in self.system_data:
            times = self.system_data['detection_times']
            metrics['average_detection_time'] = np.mean(times) if times else 0.0
        
        return metrics
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Analyze deadlock patterns
        deadlock_analysis = self.analyze_deadlock_patterns()
        
        if deadlock_analysis['deadlock_frequency'] > 0.1:
            recommendations.append("High deadlock frequency detected. Consider implementing timeout mechanisms.")
        
        if deadlock_analysis['total_deadlocks'] > 10:
            recommendations.append("Multiple deadlocks detected. Review communication patterns and dependencies.")
        
        # Analyze performance metrics
        performance_metrics = self.calculate_performance_metrics()
        
        if performance_metrics['detection_accuracy'] < 0.9:
            recommendations.append("Detection accuracy below threshold. Consider tuning detection parameters.")
        
        if performance_metrics['false_positive_rate'] > 0.1:
            recommendations.append("High false positive rate. Review detection criteria.")
        
        # Analyze communication patterns
        if 'communication_patterns' in self.system_data:
            patterns = self.system_data['communication_patterns']
            if len(patterns) > 100:
                recommendations.append("High communication complexity. Consider simplifying communication patterns.")
        
        return recommendations
    
    def export_analysis_report(self, output_path: str):
        """Export comprehensive analysis report"""
        report = {
            'timestamp': time.time(),
            'system_info': {
                'total_philosophers': self.system_data.get('num_philosophers', 100),
                'total_operations': len(self.system_data.get('operations', [])),
                'simulation_duration': self.system_data.get('simulation_duration', 0)
            },
            'deadlock_analysis': self.analyze_deadlock_patterns(),
            'performance_metrics': self.calculate_performance_metrics(),
            'recommendations': self.generate_recommendations(),
            'raw_data': self.system_data
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Analysis report exported to {output_path}")


def create_comprehensive_visualization(system_data: Dict[str, Any], output_dir: str = "geo/100phill"):
    """Create comprehensive visualization suite"""
    print("Creating comprehensive visualization suite...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize visualization suite
    viz_suite = DL2VisualizationSuite(system_data)
    
    # Create dashboards
    print("Creating deadlock detection dashboard...")
    dashboard = viz_suite.create_deadlock_detection_dashboard(
        f"{output_dir}/deadlock_detection_dashboard.html"
    )
    
    print("Creating philosophers network visualization...")
    network_viz = viz_suite.create_philosophers_network_visualization(
        system_data.get('philosophers_data', {}),
        f"{output_dir}/philosophers_network.html"
    )
    
    print("Creating performance analysis chart...")
    performance_chart = viz_suite.create_performance_analysis_chart(
        system_data.get('performance_data', {}),
        f"{output_dir}/performance_analysis.html"
    )
    
    print("Creating communication pattern analysis...")
    pattern_analysis = viz_suite.create_communication_pattern_analysis(
        system_data.get('pattern_data', {}),
        f"{output_dir}/communication_patterns.html"
    )
    
    # Initialize analysis tools
    analysis_tools = DL2AnalysisTools(system_data)
    
    print("Generating analysis report...")
    analysis_tools.export_analysis_report(f"{output_dir}/analysis_report.json")
    
    print("Visualization suite completed!")
    return viz_suite, analysis_tools


def create_sample_system_data() -> Dict[str, Any]:
    """Create sample system data for visualization"""
    return {
        'num_philosophers': 100,
        'simulation_duration': 30.0,
        'deadlock_events': [
            (time.time() - 20, [], {}),
            (time.time() - 15, [], {}),
            (time.time() - 10, [], {}),
            (time.time() - 5, [], {})
        ],
        'communication_patterns': [
            {'type': 'broadcast', 'participants': [0, 1, 2], 'timestamp': time.time() - 25},
            {'type': 'send', 'participants': [3, 4], 'timestamp': time.time() - 20},
            {'type': 'recv', 'participants': [5, 6], 'timestamp': time.time() - 15}
        ],
        'philosophers_data': {
            'positions': {i: [np.cos(2*np.pi*i/100), np.sin(2*np.pi*i/100)] for i in range(100)},
            'states': {i: ['thinking', 'hungry', 'eating', 'waiting'][i % 4] for i in range(100)},
            'communications': [
                {'source': i, 'target': (i+1) % 100, 'type': 'coordination'} 
                for i in range(0, 100, 10)
            ]
        },
        'performance_data': {
            'problem_sizes': [10, 20, 50, 100],
            'detection_times': [0.1, 0.3, 0.8, 2.1],
            'methods': ['dl²', 'Cycle Detection', 'Resource Analysis'],
            'accuracies': [0.95, 0.88, 0.82],
            'communication_overhead': np.random.rand(20),
            'resolution_times': [0.05, 0.1, 0.15, 0.2]
        },
        'pattern_data': {
            'communication_types': {'broadcast': 45, 'send': 30, 'recv': 25},
            'patterns': {'pattern1': 20, 'pattern2': 15, 'pattern3': 10},
            'deadlock_risk': {'pattern1': 0.8, 'pattern2': 0.6, 'pattern3': 0.4},
            'temporal_patterns': [
                {'timestamp': time.time() - i*5, 'count': np.random.randint(1, 10)}
                for i in range(20)
            ]
        },
        'detection_results': {
            'true_positives': 8,
            'false_positives': 2,
            'true_negatives': 85,
            'false_negatives': 5
        },
        'detection_times': [0.1, 0.2, 0.15, 0.3, 0.25]
    }


if __name__ == "__main__":
    # Create sample data and run visualization
    sample_data = create_sample_system_data()
    viz_suite, analysis_tools = create_comprehensive_visualization(sample_data)
    
    print("Visualization suite created successfully!")
    print("Check the geo/100phill/ directory for generated visualizations.")
