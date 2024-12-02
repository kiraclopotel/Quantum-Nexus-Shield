import tkinter as tk
from tkinter import ttk, filedialog
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import time
from matplotlib.ticker import FixedLocator

class VisualizationTab(ttk.Frame):
    def __init__(self, parent: ttk.Notebook, shared_state: Dict[str, Any]):
        super().__init__(parent)
        self.shared_state = shared_state
        
        # Initialize visualization components
        self.figures: Dict[str, Figure] = {}
        self.canvases: Dict[str, FigureCanvasTkAgg] = {}
        self.current_view = "combined"
        self.auto_refresh = tk.BooleanVar(value=True)
        
        self.setup_ui()

    def setup_ui(self):
        """Setup the visualization tab UI"""
        # Controls Frame
        controls_frame = ttk.Frame(self)
        controls_frame.pack(fill="x", padx=5, pady=5)
        
        # View Selection
        ttk.Label(controls_frame, text="View:").pack(side="left", padx=5)
        self.view_var = tk.StringVar(value="combined")
        views = ttk.Combobox(controls_frame, textvariable=self.view_var,
                            values=["combined", "entropy", "timeline", "layers"])
        views.pack(side="left", padx=5)
        views.bind('<<ComboboxSelected>>', self.on_view_changed)
        
        # Auto-refresh Toggle
        ttk.Checkbutton(controls_frame, text="Auto Refresh",
                        variable=self.auto_refresh).pack(side="left", padx=10)
        
        # Control Buttons
        ttk.Button(controls_frame, text="Refresh",
                   command=self.refresh).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="Save Plot",
                   command=self.save_current_plot).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="Reset View",
                   command=self.reset_view).pack(side="left", padx=5)
        
        # Create notebook for different visualizations
        self.viz_notebook = ttk.Notebook(self)
        self.viz_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create visualization tabs
        self.create_combined_view()
        self.create_entropy_view()
        self.create_timeline_view()
        self.create_layer_view()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var).pack(fill="x", padx=5)

    def create_combined_view(self):
        """Create the combined visualization view"""
        combined_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(combined_frame, text="Combined View")
        
        fig = Figure(figsize=(12, 8), dpi=100)
        self.figures['combined'] = fig
        canvas = FigureCanvasTkAgg(fig, master=combined_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvases['combined'] = canvas

    def create_entropy_view(self):
        """Create the entropy visualization view"""
        entropy_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(entropy_frame, text="Entropy Analysis")
        
        fig = Figure(figsize=(12, 8), dpi=100)
        self.figures['entropy'] = fig
        canvas = FigureCanvasTkAgg(fig, master=entropy_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvases['entropy'] = canvas

    def create_timeline_view(self):
        """Create the timeline visualization view"""
        timeline_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(timeline_frame, text="Timeline Analysis")
        
        fig = Figure(figsize=(12, 8), dpi=100)
        self.figures['timeline'] = fig
        canvas = FigureCanvasTkAgg(fig, master=timeline_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvases['timeline'] = canvas

    def create_layer_view(self):
        """Create the layer visualization view"""
        layer_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(layer_frame, text="Layer Analysis")
        
        fig = Figure(figsize=(12, 8), dpi=100)
        self.figures['layers'] = fig
        canvas = FigureCanvasTkAgg(fig, master=layer_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvases['layers'] = canvas

    def update_combined_view(self):
        """Update the combined visualization"""
        fig = self.figures['combined']
        fig.clear()
        gs = GridSpec(2, 2, figure=fig)
        
        encryption = self.shared_state['encryption']
        if not encryption.messages:
            self._show_no_data(fig)
            return
            
        # Entropy Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        entropy_data = [e[2] for e in encryption.encryption_data]
        ax1.plot(entropy_data, 'b-', marker='o', label='Entropy')
        ax1.set_title("Entropy Distribution")
        ax1.set_xlabel("Message Index")
        ax1.set_ylabel("Entropy")
        ax1.grid(True)
        
        # Timeline View
        ax2 = fig.add_subplot(gs[0, 1])
        timeline_data = encryption.timeline.get_visualization_data()
        if timeline_data['timestamps']:
            ax2.scatter(timeline_data['timestamps'], range(len(timeline_data['timestamps'])),
                       c=timeline_data['layers'], cmap='viridis')
            ax2.set_title("Message Timeline")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Message Index")
        
        # Layer Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        if timeline_data['layers']:
            ax3.hist(timeline_data['layers'], bins='auto', alpha=0.7)
            ax3.set_title("Layer Distribution")
            ax3.set_xlabel("Layer")
            ax3.set_ylabel("Count")
        
        # Statistical Analysis
        ax4 = fig.add_subplot(gs[1, 1])
        if len(entropy_data) > 1:
            entropy_changes = np.diff(entropy_data)
            ax4.hist(entropy_changes, bins='auto', alpha=0.7)
            ax4.set_title("Entropy Changes")
            ax4.set_xlabel("Entropy Difference")
            ax4.set_ylabel("Frequency")
        
        fig.tight_layout()
        self.canvases['combined'].draw()

    def update_entropy_view(self):
        """Update the entropy visualization"""
        fig = self.figures['entropy']
        fig.clear()
        gs = GridSpec(2, 1, figure=fig)
        
        encryption = self.shared_state['encryption']
        if not encryption.messages:
            self._show_no_data(fig)
            return
        
        entropy_data = [e[2] for e in encryption.encryption_data]
        
        # Entropy Over Time
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(entropy_data, 'b-', marker='o', label='Entropy')
        ax1.fill_between(range(len(entropy_data)), entropy_data, alpha=0.3)
        ax1.set_title("Entropy Distribution Over Messages")
        ax1.set_xlabel("Message Index")
        ax1.set_ylabel("Entropy")
        ax1.grid(True)
        
        # Entropy Heatmap
        ax2 = fig.add_subplot(gs[1])
        entropy_data_array = np.array(entropy_data[-100:])
        num_elements = len(entropy_data_array)
        matrix_size = min(10, num_elements)
        
        # Calculate the number of rows needed
        num_rows = int(np.ceil(num_elements / matrix_size))
        
        # Pad the array if necessary
        padding_size = num_rows * matrix_size - num_elements
        if padding_size > 0:
            entropy_data_array = np.pad(entropy_data_array, (0, padding_size), mode='edge')
        
        # Reshape the array
        entropy_matrix = entropy_data_array.reshape(num_rows, matrix_size)
        
        # Proceed with plotting
        im = ax2.imshow(entropy_matrix.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax2.set_title("Entropy Heatmap (Recent Messages)")
        ax2.set_xlabel("Message Group")
        ax2.set_ylabel("Position in Group")
        fig.colorbar(im, ax=ax2, label="Entropy")
        
        fig.tight_layout()
        self.canvases['entropy'].draw()


    def update_timeline_view(self):
        fig = self.figures['timeline']
        fig.clear()
        gs = GridSpec(2, 1, figure=fig)

        timeline = self.shared_state['encryption'].timeline
        timeline_data = timeline.get_visualization_data()
    
        if not timeline_data['timestamps']:
            self._show_no_data(fig)
            return
    
        # Message Timeline Visualization
        ax1 = fig.add_subplot(gs[0])
        scatter = ax1.scatter(
            timeline_data['timestamps'], 
            range(len(timeline_data['timestamps'])),
            c=timeline_data['layers'], 
            cmap='viridis',
            s=100
        )
        ax1.set_title("Message Timeline")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Message Index")
        fig.colorbar(scatter, ax=ax1, label="Layer")
    
        # Timeline Metrics Visualization
        ax2 = fig.add_subplot(gs[1])
        metrics = timeline.get_layer_statistics()
        if metrics:
            ax2.bar(metrics.keys(), metrics.values(), alpha=0.7)
            ax2.set_title("Timeline Metrics")
        
            ax2.set_xticks(range(len(metrics.keys())))
            ax2.set_xticklabels(metrics.keys(), rotation=45)
            ax2.grid(True)
    
        fig.tight_layout()
        self.canvases['timeline'].draw()

    def update_layer_view(self):
        """Update the size visualization instead of layer"""
        fig = self.figures['layers']
        fig.clear()
        gs = GridSpec(2, 2, figure=fig)
    
        timeline = self.shared_state['encryption'].timeline
        timeline_data = timeline.get_visualization_data()
    
        if not timeline_data['layers']:
            self._show_no_data(fig)
            return
    
        # Size Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(timeline_data['layers'], bins='auto', alpha=0.7)
        ax1.set_title("Size Distribution")
        ax1.set_xlabel("Size")
        ax1.set_ylabel("Count")
    
        # Size Transitions (if applicable)
        ax2 = fig.add_subplot(gs[0, 1])
        size_transitions = np.diff(timeline_data['layers'])
        ax2.plot(size_transitions, marker='o')
        ax2.set_title("Size Transitions")
        ax2.set_xlabel("Message Index")
        ax2.set_ylabel("Size Change")
    
        fig.tight_layout()
        self.canvases['layers'].draw()

    def _show_no_data(self, fig: Figure):
        """Show no data message on empty plot"""
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No data available",
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        fig.tight_layout()

    def refresh(self):
        """Refresh all visualizations"""
        start_time = time.time()
        self.update_status("Refreshing visualizations...")
        
        self.update_combined_view()
        self.update_entropy_view()
        self.update_timeline_view()
        self.update_layer_view()
        
        end_time = time.time()
        self.update_status(f"Refresh completed in {end_time - start_time:.2f}s")

    def on_view_changed(self, event):
        """Handle view selection change"""
        view = self.view_var.get()
        self.current_view = view
        self.viz_notebook.select(["combined", "entropy", "timeline", "layers"].index(view))
        self.refresh()

    def save_current_plot(self):
        """Save current plot to file"""
        view = self.current_view
        if view not in self.figures:
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=f"quantum_stack_{view}_visualization"
        )
        
        if filename:
            self.figures[view].savefig(filename, dpi=300, bbox_inches='tight')
            self.update_status(f"Plot saved to {filename}")

    def reset_view(self):
        """Reset current view to default state"""
        view = self.current_view
        if view in self.figures:
            self.figures[view].clear()
            self.canvases[view].draw()
        self.refresh()

    def update_status(self, message: str):
        """Update status bar message"""
        self.status_var.set(message)
        self.update()

    def clear(self):
        """Clear all visualizations"""
        for fig in self.figures.values():
            fig.clear()
        for canvas in self.canvases.values():
            canvas.draw()
        self.update_status("Visualizations cleared")