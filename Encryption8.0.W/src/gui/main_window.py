# main_window.py

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any
import sys
from pathlib import Path

# Append parent directory to sys.path to import modules correctly
sys.path.append(str(Path(__file__).parent.parent))

from core.encryption import QuantumStackEncryption
from core.timeline import TimelineManager
from core.mathematics import MathematicalCore
from utils.helpers import ConfigManager, MetricsCollector
from gui.encryption_tab import EncryptionTab
from gui.visualization_tab import VisualizationTab
from gui.tests_tab import TestsTab
from gui.decryption_tab import DecryptionTab
from gui.compression_tabs import CompressedEncryptionTab, CompressedDecryptionTab

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Initialize core components
        self.encryption = QuantumStackEncryption()
        self.timeline_manager = TimelineManager()
        self.math_core = MathematicalCore()
        self.config_manager = ConfigManager()
        self.metrics_collector = MetricsCollector()
        
        # Setup main window
        self.title("Quantum Stack Encryption")
        self.geometry("1200x900")
        
        # Create shared state dictionary
        self.shared_state: Dict[str, Any] = {
            'encryption': self.encryption,
            'timeline': self.timeline_manager,
            'math_core': self.math_core,
            'metrics': self.metrics_collector,
            'config': self.config_manager,
            'pattern_storage': {},
            'encrypted_storage': {}  # Added encrypted storage
        }
        
        # Setup UI after shared state is initialized
        self.setup_ui()

    def setup_ui(self):
        """Setup the main user interface"""
        # Create main menu
        self.create_menu()
    
        # Create main container
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)
    
        # Create notebook for tabs
        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
    
        # Create tabs with shared state
        self.encryption_tab = EncryptionTab(self.notebook, self.shared_state)
        self.decryption_tab = DecryptionTab(self.notebook, self.shared_state)
        self.visualization_tab = VisualizationTab(self.notebook, self.shared_state)
        self.tests_tab = TestsTab(self.notebook, self.shared_state)
        self.compressed_encryption_tab = CompressedEncryptionTab(self.notebook, self.shared_state)
        self.compressed_decryption_tab = CompressedDecryptionTab(self.notebook, self.shared_state)
    
        # Add tabs to notebook in the desired order
        self.notebook.add(self.encryption_tab, text="Encryption")
        self.notebook.add(self.decryption_tab, text="Decryption")
        self.notebook.add(self.visualization_tab, text="Visualization")
        self.notebook.add(self.tests_tab, text="Tests")
        self.notebook.add(self.compressed_encryption_tab, text="Compressed Encryption")
        self.notebook.add(self.compressed_decryption_tab, text="Compressed Decryption")
        
        # Create status bar
        self.status_bar = ttk.Label(container, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
        # Bind tab change event
        self.notebook.bind('<<NotebookTabChanged>>', self.on_tab_changed)

    def create_menu(self):
        """Create the main menu bar"""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Session", command=self.new_session)
        file_menu.add_command(label="Save State", command=self.save_state)
        file_menu.add_command(label="Load State", command=self.load_state)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Clear All", command=self.clear_all)
        tools_menu.add_command(label="Reset Statistics", command=self.reset_stats)
        tools_menu.add_command(label="Export Metrics", command=self.export_metrics)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_docs)
        help_menu.add_command(label="About", command=self.show_about)

    def on_tab_changed(self, event):
        """Handle tab change events"""
        current_tab = self.notebook.select()
        tab_text = self.notebook.tab(current_tab, "text")
        
        # Update status bar
        self.status_bar.config(text=f"Current View: {tab_text}")
        
        # Refresh current tab
        if tab_text == "Visualization":
            self.visualization_tab.refresh()
        elif tab_text == "Tests":
            self.tests_tab.refresh()

    def new_session(self):
        """Start a new session"""
        if messagebox.askyesno("New Session", "Start a new session? This will clear all current data."):
            self.clear_all()

    def save_state(self):
        """Save current state to file"""
        self.config_manager.save_config()
        self.status_bar.config(text="State saved")

    def load_state(self):
        """Load state from file"""
        if self.config_manager.load_config():
            self.status_bar.config(text="State loaded")
        else:
            self.status_bar.config(text="Error loading state")

    def clear_all(self):
        """Clear all data and reset"""
        self.encryption = QuantumStackEncryption()
        self.timeline_manager = TimelineManager()
        self.metrics_collector.clear_metrics()
        
        # Update shared state
        self.shared_state.update({
            'encryption': self.encryption,
            'timeline': self.timeline_manager,
            'pattern_storage': {},
            'encrypted_storage': {}
        })
        
        # Reset all tabs
        self.encryption_tab.clear()
        self.decryption_tab.clear()
        self.visualization_tab.clear()
        self.tests_tab.clear()
        self.compressed_encryption_tab.clear_all()
        self.compressed_decryption_tab.clear()
        
        self.status_bar.config(text="All data cleared")

    def reset_stats(self):
        """Reset statistics and metrics"""
        self.metrics_collector.clear_metrics()
        self.visualization_tab.refresh()
        self.status_bar.config(text="Statistics reset")

    def export_metrics(self):
        """Export collected metrics"""
        stats = self.metrics_collector.get_statistics()
        if stats:
            path = Path("metrics_export.json")
            self.metrics_collector.export_to_file(path)
            self.status_bar.config(text=f"Metrics exported to {path}")

    def show_docs(self):
        """Show documentation"""
        messagebox.showinfo("Documentation", 
            "Please visit the documentation for complete information.")

    def show_about(self):
        """Show about dialog"""
        about_text = """Quantum Stack Encryption
Version 2.0

An advanced encryption system using quantum-inspired algorithms
and mathematical optimization."""
        
        messagebox.showinfo("About", about_text)

    def run(self):
        """Start the application"""
        self.mainloop()

if __name__ == "__main__":
    app = MainWindow()
    app.run()
