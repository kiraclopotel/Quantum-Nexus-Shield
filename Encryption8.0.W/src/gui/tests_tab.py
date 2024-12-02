# tests_tab.py
import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Dict, Any, List
import time
import numpy as np
import threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

class TestsTab(ttk.Frame):
    def __init__(self, parent: ttk.Notebook, shared_state: Dict[str, Any]):
        super().__init__(parent)
        self.shared_state = shared_state
        self.test_results: Dict[str, bool] = {}
        self.current_test = None
        self.test_running = False
        self.setup_ui()

    def setup_ui(self):
        """Setup the tests tab UI"""
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=5, pady=5)

        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=1)

        selection_frame = ttk.LabelFrame(left_panel, text="Test Selection", padding=10)
        selection_frame.pack(fill="x", padx=5, pady=5)

        self.select_all_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(selection_frame, text="Select/Deselect All",
                        variable=self.select_all_var,
                        command=self.toggle_all_tests).pack(fill="x")

        self.test_vars = {}
        test_frame = ttk.Frame(selection_frame)
        test_frame.pack(fill="x", pady=5)

        tests = [
            ("Key Generation", "Tests key generation uniqueness and strength"),
            ("Entropy Analysis", "Validates entropy calculation and distribution"),
            ("Timeline Verification", "Checks timeline integrity and continuity"),
            ("Layer Functions", "Tests layer computation and transitions"),
            ("Mathematical Properties", "Validates mathematical foundations"),
            ("Encryption/Decryption", "Tests core encryption functionality"),
            ("Performance Metrics", "Measures operational performance"),
            ("Randomness Tests", "Statistical tests for randomness"),
            ("Seed Uniqueness", "Verifies unique seed generation"),
            ("Hash Integrity", "Tests hash generation and verification")
        ]

        for test_name, tooltip in tests:
            var = tk.BooleanVar(value=True)
            self.test_vars[test_name] = var
            cb = ttk.Checkbutton(test_frame, text=test_name, variable=var)
            cb.pack(fill="x")
            self.create_tooltip(cb, tooltip)

        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(control_frame, text="Run Selected Tests",
                   command=self.run_selected_tests).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Stop Tests",
                   command=self.stop_tests).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Clear Results",
                   command=self.clear_results).pack(side="left", padx=5)

        progress_frame = ttk.Frame(left_panel)
        progress_frame.pack(fill="x", padx=5, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var).pack()

        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=2)

        self.results_notebook = ttk.Notebook(right_panel)
        self.results_notebook.pack(fill="both", expand=True)

        text_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(text_frame, text="Test Results")

        self.results_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD)
        self.results_text.pack(fill="both", expand=True)

        viz_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(viz_frame, text="Visualization")

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_tooltip(self, widget, text):
        """Create tooltip for test descriptions"""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = ttk.Label(tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1)
            label.pack()
            def hide_tooltip(event=None):
                tooltip.destroy()
            tooltip.bind("<Leave>", hide_tooltip)
            widget.bind("<Leave>", hide_tooltip)
        widget.bind("<Enter>", show_tooltip)

    def toggle_all_tests(self):
        """Toggle all test checkboxes"""
        state = self.select_all_var.get()
        for var in self.test_vars.values():
            var.set(state)

    def run_selected_tests(self):
        """Run selected tests in a separate thread"""
        if self.test_running:
            return
        selected_tests = [name for name, var in self.test_vars.items() if var.get()]
        if not selected_tests:
            self.update_status("No tests selected")
            return
        self.test_running = True
        self.progress_var.set(0)
        self.clear_results()
        self.update_status("Running tests...")
        threading.Thread(target=self.run_tests, args=(selected_tests,), daemon=True).start()

    def run_tests(self, selected_tests: List[str]):
        """Execute selected tests"""
        total_tests = len(selected_tests)
        passed_tests = 0
        start_time = time.time()
        self.results_text.insert(tk.END, "Running Tests:\n\n")

        for i, test_name in enumerate(selected_tests):
            if not self.test_running:
                break
            self.current_test = test_name
            self.progress_var.set((i / total_tests) * 100)
            self.update_status(f"Running {test_name}...")

            method_name = f"{test_name.lower().replace(' ', '_').replace('/', '_')}_test"
            test_method = getattr(self, method_name, None)

            if test_method is None:
                self.results_text.insert(tk.END, f"Error: Test '{test_name}' not found.\n")
                continue

            success = test_method()
            self.test_results[test_name] = success
            if success:
                passed_tests += 1
            self.after(100)

        end_time = time.time()
        self.show_summary(total_tests, passed_tests, end_time - start_time)
        self.update_visualization()
        self.test_running = False
        self.current_test = None
        self.progress_var.set(100)
        self.update_status("Testing completed")

    def stop_tests(self):
        """Stop running tests"""
        self.test_running = False
        self.update_status("Tests stopped")

    def clear_results(self):
        """Clear test results"""
        self.test_results.clear()
        self.results_text.delete(1.0, tk.END)
        self.fig.clear()
        self.canvas.draw()
        self.progress_var.set(0)
        self.update_status("Results cleared")

    def show_summary(self, total: int, passed: int, duration: float):
        """Display test results summary"""
        summary = f"\nTest Summary:\n"
        summary += f"Total Tests: {total}\n"
        summary += f"Passed: {passed}\n"
        summary += f"Failed: {total - passed}\n"
        summary += f"Success Rate: {(passed/total)*100:.1f}%\n"
        summary += f"Duration: {duration:.2f} seconds\n"
        
        self.results_text.insert(tk.END, summary)
        self.results_text.see(tk.END)

    def update_visualization(self):
        """Update the test results visualization"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        if not self.test_results:
            ax.text(0.5, 0.5, "No test results available",
                   ha='center', va='center')
            ax.set_axis_off()
        else:
            # Prepare data
            tests = list(self.test_results.keys())
            results = [1 if result else 0 for result in self.test_results.values()]
            
            # Create bars
            bars = ax.bar(range(len(tests)), results)
            
            # Customize appearance
            ax.set_xticks(range(len(tests)))
            ax.set_xticklabels(tests, rotation=45, ha='right')
            ax.set_ylim(0, 1.2)
            ax.set_title("Test Results Summary")
            
            # Color bars based on results
            for bar, result in zip(bars, results):
                bar.set_color('green' if result else 'red')
                
            self.fig.tight_layout()
        
        self.canvas.draw()

    def update_status(self, message: str):
        """Update status message"""
        self.status_var.set(message)
        self.update()

    def refresh(self):
        """Refresh test results display"""
        if self.test_results:
            self.update_visualization()

    # --- Complete Test Methods ---

    def key_generation_test(self) -> bool:
        """Test key generation functionality"""
        self.results_text.insert(tk.END, "Testing Key Generation:\n")
        success = True
        keys = set()
        for seed in range(100):
            key = self.shared_state['encryption'].generate_adaptive_key(seed, 32)
            if key in keys:
                success = False
                self.results_text.insert(tk.END, "  - Failed: Duplicate key found\n")
                break
            keys.add(key)
        self.results_text.insert(tk.END, "  - Passed: All key generation tests\n" if success else "  - Failed\n")
        return success

    def entropy_analysis_test(self) -> bool:
        """Test entropy calculation and analysis"""
        self.results_text.insert(tk.END, "Testing Entropy Analysis:\n")
        success = True
        test_messages = [b"Test1", b"Test2", b"Test3"]
        entropy_values = []
        for msg in test_messages:
            _, _, _, entropy = self.shared_state['encryption'].find_perfect_entropy_seed(msg)
            if entropy is not None:
                entropy_values.append(entropy)
                if not 0 <= entropy <= 1:
                    success = False
                    self.results_text.insert(tk.END, "  - Failed: Entropy out of bounds\n")
                    break
        if len(entropy_values) > 1 and np.std(entropy_values) > 0.1:
            success = False
            self.results_text.insert(tk.END, "  - Failed: Inconsistent entropy values\n")
        self.results_text.insert(tk.END, "  - Passed: All entropy tests\n" if success else "  - Failed\n")
        return success

    def timeline_verification_test(self) -> bool:
        """Test timeline verification"""
        self.results_text.insert(tk.END, "Testing Timeline Verification:\n")
        success = True
        test_message = b"Timeline test message"
        seed = 12345
        message_id = 0
        marker = self.shared_state['encryption'].timeline.create_marker(seed, message_id, test_message, 1.0)
        required_fields = ['seed', 'id', 'timestamp', 'entropy', 'layer', 'timeline', 'checksum']
        for field in required_fields:
            if field not in marker:
                success = False
                self.results_text.insert(tk.END, f"  - Failed: Missing {field} in marker\n")
        if not self.shared_state['encryption'].timeline.verify_marker(seed, message_id, test_message):
            success = False
            self.results_text.insert(tk.END, "  - Failed: Marker verification\n")
        self.results_text.insert(tk.END, "  - Passed: All timeline tests\n" if success else "  - Failed\n")
        return success

    def layer_functions_test(self) -> bool:
        """Test layer function calculations"""
        self.results_text.insert(tk.END, "Testing Layer Functions:\n")
        success = True
        test_values = [10, 100, 1000, 10000]
        expected_layers = [2, 3, 4, 5]
    
        for value, expected_layer in zip(test_values, expected_layers):
            current_layer = self.shared_state['encryption'].math_core.compute_layer(value)
            if current_layer != expected_layer:
                success = False
                self.results_text.insert(tk.END, f"  - Failed: Incorrect layer for value {value}. Expected {expected_layer}, got {current_layer}\n")
            else:
                self.results_text.insert(tk.END, f"  - Passed: Correct layer {current_layer} for value {value}\n")
    
        return success

    def mathematical_properties_test(self) -> bool:
        """Test mathematical properties"""
        self.results_text.insert(tk.END, "Testing Mathematical Properties:\n")
        success = True
        for n in range(1, 5):
            for k in range(1, 4):
                value = self.shared_state['encryption'].math_core.layer_function(n, k)
                if value < 0:
                    success = False
                    self.results_text.insert(tk.END, f"  - Failed: Negative layer function value for n={n}, k={k}\n")
        self.results_text.insert(tk.END, "  - Passed: All mathematical property tests\n" if success else "  - Failed\n")
        return success

    def encryption_decryption_test(self) -> bool:
        """Test encryption and decryption"""
        self.results_text.insert(tk.END, "Testing Encryption/Decryption:\n")
        success = True
        test_sizes = [16, 64, 256]
        for size in test_sizes:
            message = bytes([i % 256 for i in range(size)])
            seed = 12345
            iv, ciphertext = self.shared_state['encryption'].encrypt_with_seed(message, seed)
            decrypted = self.shared_state['encryption'].decrypt_with_seed(ciphertext, seed, iv)
            if message != decrypted:
                success = False
                self.results_text.insert(tk.END, f"  - Failed: Encryption/decryption mismatch for size {size}\n")
        self.results_text.insert(tk.END, "  - Passed: All encryption/decryption tests\n" if success else "  - Failed\n")
        return success

    def performance_metrics_test(self) -> bool:
        """Test performance metrics"""
        self.results_text.insert(tk.END, "Testing Performance Metrics:\n")
        success = True
        test_sizes = [16, 64, 256]
        times = {'encrypt': [], 'decrypt': []}
        used_seeds = set()
        for size in test_sizes:
            message = bytes([i % 256 for i in range(size)])
            seed_data = self.shared_state['encryption'].find_perfect_entropy_seed(message)
            if seed_data is None:
                success = False
                self.results_text.insert(tk.END, f"  - Failed: Could not find valid seed for size {size}\n")
                continue
            seed = seed_data[0]
            if seed in used_seeds:
                success = False
                self.results_text.insert(tk.END, f"  - Failed: Duplicate seed generated\n")
                continue
            used_seeds.add(seed)
            start = time.time()
            iv, ciphertext = self.shared_state['encryption'].encrypt_with_seed(message, seed)
            times['encrypt'].append(time.time() - start)
            start = time.time()
            self.shared_state['encryption'].decrypt_with_seed(ciphertext, seed, iv)
            times['decrypt'].append(time.time() - start)
        for operation, measurements in times.items():
            avg_time = np.mean(measurements)
            std_time = np.std(measurements)
            self.results_text.insert(tk.END, f"  - {operation.capitalize()}: avg={avg_time:.6f}s, std={std_time:.6f}s\n")
            if avg_time > 1.0:
                success = False
                self.results_text.insert(tk.END, f"  - Failed: {operation} operation too slow\n")
        self.results_text.insert(tk.END, "  - Passed: All performance tests\n" if success else "  - Failed\n")
        return success

    def randomness_tests_test(self) -> bool:
        """Perform statistical randomness tests"""
        self.results_text.insert(tk.END, "Running Statistical Tests:\n")
        success = True
        message = b"Statistical test message"
        seed = random.randint(1, 2**32)
        iv, ciphertext = self.shared_state['encryption'].encrypt_with_seed(message, seed)
        bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
        monobit = self.shared_state['encryption'].monobit_test(bits)
        runs = self.shared_state['encryption'].runs_test(bits)
        chi_squared = self.shared_state['encryption'].chi_squared_test(bits)
        avalanche = self.shared_state['encryption'].avalanche_test(message, seed)
        if monobit < 0.01:
            success = False
            self.results_text.insert(tk.END, "  - Failed: Monobit test\n")
        else:
            self.results_text.insert(tk.END, f"  - Passed: Monobit test (p={monobit:.4f})\n")
        if runs < 0.01:
            success = False
            self.results_text.insert(tk.END, "  - Failed: Runs test\n")
        else:
            self.results_text.insert(tk.END, f"  - Passed: Runs test (p={runs:.4f})\n")
        if chi_squared < 0.01:
            success = False
            self.results_text.insert(tk.END, "  - Failed: Chi-squared test\n")
        else:
            self.results_text.insert(tk.END, f"  - Passed: Chi-squared test (p={chi_squared:.4f})\n")
        if avalanche < 0.4:
            success = False
            self.results_text.insert(tk.END, "  - Failed: Avalanche test\n")
        else:
            self.results_text.insert(tk.END, f"  - Passed: Avalanche test (effect={avalanche:.4f})\n")
        return success

    def seed_uniqueness_test(self) -> bool:
        """Test seed uniqueness"""
        self.results_text.insert(tk.END, "Testing Seed Uniqueness:\n")
        success = True
        used_seeds = set()
        test_messages = [b"Test1", b"Test2", b"Test3"]
        for msg in test_messages:
            seed, _, _, _ = self.shared_state['encryption'].find_perfect_entropy_seed(msg)
            if seed in used_seeds:
                success = False
                self.results_text.insert(tk.END, f"  - Failed: Duplicate seed {seed} found\n")
                break
            if seed is not None:
                used_seeds.add(seed)
        self.results_text.insert(tk.END, "  - Passed: All seeds are unique\n" if success else "  - Failed\n")
        return success

    def hash_integrity_test(self) -> bool:
        """Test hash integrity using actual encrypted messages"""
        self.results_text.insert(tk.END, "Testing Hash Integrity:\n")
        success = True
        encryption_system = self.shared_state['encryption']

        # Create list of message tuples
        test_messages = [
            b"Test message 1",
            b"Another test message",
            b"Test message with different content"
        ]

        # Add messages one by one
        for message in test_messages:
            result = encryption_system.add_message(message)
            if not result[0]:  # Check if message addition was successful
                self.results_text.insert(tk.END, f"Failed to add message: {message}\n")
                success = False
                break

        if success and encryption_system.messages:
            # Generate combined data without parameters
            combined = encryption_system.combine_messages()
            
            if combined:
                # Generate and verify hash
                hash_value = encryption_system.format_hash(combined)
                if not encryption_system.verify_hash(hash_value):
                    success = False
                    self.results_text.insert(tk.END, "  - Failed: Hash verification mismatch\n")
                else:
                    self.results_text.insert(tk.END, "  - Passed: Hash integrity verified\n")
            else:
                success = False
                self.results_text.insert(tk.END, "  - Failed: Could not generate combined data\n")
        else:
            success = False
            self.results_text.insert(tk.END, "  - Failed: No messages available for hash verification\n")

        return success