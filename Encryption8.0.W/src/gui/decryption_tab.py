import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from typing import Dict, Any, List, Optional
from utils.helpers import EncryptionHelper

class DecryptionTab(ttk.Frame):
    def __init__(self, parent: ttk.Notebook, shared_state: Dict[str, Any]):
        super().__init__(parent)
        self.shared_state = shared_state
        self.helper = EncryptionHelper()
        
        self.setup_ui()

    def setup_ui(self):
        """Setup the decryption tab UI"""
        # Hash Input Frame
        hash_frame = ttk.LabelFrame(self, text="Input Hash for Decryption", padding=10)
        hash_frame.pack(fill="x", padx=10, pady=5)

        # Hash Textbox
        self.hash_text = scrolledtext.ScrolledText(hash_frame, height=3)
        self.hash_text.pack(fill="x", padx=5, pady=5)

        # Hash Controls
        hash_controls = ttk.Frame(hash_frame)
        hash_controls.pack(fill="x", padx=5, pady=5)

        ttk.Button(hash_controls, text="Load Hash",
                   command=self.load_hash).pack(side="left", padx=5)

        # Seed Input Frame
        seed_frame = ttk.LabelFrame(self, text="Input Seed(s) for Decryption", padding=10)
        seed_frame.pack(fill="x", padx=10, pady=5)

        self.seed_entry = ttk.Entry(seed_frame, width=50)
        self.seed_entry.pack(side="left", padx=5, pady=5)
        
        # Add Seed Button
        ttk.Button(seed_frame, text="Add Seed",
                   command=self.add_seed).pack(side="left", padx=5)

        # List of Seeds
        self.seed_list = []
        self.seed_listbox = tk.Listbox(seed_frame, height=5, selectmode=tk.SINGLE)
        self.seed_listbox.pack(fill="x", padx=5, pady=5)

        # Decrypt Button
        ttk.Button(seed_frame, text="Decrypt Selected",
                   command=self.decrypt_selected).pack(side="left", padx=5)

        # Results Frame
        results_frame = ttk.LabelFrame(self, text="Decryption Results", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=15)
        self.results_text.pack(fill="both", expand=True)

    def load_hash(self):
        """Load a combined hash from a file"""
        filename = filedialog.askopenfilename(
            filetypes=[("Hash files", "*.hash"), ("Text files", "*.txt"), 
                      ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    hash_value = f.read().strip()
                self.hash_text.delete(1.0, tk.END)
                self.hash_text.insert(tk.END, hash_value)
            except Exception as e:
                self.results_text.insert(tk.END, f"Error loading hash: {str(e)}\n")

    def add_seed(self):
        """Add a seed to the list of seeds"""
        seed = self.seed_entry.get().strip()
        if seed.isdigit():
            self.seed_list.append(int(seed))
            self.seed_listbox.insert(tk.END, f"Seed: {seed}")
            self.seed_entry.delete(0, tk.END)
        else:
            self.results_text.insert(tk.END, "Invalid seed format. Only integers are allowed.\n")

    def decrypt_selected(self):
        """Decrypt messages based on selected seeds in the seed list"""
        hash_data = self.hash_text.get(1.0, tk.END).strip()
        if not hash_data:
            self.results_text.insert(tk.END, "No hash data to decrypt.\n")
            return

        encryption_system = self.shared_state['encryption']
        combined_data = bytes.fromhex(hash_data)

        self.results_text.insert(tk.END, "Starting decryption...\n")
        for seed in self.seed_list:
            try:
                message, timestamp = encryption_system.extract_message(combined_data, seed)
                if message:
                    decoded_message = message.decode("utf-8", errors="replace")
                    self.results_text.insert(
                        tk.END, f"Decrypted Message (Seed {seed}):\n{decoded_message}\n\n"
                    )
                else:
                    self.results_text.insert(
                        tk.END, f"Failed to decrypt message with Seed {seed}.\n"
                    )
            except Exception as e:
                self.results_text.insert(
                    tk.END, f"Error decrypting message with Seed {seed}: {str(e)}\n"
                )

    def clear_results(self):
        """Clear the decryption results and seed list"""
        self.results_text.delete(1.0, tk.END)
        self.seed_listbox.delete(0, tk.END)
        self.seed_list.clear()
