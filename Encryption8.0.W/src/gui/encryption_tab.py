import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog, filedialog
from typing import Dict, Any, List, Optional
import time
import random
import string
from utils.helpers import DataValidator, EncryptionHelper

class EncryptionTab(ttk.Frame):
    def __init__(self, parent: ttk.Notebook, shared_state: Dict[str, Any]):
        super().__init__(parent)
        self.shared_state = shared_state
        self.validator = DataValidator()
        self.helper = EncryptionHelper()
        
        self.message_entries: List[ttk.Entry] = []
        self.entropy_labels: List[ttk.Label] = []
        self.seed_entries: List[ttk.Entry] = []
        
        self.processing = False
        self.last_hash = ""
        
        self.setup_ui()

    def setup_ui(self):
        """Setup the encryption tab UI"""
        # Message Input Frame
        input_frame = ttk.LabelFrame(self, text="Message Input", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)

        # Create initial message entries
        for i in range(5):
            self.create_message_row(input_frame, i)

        # Buttons Frame
        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(buttons_frame, text="Add More Rows",
                   command=self.add_more_rows).pack(side="left", padx=5)
                   
        ttk.Button(buttons_frame, text="Fill Random",
                   command=self.fill_random_data).pack(side="left", padx=5)

        ttk.Button(buttons_frame, text="Clear All",
                   command=self.clear_all).pack(side="left", padx=5)

        # Hash Operations Frame
        hash_frame = ttk.LabelFrame(self, text="Hash Operations", padding=10)
        hash_frame.pack(fill="x", padx=10, pady=5)

        # Hash Display
        self.hash_text = scrolledtext.ScrolledText(hash_frame, height=3)
        self.hash_text.pack(fill="x", padx=5, pady=5)

        # Hash Controls
        hash_controls = ttk.Frame(hash_frame)
        hash_controls.pack(fill="x", padx=5, pady=5)

        ttk.Button(hash_controls, text="Stack Messages",
                   command=self.stack_messages).pack(side="left", padx=5)
        ttk.Button(hash_controls, text="Copy Hash",
                   command=self.copy_hash).pack(side="left", padx=5)
        ttk.Button(hash_controls, text="Save Hash",
                   command=self.save_hash).pack(side="left", padx=5)
        ttk.Button(hash_controls, text="Load Hash",
                   command=self.load_hash).pack(side="left", padx=5)
        
        # New Decrypt Button
        ttk.Button(hash_controls, text="Decrypt Last Message",
                   command=self.decrypt_last_message).pack(side="left", padx=5)

        # Results Frame
        results_frame = ttk.LabelFrame(self, text="Results & Status", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=15)
        self.results_text.pack(fill="both", expand=True)

        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(results_frame, 
                                          variable=self.progress_var,
                                          maximum=100)
        self.progress_bar.pack(fill="x", padx=5, pady=5)

        # Status Label
        self.status_label = ttk.Label(results_frame, text="Ready")
        self.status_label.pack(pady=5)

    def create_message_row(self, parent: ttk.Frame, index: int):
        """Create a new message input row"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)

        # Message Label and Entry
        ttk.Label(frame, text=f"Message {index + 1}:").pack(side="left")
        
        entry = ttk.Entry(frame, width=50)
        entry.pack(side="left", padx=5)
        self.message_entries.append(entry)

        # Entropy Label
        entropy_label = ttk.Label(frame, text="Entropy: -")
        entropy_label.pack(side="left", padx=5)
        self.entropy_labels.append(entropy_label)

        # Seed Entry (Optional)
        ttk.Label(frame, text="Seed (optional):").pack(side="left", padx=5)
        seed_entry = ttk.Entry(frame, width=20)
        seed_entry.pack(side="left", padx=5)
        self.seed_entries.append(seed_entry)

        # Delete Button
        ttk.Button(frame, text="×", width=2,
                   command=lambda: self.delete_row(frame, index)).pack(side="right", padx=5)

    def delete_row(self, frame: ttk.Frame, index: int):
        """Delete a message row"""
        frame.destroy()
        self.message_entries.pop(index)
        self.entropy_labels.pop(index)
        self.seed_entries.pop(index)
        self.update_row_numbers()

    def update_row_numbers(self):
        """Update message row numbers after deletion"""
        for i, frame in enumerate(self.message_entries[0].master.winfo_children()):
            if isinstance(frame, ttk.Frame):
                label = frame.winfo_children()[0]
                label.configure(text=f"Message {i + 1}:")

    def add_more_rows(self):
        """Add additional message input rows"""
        num_rows = simpledialog.askinteger("Add Rows", 
                                        "Enter number of rows to add:",
                                        minvalue=1, maxvalue=20)
        if num_rows:
            for i in range(num_rows):
                self.create_message_row(self.message_entries[0].master,
                                     len(self.message_entries))

    def fill_random_data(self):
        """Fill entries with random test data"""
        for entry in self.message_entries:
            length = random.randint(10, 50)
            random_text = ''.join(random.choices(string.ascii_letters + 
                                               string.digits, k=length))
            entry.delete(0, tk.END)
            entry.insert(0, random_text)

    def stack_messages(self):
        """Process and stack all messages"""
        if self.processing:
            return
            
        self.processing = True
        self.results_text.delete(1.0, tk.END)
        self.update_status("Processing messages...")
        start_time = time.time()

        valid_messages = []
        total_messages = len([e for e in self.message_entries if e.get().strip()])

        for i, (entry, seed_entry) in enumerate(zip(
                self.message_entries, self.seed_entries)):
            message = entry.get().strip()
            if not message:
                continue

            self.progress_var.set((i / total_messages) * 100)
            self.update()

            seed_input = seed_entry.get().strip()
            seed = None
            if seed_input:
                try:
                    seed = int(seed_input)
                    if seed < 0:
                        raise ValueError
                except ValueError:
                    self.results_text.insert(tk.END, 
                        f"Invalid seed for Message {i + 1}. Using automatic seed.\n")

            valid_messages.append((message, seed))

        self.process_messages(valid_messages)

        end_time = time.time()
        self.update_status(f"Processing completed in {end_time - start_time:.2f} seconds")
        self.progress_var.set(100)
        self.processing = False

    def process_messages(self, messages: List[tuple]):
        """Process validated messages"""
        for i, (message, seed) in enumerate(messages):
            self.results_text.insert(tk.END, f"\nProcessing Message {i + 1}: {message}\n")
            self.update()

            success, entropy = self.shared_state['encryption'].add_message(
                message.encode(), seed)

            if success:
                self.entropy_labels[i].configure(text=f"Entropy: {entropy:.10f}")
                self.results_text.insert(tk.END,
                    f"Added with seed: {self.shared_state['encryption'].perfect_seeds[-1]}\n")
            else:
                self.entropy_labels[i].configure(text="Entropy: Failed")
                self.results_text.insert(tk.END, "Failed to find perfect entropy\n")

        # Generate combined hash
        if self.shared_state['encryption'].messages:
            combined = self.shared_state['encryption'].combine_messages()
            hash_value = self.shared_state['encryption'].format_hash(combined)
            
            self.hash_text.delete(1.0, tk.END)
            self.hash_text.insert(tk.END, hash_value)
            self.last_hash = hash_value

            self.results_text.insert(tk.END, f"\nCombined size: {len(combined)} bytes\n")
            
            # Update visualization if available
            if 'visualization' in self.shared_state:
                self.shared_state['visualization'].refresh()

    def copy_hash(self):
        """Copy hash to clipboard"""
        hash_value = self.hash_text.get(1.0, tk.END).strip()
        if hash_value:
            self.clipboard_clear()
            self.clipboard_append(hash_value)
            self.update_status("Hash copied to clipboard")

    def save_hash(self):
        """Save hash to file"""
        hash_value = self.hash_text.get(1.0, tk.END).strip()
        if not hash_value:
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".hash",
            filetypes=[("Hash files", "*.hash"), ("Text files", "*.txt"), 
                      ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(hash_value)
            self.update_status(f"Hash saved to {filename}")

    def load_hash(self):
        """Load hash from file"""
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
                self.update_status(f"Hash loaded from {filename}")
            except Exception as e:
                self.update_status(f"Error loading hash: {str(e)}")

    def clear_all(self):
        """Clear all entries and results"""
        for entry in self.message_entries:
            entry.delete(0, tk.END)
        for label in self.entropy_labels:
            label.configure(text="Entropy: -")
        for seed_entry in self.seed_entries:
            seed_entry.delete(0, tk.END)

        self.hash_text.delete(1.0, tk.END)
        self.results_text.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.update_status("All fields cleared")
        self.last_hash = ""

    def decrypt_last_message(self):
        """Decrypt the last stored message and display it."""
        encryption_system = self.shared_state['encryption']

        # Check if there is a message to decrypt
        if not encryption_system.messages:
            self.results_text.insert(tk.END, "No messages available for decryption.\n")
            return

        # Retrieve the last encrypted data
        last_index = len(encryption_system.messages) - 1
        seed = encryption_system.perfect_seeds[last_index]
        iv, ciphertext, _ = encryption_system.encryption_data[last_index]

        # Decrypt the message
        try:
            decrypted_message = encryption_system.decrypt_with_seed(ciphertext, seed, iv)
            decrypted_text = decrypted_message.decode("utf-8", errors="replace")
            self.results_text.insert(tk.END, f"Decrypted Message:\n{decrypted_text}\n")
        except Exception as e:
            self.results_text.insert(tk.END, f"Error during decryption: {str(e)}\n")

    def update_status(self, message: str):
        """Update status label"""
        self.status_label.configure(text=message)
        self.update()

    def validate_all(self) -> bool:
        """Validate all inputs"""
        valid = True
        for i, (entry, seed_entry) in enumerate(zip(
                self.message_entries, self.seed_entries)):
            message = entry.get().strip()
            if not message:
                continue

            if not self.validator.validate_message(message):
                self.results_text.insert(tk.END, 
                    f"Invalid message format in row {i + 1}\n")
                valid = False

            seed = seed_entry.get().strip()
            if seed and not self.validator.validate_seed(seed):
                self.results_text.insert(tk.END, 
                    f"Invalid seed format in row {i + 1}\n")
                valid = False

        return valid
