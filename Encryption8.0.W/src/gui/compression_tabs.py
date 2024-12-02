# compression_tabs.py

import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog, filedialog, messagebox
from typing import Dict, Any, List, Tuple, Optional
import hashlib
import time
import random
import string

# Assuming these modules exist and are correctly implemented
from utils.helpers import DataValidator, EncryptionHelper
from core.compress import OptimizedCompressor, CompressionResult


class MessageRow:
    """Class to encapsulate all UI elements related to a single message entry."""
    def __init__(self, parent: ttk.Frame, index: int, delete_callback):
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="x", pady=2)

        # Message Label and Entry
        self.message_label = ttk.Label(self.frame, text=f"Message {index + 1}:")
        self.message_label.pack(side="left")
        self.message_entry = ttk.Entry(self.frame, width=40)
        self.message_entry.pack(side="left", padx=5)

        # Optional Seed Entry
        self.seed_label = ttk.Label(self.frame, text="Seed (Optional):")
        self.seed_label.pack(side="left", padx=5)
        self.seed_entry = ttk.Entry(self.frame, width=10)
        self.seed_entry.pack(side="left", padx=5)

        # Optional Keyword Entry
        self.keyword_label = ttk.Label(self.frame, text="Keyword (Optional):")
        self.keyword_label.pack(side="left", padx=5)
        self.keyword_entry = ttk.Entry(self.frame, width=15)
        self.keyword_entry.pack(side="left", padx=5)

        # Entropy Label
        self.entropy_label = ttk.Label(self.frame, text="Entropy: -")
        self.entropy_label.pack(side="left", padx=5)

        # Delete Button
        self.delete_button = ttk.Button(self.frame, text="×", width=2,
                                        command=lambda: delete_callback(self))
        self.delete_button.pack(side="right", padx=5)

    def update_label(self, index: int):
        """Update the message label based on the current index."""
        self.message_label.configure(text=f"Message {index + 1}:")


class CompressedEncryptionTab(ttk.Frame):
    def __init__(self, parent, shared_state):
        """Initialize the compressed encryption tab."""
        super().__init__(parent)
        self.shared_state = shared_state
        self.validator = DataValidator()
        self.helper = EncryptionHelper()
        self.compressor = OptimizedCompressor()

        # Initialize storage lists for UI elements
        self.message_rows: List[MessageRow] = []

        # Track processing state and identifiers
        self.processing = False
        self.last_identifier = ""

        # Initialize encryption state storage
        encryption_system = self.shared_state.get('encryption')
        if encryption_system and hasattr(encryption_system, 'state'):
            if 'compressed_data' not in encryption_system.state:
                encryption_system.state['compressed_data'] = {
                    'encrypted_storage': {},
                    'pattern_storage': {},
                    'active_identifiers': set()
                }

            # Load storage from encryption state
            compressed_data = encryption_system.state['compressed_data']
            self.shared_state['encrypted_storage'] = compressed_data['encrypted_storage']
            self.shared_state['pattern_storage'] = compressed_data['pattern_storage']
            self.shared_state['active_identifiers'] = compressed_data['active_identifiers']

        self.setup_ui()

    def setup_ui(self):
        """Setup the compressed encryption tab UI."""
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

        # Compression Settings Frame
        compression_frame = ttk.LabelFrame(self, text="Compression Settings", padding=10)
        compression_frame.pack(fill="x", padx=10, pady=5)

        self.target_size_var = tk.StringVar(value="15")
        ttk.Label(compression_frame, text="Identifier Size:").pack(side="left")
        ttk.Entry(compression_frame, textvariable=self.target_size_var, width=10).pack(side="left", padx=5)

        # Identifier Operations Frame
        identifier_frame = ttk.LabelFrame(self, text="Identifier Operations", padding=10)
        identifier_frame.pack(fill="x", padx=10, pady=5)

        # Identifier Display
        self.identifier_text = scrolledtext.ScrolledText(identifier_frame, height=3, state='disabled')
        self.identifier_text.pack(fill="x", padx=5, pady=5)

        # Identifier Controls
        identifier_controls = ttk.Frame(identifier_frame)
        identifier_controls.pack(fill="x", padx=5, pady=5)

        ttk.Button(identifier_controls, text="Compress & Encrypt",
                   command=self.compress_and_encrypt).pack(side="left", padx=5)
        ttk.Button(identifier_controls, text="Copy Identifier",
                   command=self.copy_identifier).pack(side="left", padx=5)
        ttk.Button(identifier_controls, text="Save Identifier",
                   command=self.save_identifier).pack(side="left", padx=5)

        # Results Frame
        results_frame = ttk.LabelFrame(self, text="Results & Status", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, state='disabled')
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
        """Create a message input row with optional seed and keyword fields."""
        row = MessageRow(parent, index, self.delete_row)
        self.message_rows.append(row)

    def compress_and_encrypt(self):
        """
        Main function to handle the compression and encryption process.
        Provides user feedback and handles potential errors gracefully.
        """
        if self.processing:
            messagebox.showwarning("Processing", "Already processing messages. Please wait.")
            return

        self.processing = True
        self.clear_results()
        self.update_status("Processing messages...")

        try:
            target_size = int(self.target_size_var.get())
            if target_size <= 0:
                raise ValueError
        except ValueError:
            self.update_status("Invalid identifier size. Please enter a positive integer.")
            self.processing = False
            return

        valid_messages = []
        total_messages = len([row for row in self.message_rows if row.message_entry.get().strip()])

        if total_messages == 0:
            self.update_status("No messages to process.")
            self.processing = False
            return

        for i, row in enumerate(self.message_rows):
            message = row.message_entry.get().strip()
            if not message:
                continue

            seed = row.seed_entry.get().strip()
            keyword = row.keyword_entry.get().strip()

            # Handle optional seed
            if seed:
                try:
                    seed = int(seed)
                except ValueError:
                    self.append_result(f"Invalid seed format for Message {i + 1}. Skipping this message.\n")
                    row.entropy_label.configure(text="Entropy: Invalid Seed")
                    continue
            else:
                seed = None  # No seed provided, will be handled later

            # Keyword can be left as is; if empty, it will be generated later
            valid_messages.append((message, seed, keyword))

            # Update progress
            self.progress_var.set((len(valid_messages) / total_messages) * 100)
            self.update()

        if not valid_messages:
            self.update_status("No valid messages to process.")
            self.processing = False
            return

        self.process_compressed_messages(valid_messages, target_size)
        self.processing = False
        self.update_status("Processing completed")

    def process_compressed_messages(self, messages: List[Tuple[str, Optional[int], str]], target_size: int) -> None:
        """Process messages with proper entropy search and keyword handling."""
        current_time = int(time.time() * 1000)
        compressed_results = []

        encryption_system = self.shared_state.get('encryption')
        if not encryption_system:
            self.append_result("Encryption system not initialized.\n")
            return

        # Clear previous encryption data
        encryption_system.messages.clear()
        encryption_system.perfect_seeds.clear()
        encryption_system.encryption_data.clear()
        self.shared_state.setdefault('pattern_storage', {})

        for i, (message, provided_seed, keyword) in enumerate(messages):
            self.append_result(f"\nProcessing Message {i + 1}\n")

            try:
                # First handle seed - either custom or find perfect
                if provided_seed is not None:
                    # Process message with custom seed
                    success, entropy, pattern_key = encryption_system.process_message_with_custom_seed(
                        message.encode(), provided_seed, keyword
                    )
                    if not success:
                        self.append_result(f"Provided seed {provided_seed} does not achieve sufficient entropy.\n")
                        self.message_rows[i].entropy_label.configure(text="Entropy: Failed")
                        continue
                    else:
                        used_seed = int(pattern_key.split('_')[0])
                        keyword = pattern_key.split('_')[1]
                        self.append_result(f"Using provided seed {used_seed} (entropy: {entropy:.10f})\n")
                else:
                    # Use standard perfect entropy seed finding
                    self.append_result("Searching for perfect entropy seed...\n")
                    seed_result = encryption_system.find_perfect_entropy_seed(message.encode())
                    if seed_result and seed_result[0] is not None:
                        used_seed, iv, ciphertext, entropy = seed_result
                        success = True
                        self.append_result(f"Found perfect seed {used_seed} (entropy: {entropy:.10f})\n")
                    else:
                        success = False
                        self.append_result("Failed to find perfect entropy seed.\n")
                        self.message_rows[i].entropy_label.configure(text="Entropy: Failed")
                        continue

                # Proceed only if success
                if success:
                    # Handle keyword after seed is determined
                    if not keyword:
                        # Generate random keyword if none provided
                        keyword = ''.join(random.choices(string.ascii_letters, k=8))

                    # Create pattern key with determined seed and keyword
                    pattern_key = f"{used_seed}_{keyword}_{hashlib.sha256(message.encode()).hexdigest()[:8]}"

                    # Now do compression
                    compression_result: CompressionResult = self.compressor.compress(message.encode())

                    # Ensure compression_result has the required attributes
                    compressed_data = compression_result.data
                    compression_ratio = compression_result.compression_ratio
                    patterns = compression_result.patterns
                    pattern_map = compression_result.pattern_map

                    # Append pattern key to compressed data
                    pattern_key_bytes = pattern_key.encode('utf-8')
                    compressed_data_with_key = compressed_data + pattern_key_bytes

                    # Encrypt the compressed data with the pattern key appended
                    iv, ciphertext = encryption_system.encrypt_with_seed(compressed_data_with_key, used_seed)

                    # Store patterns for decompression
                    self.shared_state['pattern_storage'][pattern_key] = {
                    'patterns': patterns,
                    'pattern_map': pattern_map,
                    'checksum': hashlib.sha256(message.encode()).hexdigest()
                }

                    # Store the encrypted data and related information
                    message_id = len(encryption_system.messages)
                    encryption_system.store_message(compressed_data_with_key, used_seed, iv, ciphertext, entropy, message_id)

                    # Create a combined segment for this message
                    segment = encryption_system.create_encrypted_segment(used_seed, iv, ciphertext, entropy)

                    # Add the segment to the combined data
                    if not hasattr(encryption_system, 'combined_data'):
                        encryption_system.combined_data = b''
                    encryption_system.combined_data += segment

                    compressed_results.append({
                        'pattern_key': pattern_key,
                        'seed': used_seed,
                        'keyword': keyword,
                        'verification': hashlib.sha256(message.encode()).hexdigest(),
                        'timestamp': current_time
                    })

                    # Update UI
                    self.message_rows[i].entropy_label.configure(text=f"Entropy: {entropy:.10f}")
                    self.append_result(
                        f"Compression ratio: {compression_ratio:.2%}\n"
                        f"Pattern key: {pattern_key}\n"
                        f"Keyword: {keyword}\n"
                    )
                else:
                    self.message_rows[i].entropy_label.configure(text="Entropy: Failed")
                    self.append_result("Failed to achieve required entropy.\n")

            except Exception as e:
                self.append_result(f"Error processing message {i + 1}: {str(e)}\n")
                continue

        # Generate final result
        if compressed_results and hasattr(encryption_system, 'combined_data'):
            combined = encryption_system.combined_data
            identifier = hashlib.sha256(combined).hexdigest()[:target_size]

            self.shared_state.setdefault('encrypted_storage', {})
            self.shared_state['encrypted_storage'][identifier] = {
                'combined_data': combined,
                'patterns': compressed_results,
                'timestamp': current_time
            }

            self.shared_state.setdefault('active_identifiers', set())
            self.shared_state['active_identifiers'].add(identifier)

            # Update encryption state with the latest compressed data
            if hasattr(encryption_system, 'state'):
                encryption_system.state['compressed_data'] = {
                    'encrypted_storage': self.shared_state['encrypted_storage'],
                    'pattern_storage': self.shared_state['pattern_storage'],
                    'active_identifiers': self.shared_state['active_identifiers']
                }
                # Save the state if needed
                if hasattr(encryption_system, 'save_state'):
                    encryption_system.save_state()

            # Update UI
            self.identifier_text.configure(state='normal')
            self.identifier_text.delete(1.0, tk.END)
            self.identifier_text.insert(tk.END, identifier)
            self.identifier_text.configure(state='disabled')
            self.last_identifier = identifier

            self.append_result(
                f"\nIdentifier (size {target_size} characters): {identifier}\n"
                f"Encrypted data stored for decryption.\n"
            )





    def add_more_rows(self):
        """Add additional message input rows."""
        num_rows = simpledialog.askinteger("Add Rows",
                                           "Enter number of rows to add:",
                                           minvalue=1, maxvalue=20)
        if num_rows:
            parent = self.message_rows[0].frame.master  # Assuming all rows have the same parent
            current_count = len(self.message_rows)
            for i in range(num_rows):
                self.create_message_row(parent, current_count + i)
            self.update_row_numbers()

    def fill_random_data(self):
        """Fill only message entries with random data for perfect entropy search."""
        for row in self.message_rows:
            # Generate only message text
            length = random.randint(10, 50)
            random_text = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
            row.message_entry.delete(0, tk.END)
            row.message_entry.insert(0, random_text)

            # Clear seeds and keywords - let perfect entropy search happen first
            row.seed_entry.delete(0, tk.END)
            row.keyword_entry.delete(0, tk.END)
            row.entropy_label.configure(text="Entropy: -")

        self.identifier_text.configure(state='normal')
        self.identifier_text.delete(1.0, tk.END)
        self.identifier_text.configure(state='disabled')
        self.clear_results()
        self.progress_var.set(0)
        self.update_status("Random data filled.")

    def clear_all(self):
        """
        Clear all input fields and results, resetting the tab to its initial state.
        """
        for row in self.message_rows:
            row.message_entry.delete(0, tk.END)
            row.seed_entry.delete(0, tk.END)
            row.keyword_entry.delete(0, tk.END)
            row.entropy_label.configure(text="Entropy: -")

        self.identifier_text.configure(state='normal')
        self.identifier_text.delete(1.0, tk.END)
        self.identifier_text.configure(state='disabled')
        self.clear_results()
        self.progress_var.set(0)
        self.update_status("All fields cleared")
        self.last_identifier = ""

    def delete_row(self, row: MessageRow):
        """Delete a message row."""
        if len(self.message_rows) <= 1:
            messagebox.showwarning("Delete Row", "At least one message row must remain.")
            return

        row.frame.destroy()
        self.message_rows.remove(row)
        self.update_row_numbers()

    def update_row_numbers(self):
        """Update message row numbers after deletion or addition."""
        for index, row in enumerate(self.message_rows):
            row.update_label(index)

    def copy_identifier(self):
        """Copy identifier to clipboard."""
        identifier = self.identifier_text.get(1.0, tk.END).strip()
        if identifier:
            self.clipboard_clear()
            self.clipboard_append(identifier)
            self.update_status("Identifier copied to clipboard")
        else:
            messagebox.showinfo("Copy Identifier", "No identifier to copy.")

    def save_identifier(self):
        """Save identifier to file."""
        identifier = self.identifier_text.get(1.0, tk.END).strip()
        if not identifier:
            messagebox.showinfo("Save Identifier", "No identifier to save.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".id",
            filetypes=[("Identifier files", "*.id"), ("Text files", "*.txt"),
                      ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(identifier)
                self.update_status(f"Identifier saved to {filename}")
            except Exception as e:
                messagebox.showerror("Save Identifier", f"Failed to save identifier: {str(e)}")

    def update_status(self, message: str):
        """Update status label with current operation status."""
        self.status_label.configure(text=message)
        self.update_idletasks()

    def append_result(self, message: str):
        """Append message to the results text area."""
        self.results_text.configure(state='normal')
        self.results_text.insert(tk.END, message)
        self.results_text.see(tk.END)
        self.results_text.configure(state='disabled')

    def clear_results(self):
        """Clear the results text area."""
        self.results_text.configure(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.configure(state='disabled')


class CompressedDecryptionTab(ttk.Frame):
    def __init__(self, parent: ttk.Notebook, shared_state: Dict[str, Any]):
        super().__init__(parent)
        self.shared_state = shared_state
        self.helper = EncryptionHelper()
        self.compressor = OptimizedCompressor()

        # Load storage from encryption state
        encryption_system = self.shared_state.get('encryption')
        if encryption_system and hasattr(encryption_system, 'state'):
            compressed_data = encryption_system.state.get('compressed_data', {})
            self.shared_state['encrypted_storage'] = compressed_data.get('encrypted_storage', {})
            self.shared_state['pattern_storage'] = compressed_data.get('pattern_storage', {})
            self.shared_state['active_identifiers'] = compressed_data.get('active_identifiers', set())

        self.setup_ui()

    def setup_ui(self):
        """Setup the compressed decryption tab UI."""
        # Identifier Input Frame
        identifier_frame = ttk.LabelFrame(self, text="Input Identifier for Decryption", padding=10)
        identifier_frame.pack(fill="x", padx=10, pady=5)

        # Identifier Textbox
        self.identifier_text = scrolledtext.ScrolledText(identifier_frame, height=3)
        self.identifier_text.pack(fill="x", padx=5, pady=5)

        # Identifier Controls
        identifier_controls = ttk.Frame(identifier_frame)
        identifier_controls.pack(fill="x", padx=5, pady=5)

        ttk.Button(identifier_controls, text="Load Identifier",
                   command=self.load_identifier).pack(side="left", padx=5)

        # Decryption Input Frame
        decrypt_frame = ttk.LabelFrame(self, text="Decryption Details", padding=10)
        decrypt_frame.pack(fill="x", padx=10, pady=5)

        # Seed Input
        seed_frame = ttk.Frame(decrypt_frame)
        seed_frame.pack(fill="x", pady=5)
        ttk.Label(seed_frame, text="Seed:").pack(side="left")
        self.seed_entry = ttk.Entry(seed_frame, width=20)
        self.seed_entry.pack(side="left", padx=5)

        # Keyword Input
        keyword_frame = ttk.Frame(decrypt_frame)
        keyword_frame.pack(fill="x", pady=5)
        ttk.Label(keyword_frame, text="Keyword:").pack(side="left")
        self.keyword_entry = ttk.Entry(keyword_frame, width=30)
        self.keyword_entry.pack(side="left", padx=5)

        # Decrypt Button
        ttk.Button(decrypt_frame, text="Decrypt & Decompress",
                   command=self.decrypt_and_decompress).pack(pady=10)

        # Results Frame
        results_frame = ttk.LabelFrame(self, text="Decryption Results", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, state='disabled')
        self.results_text.pack(fill="both", expand=True)

    def decrypt_and_decompress(self):
        """Decrypt and decompress messages"""
        identifier = self.identifier_text.get(1.0, tk.END).strip()
        if not identifier:
            self.append_result("No identifier provided for decryption.\n")
            return

        print(f"Debug - Identifier: {identifier}")

        # Verify identifier exists
        if identifier not in self.shared_state.get('active_identifiers', set()):
            self.append_result(f"Invalid or expired identifier: {identifier}\n")
            return

        stored = self.shared_state['encrypted_storage'].get(identifier)
        if not stored:
            self.append_result(f"No encrypted data found for identifier: {identifier}\n")
            return

        combined_data = stored['combined_data']
        print(f"Debug - Length of raw combined data: {len(combined_data)}")

        seed_input = self.seed_entry.get().strip()
        keyword = self.keyword_entry.get().strip()

        if not seed_input or not keyword:
            self.append_result("Both seed and keyword are required.\n")
            return

        try:
            seed = int(seed_input)
            print(f"Debug - Using seed: {seed}")

            # Find matching pattern
            pattern_match = None
            for pattern in stored.get('patterns', []):
                if pattern['seed'] == seed and pattern['keyword'] == keyword:
                    pattern_match = pattern
                    break

            if not pattern_match:
                self.append_result(f"No matching pattern found for seed={seed}, keyword='{keyword}'\n")
                return

            pattern_key = pattern_match['pattern_key']
            pattern_data = self.shared_state['pattern_storage'].get(pattern_key)

            if not pattern_data:
                self.append_result(f"Pattern data not found for key: {pattern_key}\n")
                return

            # Ensure combined_data is bytes
            try:
                if isinstance(combined_data, str):
                    combined_data = bytes.fromhex(combined_data)

                # Extract and decrypt message
                message, _ = self.shared_state['encryption'].extract_message(combined_data, seed)

                if not message:
                    self.append_result("Failed to decrypt message. Please verify your seed.\n")
                    return

                # Verify pattern key in decrypted message
                pattern_key_bytes = pattern_key.encode('utf-8')
                if len(message) <= len(pattern_key_bytes):
                    self.append_result("Decrypted message is too short.\n")
                    return

                extracted_pattern_key = message[-len(pattern_key_bytes):]
                if extracted_pattern_key != pattern_key_bytes:
                    self.append_result("Pattern key verification failed.\n")
                    return

                # Extract compressed data without pattern key
                compressed_data = message[:-len(pattern_key_bytes)]

                # Configure decompression with stored patterns
                decompressor = OptimizedCompressor()
                decompressor.patterns = pattern_data['patterns']
                decompressor.pattern_to_id = pattern_data['pattern_map']

                # Decompress and verify
                decompressed = decompressor.decompress(compressed_data)
                calculated_checksum = hashlib.sha256(decompressed).hexdigest()

                if calculated_checksum != pattern_data['checksum']:
                    self.append_result("Warning: Checksum verification failed.\n")

                # Display decoded message
                decoded = decompressed.decode('utf-8', errors='replace')
                self.append_result(f"Successfully decrypted:\n{decoded}\n")

            except Exception as e:
                print(f"Debug - Exception during decryption: {str(e)}")
                self.append_result(f"Error during decryption: {str(e)}\n")

        except ValueError:
            self.append_result("Invalid seed format. Seed must be an integer.\n")




    def load_identifier(self):
        """Load an identifier from file."""
        filename = filedialog.askopenfilename(
            filetypes=[("Identifier files", "*.id"), ("Text files", "*.txt"),
                      ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    identifier = f.read().strip()
                self.identifier_text.delete(1.0, tk.END)
                self.identifier_text.insert(tk.END, identifier)
                self.update_status(f"Identifier loaded from {filename}")
            except Exception as e:
                self.append_result(f"Error loading identifier: {str(e)}\n")

    def clear(self):
        """Clear all fields."""
        self.identifier_text.delete(1.0, tk.END)
        self.seed_entry.delete(0, tk.END)
        self.keyword_entry.delete(0, tk.END)
        self.results_text.configure(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.configure(state='disabled')

    def append_result(self, message: str):
        """Append message to the results text area."""
        self.results_text.configure(state='normal')
        self.results_text.insert(tk.END, message)
        self.results_text.see(tk.END)
        self.results_text.configure(state='disabled')