from scienceai.client import ScienceAI
import os
import shutil
import time
import warnings

# Setup
pdf_path = os.path.abspath("test_paper.pdf")
if not os.path.exists(pdf_path):
    print("Error: test_paper.pdf not found")
    exit(1)

print("--- Test 1: Explicit Project and Storage ---")
project_name = "TestProject_Interface_v2"
storage_path = os.path.join(os.getcwd(), "test_interface_storage_v2")
if os.path.exists(storage_path):
    shutil.rmtree(storage_path)
os.makedirs(storage_path)

client = ScienceAI(project_name, storage_path=storage_path)

# Test State Validation
try:
    client.chat("Should fail")
    print("Error: State validation failed (allowed chat before upload)")
except RuntimeError as e:
    print(f"Caught expected error: {e}")

print("Uploading paper...")
client.upload_papers([pdf_path], trigger_preprocess=True)

print("Sending chat message...")
client.chat_background("What is the title of the paper?")

print("Polling for response...")
start_poll = time.time()
while True:
    result = client.poll()
    if result:
        print(f"Response received via poll: {result}")
        break
    if time.time() - start_poll > 60:
        print("Error: Timed out polling")
        break
    time.sleep(1)

print("Checking history...")
hist = client.history()
print(f"History length: {len(hist)}")

client.close()
print("Test complete.")
