import traceback


async def compress_conversation_context(dm):
    """
    Compress the conversation by summarizing the first 50% of tool calls and tool responses.
    Each tool message is summarized into a few sentences and replaced with an assistant message.
    """
    from .llm import client
    from datetime import datetime
    import dictdatabase as DDB
    
    # Add user request message as pending (marked internal so PI won't include in context)
    user_request_msg = {
        "content": "🗜️ Please compress the conversation to free up context space.",
        "role": "user",
        "status": "Pending",
        "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z'),
        "internal": True  # Flag to exclude from PI context
    }
    dm.add_chat(user_request_msg)
    
    messages = dm.get_database_chat()
    
    # Find all tool-related messages that haven't been compressed yet
    tool_indices = []
    for i, msg in enumerate(messages):
        # Skip already compressed messages
        if msg.get("compressed"):
            continue
        if msg.get("role") == "tool" or msg.get("tool_calls"):
            tool_indices.append(i)
    
    if len(tool_indices) == 0:
        print("No tool messages to compress")
        return
    
    # Get the first 50% of tool messages to compress (round up so odd numbers eventually reach 0)
    import math
    compress_count = math.ceil(len(tool_indices) / 2)
    
    indices_to_compress = tool_indices[:compress_count]
    print(f"Compressing {len(indices_to_compress)} tool messages out of {len(tool_indices)} total")
    
    # Process each message to compress
    compressed_messages = []
    for idx in sorted(indices_to_compress, reverse=True):  # Process in reverse to maintain indices
        msg = messages[idx]
        original_content = msg.get("content", "")
        original_role = msg.get("role", "")
        
        # Skip if content is too short
        if len(original_content) < 100:
            continue
        
        # Summarize with gpt-4.1-mini
        try:
            summary_prompt = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes tool call results concisely. "
                               "Summarize the following tool output in 2-3 sentences, preserving key data points and findings. "
                               "Be factual and concise."
                },
                {
                    "role": "user", 
                    "content": f"Summarize this tool output:\n\n{original_content[:8000]}"  # Limit input size
                }
            ]
            
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=summary_prompt,
                max_completion_tokens=1000
            )
            
            summary = response.choices[0].message.content
            
            # Create replacement message
            compressed_content = (
                f"[📦 Compressed Tool Output]\n\n"
                f"{summary}\n\n"
                f"---\n"
                f"*This message was automatically summarized to free up context space. "
                f"Original content was {len(original_content)} characters.*"
            )
            
            # Replace the message in place
            messages[idx] = {
                "content": compressed_content,
                "role": "assistant",
                "status": "Processed",
                "time": msg.get("time", datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')),
                "compressed": True,
                "original_role": original_role
            }
            
            print(f"Compressed message at index {idx}: {len(original_content)} -> {len(compressed_content)} chars")
            
        except Exception as e:
            print(f"Failed to compress message at index {idx}: {e}")
            continue
    
    # Remove the CONTEXT_LIMIT_REACHED message if present
    messages = [m for m in messages if m.get("content") != "CONTEXT_LIMIT_REACHED"]
    
    # Mark user's compression request as processed and add assistant response
    # Find the user request message and mark it processed
    for msg in reversed(messages):
        if msg.get("content") == "🗜️ Please compress the conversation to free up context space." and msg.get("status") == "Pending":
            msg["status"] = "Processed"
            break
    
    # Add assistant response as pending (marked internal so PI won't include in context)
    compression_response = {
        "content": f"🗜️ **Conversation Compressed**\n\nI've summarized {len(indices_to_compress)} older tool outputs to free up context space. You can now continue the discussion.",
        "role": "assistant",
        "status": "Pending",
        "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z'),
        "internal": True  # Flag to exclude from PI context
    }
    messages.append(compression_response)
    
    # Save the compressed messages back to the database
    with DDB.at("chat").session() as (session, chat):
        chat["messages"] = messages
        session.write()
    
    # Mark the assistant response as processed
    dm.update_last_chat("Processed")
    
    print(f"Context compression complete. Compressed {len(indices_to_compress)} messages.")


def run_backend(folder, project_path, storage_path, message_queue, stop_event, ingest=True, error_queue=None):
    import asyncio
    
    async def run_backend_async():
        try:
            from .llm import update_stop_event
            update_stop_event(stop_event)
            from .process_paper import process_paper
            from .database_manager import DatabaseManager
            from .principal_investigator import PrincipalInvestigator
            import subprocess
            import sys
            import time
            import dictdatabase as DDB
            from datetime import datetime
            import re
            start = time.time()
            dm = DatabaseManager(folder, process_paper, project_path, storage_path=storage_path)
            
            # Check if last message indicates an interrupted paper upload BEFORE PI initialization
            messages = dm.get_database_chat()
            if len(messages) > 0:
                last_msg = messages[-1]
                content = last_msg.get("content", "")
                # Check for pattern: "I am uploading X new papers..."
                match = re.match(r"I am uploading (\d+) new papers\.\.\.", content)
                if match:
                    print("Detected interrupted paper upload. Resuming ingest...")
                    num_files = int(match.group(1))
                    
                    # Ingest and process papers
                    dm.ingest_papers()
                    await dm.process_all_papers()
                    
                    # Update uploading message to Processed
                    for msg in reversed(messages):
                        if msg["content"] == content and msg.get("status") == "Pending":
                            msg["status"] = "Processed"
                            break
                    with DDB.at("chat").session() as (session, chat):
                        chat["messages"] = messages
                        session.write()
                    
                    # Add completion message (no IDs since we can't accurately track them after interruption)
                    # Include total paper count in the system
                    total_papers = len(dm.get_database_papers())
                    completion_msg = {
                        "content": f"Uploaded {num_files} papers. There are now {total_papers} papers in the system.\n\n**Warning:** These papers have been added to the database, but previous analyses have NOT been updated to include them. You must explicitly ask me to re-run any analysis if you want these new papers included.",
                        "role": "user",
                        "status": "Processed",
                        "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')
                    }
                    
                    dm.add_chat(completion_msg)
                    dm.update_last_chat("Processed")
                    print("Interrupted upload recovery complete.")
            
            pi = PrincipalInvestigator(dm)
            await pi.initialize(ingest=ingest)
            
            if time.time() - start > 5*60:
                dm.save_database()
                if sys.platform == "darwin":
                    subprocess.Popen(["say", "ScienceAI is ready"])
            while True:
                if message_queue.empty():
                    await asyncio.sleep(1)
                else:
                    message = message_queue.get()
                    if message.get("TERMINATE"):
                        print("Terminating backend")
                        break
                    elif message.get("INGEST"):
                        print("Ingesting papers...")
                        from datetime import datetime
                        
                        # Add processing message
                        processing_msg = {
                            "content": "Processing papers...",
                            "role": "system",
                            "status": "Pending",
                            "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')
                        }
                        dm.add_chat(processing_msg)
                        
                        dm.ingest_papers()
                        await dm.process_all_papers()
                        
                        # Mark processing message as processed
                        dm.update_last_chat("Processed")
                        
                        # Add completion message with paper count
                        total_papers = len(dm.get_database_papers())
                        completion_msg = {
                            "content": f"All {total_papers} papers have been loaded into the system.",
                            "role": "system",
                            "status": "Processed",
                            "time": datetime.now().strftime('%B %d, %Y %I:%M:%S %p %Z')
                        }
                        dm.add_chat(completion_msg)
                        
                        print("Ingestion complete.")
                        continue
                    elif message.get("ADD_PAPERS"):
                        print("Adding new papers...")
                        # 1. Add uploading message
                        if "uploading_msg" in message:
                            dm.add_chat(message["uploading_msg"])
                            dm.update_last_chat("Pending")
                        
                        # 2. Ingest papers
                        # Get existing paper IDs to determine which ones are new
                        existing_papers = dm.get_database_papers()
                        existing_ids = set(p["paper_id"] for p in existing_papers)
                        
                        # We can use ingest_papers() which scans the directory
                        # or we can be more specific if we passed file paths, but scanning is safer/easier
                        # given we just copied files to the folder.
                        all_paper_ids = dm.ingest_papers()
                        
                        new_ids = [pid for pid in all_paper_ids if pid not in existing_ids]
                        
                        # 3. Process papers
                        # We need to process them. process_all_papers() does this.
                        await dm.process_all_papers()
                        
                        # 4. Update uploading message to Processed
                        # We need to find it first.
                        if "uploading_msg" in message:
                            messages = dm.get_database_chat()
                            for msg in reversed(messages):
                                if msg["content"] == message["uploading_msg"]["content"] and msg["status"] == "Pending":
                                    msg["status"] = "Processed"
                                    break
                            with DDB.at("chat").session() as (session, chat):
                                chat["messages"] = messages
                                session.write()

                        # 5. Add completion message
                        if "completion_msg" in message:
                            # Get total paper count and update the message
                            total_papers = len(dm.get_database_papers())
                            # Replace the uploaded count with "uploaded X papers. There are now Y papers in the system."
                            import re as re_msg
                            original_content = message["completion_msg"]["content"]
                            upload_match = re_msg.match(r"Uploaded (\d+) papers\.", original_content)
                            if upload_match:
                                num_uploaded = upload_match.group(1)
                                new_prefix = f"Uploaded {num_uploaded} papers. There are now {total_papers} papers in the system."
                                message["completion_msg"]["content"] = original_content.replace(f"Uploaded {num_uploaded} papers.", new_prefix)
                            
                            # Append new IDs to the content
                            if new_ids:
                                # only include the first 10 digits of each ID
                                new_ids = [pid[:10] for pid in new_ids]
                                ids_str = ", ".join(new_ids)
                                message["completion_msg"]["content"] += f"\n\n**New Paper IDs:** {ids_str}"
                            
                            dm.add_chat(message["completion_msg"])
                            dm.update_last_chat("Processed")
                        
                        print("Add papers complete.")
                        continue
                    elif message.get("COMPRESS_CONTEXT"):
                        print("Compressing conversation context...")
                        await compress_conversation_context(dm)
                        print("Context compression complete.")
                        continue
                    elif stop_event.is_set():
                        print("Stop event set. Terminating backend")
                        break
                    start = time.time()
                    await pi.process_message(**message)
                    end = time.time()
                    if end - start > 10 and sys.platform == "darwin":
                        subprocess.Popen(["say", "New message from ScienceAI"])
                    dm.save_database()
        except Exception as e:
            print("Backend error")
            traceback.print_exc()
            if error_queue:
                error_queue.put(e)
            raise e

    asyncio.run(run_backend_async())

