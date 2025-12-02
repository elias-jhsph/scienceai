import traceback


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
            start = time.time()
            dm = DatabaseManager(folder, process_paper, project_path, storage_path=storage_path)
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
                        
                        # Add completion message
                        completion_msg = {
                            "content": "All papers have been loaded into the system.",
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
                            # Update the paper count in the completion message if needed
                            # But the message was pre-formatted.
                            
                            # Append new IDs to the content
                            if new_ids:
                                # only incldue the first 10 digits of each ID
                                new_ids = [pid[:10] for pid in new_ids]
                                ids_str = ", ".join(new_ids)
                                message["completion_msg"]["content"] += f"\n\n**New Paper IDs:** {ids_str}"
                            
                            dm.add_chat(message["completion_msg"])
                            dm.update_last_chat("Processed")
                        
                        print("Add papers complete.")
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

