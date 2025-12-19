import json
import os
import traceback

# Module-level flag to prevent double-starting compression
_compression_running = False

# Module-level flag for pause requests
_pause_requested = False


def is_pause_requested():
    """Check if a pause has been requested."""
    return _pause_requested


def set_pause_requested(value: bool):
    """Set the pause request flag."""
    global _pause_requested
    _pause_requested = value


def clear_pause_request():
    """Clear the pause request flag (called after pause completes)."""
    global _pause_requested
    _pause_requested = False


async def compress_conversation_context(dm, skip_user_message=False):
    """
    Compress the conversation by summarizing the first 50% of tool calls and tool responses.
    Each tool message is summarized into a few sentences and replaced with an assistant message.

    Args:
        dm: DatabaseManager instance
        skip_user_message: If True, don't add the user request message (used when resuming interrupted compression)
    """
    from datetime import datetime

    import dictdatabase as DDB

    from .llm import client, get_config

    # Add user request message as pending (marked internal so PI won't include in context)
    if not skip_user_message:
        user_request_msg = {
            "content": "🗜️ Please compress the conversation to free up context space.",
            "role": "user",
            "status": "Pending",
            "time": datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
            "internal": True,  # Flag to exclude from PI context
        }
        dm.add_chat(user_request_msg)

    messages = dm.get_database_chat()

    # Find tool call groups (assistant with tool_calls + their corresponding tool responses)
    # We must compress these together to avoid orphaned tool calls
    tool_groups = []  # List of lists: each inner list is [assistant_idx, tool_idx1, tool_idx2, ...]
    i = 0
    while i < len(messages):
        msg = messages[i]
        # Skip already compressed messages
        if msg.get("compressed"):
            i += 1
            continue

        # Look for assistant messages with tool_calls
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            group = [i]
            tool_call_ids = {tc["id"] for tc in msg["tool_calls"]}

            # Find all corresponding tool responses that follow
            j = i + 1
            while j < len(messages):
                next_msg = messages[j]
                if next_msg.get("role") == "tool":
                    tool_call_id = next_msg.get("tool_call_id")
                    if tool_call_id in tool_call_ids:
                        group.append(j)
                        tool_call_ids.discard(tool_call_id)
                    j += 1
                else:
                    break

            # Only add complete groups (all tool responses found)
            if len(tool_call_ids) == 0 and not msg.get("compressed"):
                tool_groups.append(group)
            i = j if j > i + 1 else i + 1
        elif msg.get("role") == "tool" and not msg.get("compressed"):
            # Standalone tool message (orphan) - compress individually
            tool_groups.append([i])
            i += 1
        else:
            i += 1

    if len(tool_groups) == 0:
        print("No tool message groups to compress")
        return

    # Get the first 50% of groups to compress (round up so odd numbers eventually reach 0)
    import math

    compress_count = math.ceil(len(tool_groups) / 2)

    groups_to_compress = tool_groups[:compress_count]
    # Flatten to get all indices
    indices_to_compress = [idx for group in groups_to_compress for idx in group]
    print(
        f"Compressing {len(groups_to_compress)} tool groups ({len(indices_to_compress)} messages) out of {len(tool_groups)} groups total"
    )

    # Process each message to compress
    for idx in sorted(indices_to_compress, reverse=True):  # Process in reverse to maintain indices
        msg = messages[idx]
        original_content = msg.get("content", "")
        original_role = msg.get("role", "")

        # Skip if content is too short
        if len(original_content) < 100:
            continue

        # Summarize with LLM
        try:
            summary_prompt = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes tool call results concisely. "
                    "Summarize the following tool output in 2-3 sentences, preserving key data points and findings. "
                    "Be factual and concise.",
                },
                {
                    "role": "user",
                    "content": f"Summarize this tool output:\n\n{original_content[:8000]}",  # Limit input size
                },
            ]

            response = client.chat.completions.create(
                model=get_config().default_fast_model, messages=summary_prompt, max_tokens=2000
            )

            summary = response.choices[0].message.content

            # Validate summary is not empty
            if not summary or len(summary.strip()) < 10:
                print(
                    f"Warning: Summary generation returned empty/short result for index {idx}, keeping truncated original"
                )
                # Keep a truncated version of the original instead of losing all info
                summary = f"[Summary failed - Original content preview]: {original_content[:2000]}..."
            else:
                print(f"Summary generated for index {idx}: {summary[:200]}...")

            # Extract metadata using regex - differentiate between delegate_research and run_python_code
            import re

            metadata_parts = []

            # Check if this is a delegate_research call (has "Response from X:" pattern)
            is_delegate_research = bool(re.search(r"Response from [^:]+:", original_content))

            # For python code detection, we need to check:
            # 1. The content itself (for tool responses)
            # 2. The preceding assistant message's tool_calls arguments (for run_python_code)

            # Build a combined text to search - include tool_call arguments if present
            searchable_content = original_content

            # If this is a tool response, look at the preceding assistant message for the code
            if original_role == "tool":
                # Find the tool_call_id
                tool_call_id = msg.get("tool_call_id", "")
                # Look backwards for the assistant message that called this tool
                for prev_idx in range(idx - 1, -1, -1):
                    prev_msg = messages[prev_idx]
                    if prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls"):
                        for tc in prev_msg["tool_calls"]:
                            if tc.get("id") == tool_call_id:
                                # Found it - check if it's run_python_code
                                func_name = tc.get("function", {}).get("name", "")
                                if func_name == "run_python_code":
                                    # Add the code arguments to searchable content
                                    args_str = tc.get("function", {}).get("arguments", "")
                                    searchable_content += "\n" + args_str
                                break
                        break

            # Also check if the message itself has tool_calls (for assistant messages)
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func_name = tc.get("function", {}).get("name", "")
                    if func_name == "run_python_code":
                        args_str = tc.get("function", {}).get("arguments", "")
                        searchable_content += "\n" + args_str

            # Check if this is a run_python_code call (has load_analyst_data pattern)
            is_python_call = bool(re.search(r"load_analyst_data\s*\(", searchable_content))

            if is_delegate_research:
                # For delegate_research: Extract analyst names from "Response from X Analyst:" patterns
                analyst_matches = re.findall(r"Response from ([^:]+):", original_content)
                analyst_names = list(set(analyst_matches))  # Remove duplicates

                # Extract collection names from CSV file paths like viewCSV('download/.../Name_dates.csv')
                # Remove the timestamp portion (_YYYY-MM-DD_HH_MM_SS.csv)
                csv_matches = re.findall(r"viewCSV\(['\"]download/[^'\"]*?/([^/'\"]+)\.csv['\"]", original_content)
                collection_names = []
                for csv_name in csv_matches:
                    # Remove timestamp suffix like _2025-12-12_14_21_26
                    clean_name = re.sub(r"_\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}$", "", csv_name)
                    if clean_name not in collection_names:
                        collection_names.append(clean_name)

                if analyst_names:
                    metadata_parts.append(f"**Analyst(s):** {', '.join(analyst_names)}")
                if collection_names:
                    metadata_parts.append(f"**Data Extraction(s):** {', '.join(collection_names)}")

            elif is_python_call:
                # For run_python_code: Extract from load_analyst_data('Analyst Name', 'Collection Name')
                load_matches = re.findall(
                    r"load_analyst_data\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)", searchable_content
                )
                if load_matches:
                    analysts_from_python = list({m[0] for m in load_matches})
                    collections_from_python = list({m[1] for m in load_matches})

                    if analysts_from_python:
                        metadata_parts.append(f"**Used Python to access data from:** {', '.join(analysts_from_python)}")
                    if collections_from_python:
                        metadata_parts.append(f"**Data Extraction(s) accessed:** {', '.join(collections_from_python)}")

                # Extract file names from quoted strings ending in common extensions
                # Use non-capturing group (?:...) so findall returns strings, not tuples
                file_extensions = r"\.(?:csv|png|jpg|jpeg|gif|json|html|pdf|txt|xlsx|zip)"
                # Match both single and double quoted strings ending in file extensions
                file_matches = re.findall(
                    r"['\"]([^'\"]*" + file_extensions + r")['\"]", searchable_content, re.IGNORECASE
                )
                if file_matches:
                    # Get just the base names
                    import os as os_module

                    file_names = list({os_module.path.basename(f) for f in file_matches})
                    if file_names:
                        metadata_parts.append(f"**Files Referenced:** {', '.join(file_names)}")

            metadata_section = "\n".join(metadata_parts) if metadata_parts else ""

            # Create replacement message
            compressed_content = (
                f"[📦 Compressed Tool Output]\n\n"
                f"{summary}\n\n" + (f"{metadata_section}\n\n" if metadata_section else "") + f"---\n"
                f"*This message was automatically summarized to free up context space. "
                f"Original content was {len(original_content)} characters.*"
            )

            # Replace the message in place
            # NOTE: Role stays "assistant" in DB for proper display
            # On-the-fly conversion to "user" happens in llm_providers.py when thinking is enabled
            messages[idx] = {
                "content": compressed_content,
                "role": "assistant",
                "status": "Processed",
                "time": msg.get("time", datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z")),
                "compressed": True,
                "original_role": original_role,
            }

            print(f"Compressed message at index {idx}: {len(original_content)} -> {len(compressed_content)} chars")

        except Exception as e:
            print(f"Failed to compress message at index {idx}: {e}")
            continue

    # Remove the CONTEXTLIMITREACHED message if present
    messages = [m for m in messages if "CONTEXTLIMITREACHED" not in m.get("content", "")]

    # Heal orphaned tool calls - remove assistant messages with tool_calls that don't have matching responses
    def heal_orphaned_tool_calls(msgs):
        """Remove assistant messages with tool_calls that are missing their tool responses."""
        healed = []
        skip_indices = set()

        for i, msg in enumerate(msgs):
            if i in skip_indices:
                continue

            # If it's an assistant message with tool_calls
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
                required_ids = {tc["id"] for tc in tool_calls}
                found_ids = set()

                # Look ahead for corresponding tool results
                j = i + 1
                tool_result_indices = []
                while j < len(msgs):
                    next_msg = msgs[j]
                    if next_msg.get("role") == "tool":
                        tool_call_id = next_msg.get("tool_call_id")
                        if tool_call_id in required_ids:
                            found_ids.add(tool_call_id)
                            tool_result_indices.append(j)
                        j += 1
                    else:
                        break

                # If we're missing any tool responses, skip this message and its partial results
                if not required_ids.issubset(found_ids):
                    print(f"Healing: Removing orphaned tool call at index {i} (missing {required_ids - found_ids})")
                    skip_indices.add(i)
                    for idx in tool_result_indices:
                        skip_indices.add(idx)
                    continue

            healed.append(msg)

        return healed

    messages = heal_orphaned_tool_calls(messages)

    # Mark user's compression request as processed and add assistant response
    # Find the user request message and mark it processed
    for msg in reversed(messages):
        if (
            msg.get("content") == "🗜️ Please compress the conversation to free up context space."
            and msg.get("status") == "Pending"
        ):
            msg["status"] = "Processed"
            break

    # Add assistant response as pending (marked internal so PI won't include in context)
    compression_response = {
        "content": f"🗜️ **Conversation Compressed**\n\nI've summarized {len(indices_to_compress)} older tool outputs to free up context space. You can now continue the discussion.",
        "role": "assistant",
        "status": "Pending",
        "time": datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
        "internal": True,  # Flag to exclude from PI context
    }
    messages.append(compression_response)

    # Save the compressed messages back to the database
    with DDB.at("chat").session() as (session, chat):
        chat["messages"] = messages
        session.write()

    # Mark the assistant response as processed
    dm.update_last_chat("Processed")

    print(f"Context compression complete. Compressed {len(indices_to_compress)} messages.")


def compact_analyst_interaction(dm, analyst_name: str, analysis_notes: dict) -> bool:
    """
    Compact a specific analyst delegation and response pair.

    This performs targeted compaction using the PI's investigation notes,
    rather than the general LLM-based summarization.

    Args:
        dm: DatabaseManager instance
        analyst_name: Name of the analyst whose interaction to compact
        analysis_notes: Dict containing:
            - investigation_summary: PI's summary of what was found/done
            - normalization_applied: What normalizations were applied
            - copies_reconciled: List of copy analyst names
            - all_analyst_names: All analyst names including copies

    Returns:
        True if messages were compacted, False otherwise
    """
    import re
    from datetime import datetime

    import dictdatabase as DDB

    messages = dm.get_database_chat()
    all_analyst_names = analysis_notes.get("all_analyst_names", [analyst_name])

    # Build pattern to match delegate_research calls for these analysts
    # Look for tool calls that contain the analyst name
    indices_to_compact = []

    for idx, msg in enumerate(messages):
        # Skip already compressed messages
        if msg.get("compressed") or msg.get("analyzed"):
            continue

        # Check for assistant message with delegate_research tool call
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tool_call in msg.get("tool_calls", []):
                func = tool_call.get("function", {})
                if func.get("name") == "delegate_research":
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                        called_name = args.get("name", "")
                        # Check if this matches any of our analyst names
                        if called_name in all_analyst_names:
                            indices_to_compact.append(idx)
                            # Also find the corresponding tool response(s)
                            tool_call_id = tool_call.get("id")
                            if tool_call_id:
                                # Look for matching tool response
                                for resp_idx in range(idx + 1, len(messages)):
                                    resp_msg = messages[resp_idx]
                                    if resp_msg.get("role") == "tool" and resp_msg.get("tool_call_id") == tool_call_id:
                                        indices_to_compact.append(resp_idx)
                                        break
                    except (json.JSONDecodeError, TypeError):
                        continue

        # Also check for tool responses that mention "Response from [analyst_name]:"
        if msg.get("role") == "tool" and not msg.get("compressed"):
            content = msg.get("content", "")
            for name in all_analyst_names:
                if f"Response from {name}:" in content:
                    if idx not in indices_to_compact:
                        indices_to_compact.append(idx)
                    break

    if not indices_to_compact:
        return False

    # Remove duplicates and sort
    indices_to_compact = sorted(set(indices_to_compact))

    # Build the compressed message content
    investigation_summary = analysis_notes.get("investigation_summary", "Investigation complete.")
    normalization_applied = analysis_notes.get("normalization_applied", "")
    copies_reconciled = analysis_notes.get("copies_reconciled", [])
    # Defensive handling: ensure copies_reconciled is a list, not a string
    if isinstance(copies_reconciled, str):
        import ast

        try:
            copies_reconciled = ast.literal_eval(copies_reconciled)
            if not isinstance(copies_reconciled, list):
                copies_reconciled = [copies_reconciled]
        except (ValueError, SyntaxError):
            copies_reconciled = [copies_reconciled] if copies_reconciled else []

    # Calculate original size for reporting
    original_size = sum(len(messages[idx].get("content", "") or "") for idx in indices_to_compact)

    # Build metadata section
    metadata_parts = [f"**Analyst**: {analyst_name}"]
    if copies_reconciled:
        metadata_parts.append(f"**Copies reconciled**: {', '.join(copies_reconciled)}")
    if normalization_applied:
        metadata_parts.append(f"**Normalizations**: {normalization_applied}")

    # Extract collection names from the original content
    collection_names = []
    for idx in indices_to_compact:
        content = messages[idx].get("content", "") or ""
        # Match viewCSV patterns
        csv_matches = re.findall(r"viewCSV\(['\"]download/[^'\"]*?/([^/'\"]+)\.csv['\"]\)", content)
        for csv_name in csv_matches:
            # Remove timestamp suffix
            clean_name = re.sub(r"_\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}$", "", csv_name)
            if clean_name not in collection_names:
                collection_names.append(clean_name)

    if collection_names:
        metadata_parts.append(f"**Data collections**: {', '.join(collection_names)}")

    metadata_section = "\n".join(metadata_parts)

    # Build data access reference section showing which copy owns which collection
    data_access_lines = []
    all_analyst_names = analysis_notes.get("all_analyst_names", [analyst_name])

    for copy_name in all_analyst_names:
        try:
            analyst_meta = dm.get_analyst_metadata(copy_name)
            copy_collections = []
            for tool in analyst_meta.get("tools", []):
                tool_name = tool.get("tool_name", "")
                # Remove timestamp suffix to get clean collection name
                clean_name = re.sub(r"_\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}$", "", tool_name)
                if clean_name:
                    copy_collections.append(clean_name)

            if copy_collections:
                data_access_lines.append(f"**{copy_name}**: {', '.join(copy_collections)}")
                for coll in copy_collections:
                    data_access_lines.append(f"  → `pd.read_csv(load_analyst_data('{copy_name}', '{coll}'))`")
        except (ValueError, KeyError):
            # Analyst not found or no tools - skip silently
            pass

    data_access_section = ""
    if data_access_lines:
        data_access_section = (
            "**Data Access Reference** (use in `run_python_code` to access data or to make corrections/normalization that may be discovered later):\n"
            + "\n".join(data_access_lines)
            + "\n\n"
        )

    compressed_content = (
        f"[📊 Analyzed Delegation Result]\n\n"
        f"{metadata_section}\n\n"
        f"---\n\n"
        f"**Investigation Summary**:\n{investigation_summary}\n\n"
        f"---\n\n"
        f"{data_access_section}"
        f"*This delegation was analyzed and compacted. Original content was {original_size} characters.*"
    )

    # Instead of replacing the first message in-place (which can break thinking block structure),
    # we:
    # 1. Insert a new compressed message at the position of the first message
    # 2. Mark ALL original messages for removal (including the first)
    first_idx = indices_to_compact[0]

    new_compressed_message = {
        "content": compressed_content,
        "role": "assistant",
        "status": "Processed",
        "time": datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
        "compressed": True,
        "analyzed": True,
        "analysis_notes": analysis_notes,
    }

    # Mark ALL messages in indices_to_compact for removal
    for idx in indices_to_compact:
        messages[idx]["_remove"] = True

    # Filter out removed messages
    messages = [m for m in messages if not m.get("_remove")]

    # Insert the new compressed message at the position where the first removed message was
    # Since we removed messages, we need to calculate the new position
    # The new position is first_idx minus the number of removed messages before it (which is 0)
    messages.insert(first_idx, new_compressed_message)

    # Save the updated messages
    with DDB.at("chat").session() as (session, chat):
        chat["messages"] = messages
        session.write()

    print(f"Compacted analyst interaction for {analyst_name}: {len(indices_to_compact)} messages -> 1 message")
    return True


def run_backend(folder, project_path, storage_path, message_queue, stop_event, ingest=True, error_queue=None):
    import asyncio

    async def run_backend_async():
        try:
            from .llm import update_stop_event

            update_stop_event(stop_event)
            import re
            import subprocess  # nosec B404
            import sys
            import time
            from datetime import datetime

            import dictdatabase as DDB

            from .database_manager import DatabaseManager
            from .principal_investigator import PrincipalInvestigator
            from .process_paper import process_paper

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
                        "time": datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
                    }

                    dm.add_chat(completion_msg)
                    dm.update_last_chat("Processed")
                    print("Interrupted upload recovery complete.")

            # Check for interrupted compression - look for pending compression request
            messages = dm.get_database_chat()
            for msg in messages:
                if (
                    msg.get("content") == "🗜️ Please compress the conversation to free up context space."
                    and msg.get("status") == "Pending"
                ):
                    print("Detected interrupted compression. Resuming...")
                    await compress_conversation_context(dm, skip_user_message=True)
                    print("Interrupted compression recovery complete.")
                    break

            pi = PrincipalInvestigator(dm)
            await pi.initialize(ingest=ingest)

            if time.time() - start > 5 * 60:
                dm.save_database()
                if sys.platform == "darwin":
                    subprocess.Popen(["/usr/bin/say", "ScienceAI is ready"])  # nosec B603
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
                            "time": datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
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
                            "time": datetime.now().strftime("%B %d, %Y %I:%M:%S %p %Z"),
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
                        existing_ids = {p["paper_id"] for p in existing_papers}

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
                                message["completion_msg"]["content"] = original_content.replace(
                                    f"Uploaded {num_uploaded} papers.", new_prefix
                                )

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
                        global _compression_running
                        if _compression_running:
                            print("Compression already in progress, skipping...")
                            continue
                        _compression_running = True
                        try:
                            print("Compressing conversation context...")
                            await compress_conversation_context(dm)
                            print("Context compression complete.")
                        finally:
                            _compression_running = False
                        continue
                    elif message.get("UNDO_LAST_REQUEST"):
                        print("Undoing last request...")
                        last_user_idx = message.get("last_user_idx")
                        analysts_to_delete = message.get("analysts_to_delete", [])

                        # Remove analysts
                        for analyst_name in analysts_to_delete:
                            try:
                                dm.remove_analyst(analyst_name)
                                print(f"Removed analyst: {analyst_name}")
                            except Exception as e:
                                print(f"Failed to remove analyst {analyst_name}: {e}")

                        # Revert chat history
                        dm.revert_chat_from_index(last_user_idx)

                        print("Undo complete.")
                        continue
                    elif message.get("RESET_CONVERSATION"):
                        print("Resetting conversation...")
                        import shutil

                        # Remove all analysts
                        try:
                            all_analysts = dm.get_all_analysts()
                            for analyst in all_analysts:
                                analyst_name = analyst.get("name")
                                if analyst_name:
                                    try:
                                        dm.remove_analyst(analyst_name)
                                        print(f"Reset: Removed analyst {analyst_name}")
                                    except Exception as e:
                                        print(f"Reset: Failed to remove analyst {analyst_name}: {e}")
                        except Exception as e:
                            print(f"Reset: Error getting analysts: {e}")

                        # Clear pi_generated directory
                        pi_generated_path = os.path.join(dm.project_path, "pi_generated")
                        if os.path.exists(pi_generated_path):
                            try:
                                shutil.rmtree(pi_generated_path)
                                os.makedirs(pi_generated_path)
                                print("Reset: Cleared pi_generated directory")
                            except Exception as e:
                                print(f"Reset: Failed to clear pi_generated: {e}")

                        # Clear analyst tool tracker directory where data extraction request tracker exports are stored

                        # Clear chat messages but keep first 2 (intro messages)
                        dm.revert_chat_from_index(2)

                        print("Reset complete.")
                        continue
                    elif message.get("SET_PARALLEL_CALLS"):
                        updates = message.get("count", 1)
                        try:
                            count = int(updates)
                            # Enforce bounds
                            if count < 1:
                                count = 1
                            if count > 3:
                                count = 3
                            pi.n_parallel_calls = count
                            # Persist the setting to the database for restarts
                            dm.set_project_setting("n_parallel_calls", count)
                            print(f"Updated parallel calls to {count} (persisted)")
                        except ValueError:
                            print(f"Invalid parallel calls count: {updates}")
                        continue
                    elif stop_event.is_set():
                        print("Stop event set. Terminating backend")
                        break
                    start = time.time()
                    await pi.process_message(**message)
                    end = time.time()
                    if end - start > 10 and sys.platform == "darwin":
                        subprocess.Popen(["/usr/bin/say", "New message from ScienceAI"])  # nosec B603
                    dm.save_database()
        except asyncio.CancelledError:
            # Expected when shutdown cancels tasks - exit gracefully
            print("Backend async tasks cancelled during shutdown")
        except Exception as e:
            print("Backend error")
            traceback.print_exc()
            if error_queue:
                error_queue.put(e)
            raise e

    async def run_with_cleanup():
        """Wrapper that ensures all tasks are cancelled on shutdown."""
        try:
            await run_backend_async()
        finally:
            # Cancel all remaining tasks to prevent "activity after close"
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if tasks:
                print(f"Cancelling {len(tasks)} pending async tasks...")
                for task in tasks:
                    task.cancel()
                # Wait for all tasks to complete their cancellation
                await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run_with_cleanup())
