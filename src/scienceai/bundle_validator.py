"""
Analytic Bundle Validator Agent

An LLM-powered agent that validates analytic bundles before delivery.
Uses a loop with Python execution to thoroughly check data quality.
"""

import io
import json
import logging
import os
import shutil
import tempfile
import time
import traceback
import zipfile

from .llm import MODEL_VISION, client, get_config, get_model_for_role

logger = logging.getLogger(__name__)

# Safety limits
MAX_ITERATIONS = 100  # Max tool calls before forcing completion
MAX_NO_TOOL_RESPONSES = 5  # Max consecutive responses without tool calls before forcing
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

# Model for validation - uses the configured default model
# This will be resolved at runtime from the config

# Load the prompt
path_to_app = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(path_to_app, "bundle_validator_prompt.txt")) as f:
        VALIDATOR_SYSTEM_PROMPT = f.read()
except Exception:
    VALIDATOR_SYSTEM_PROMPT = "You are a bundle validator agent. Check data quality thoroughly."


def validate_bundle(zip_path):
    """
    Main entry point for bundle validation.

    Runs an LLM agent in a loop that can execute Python code to validate the bundle.
    The agent decides what to check and signals when done.

    Args:
        zip_path: Path to the analysis bundle zip file

    Returns:
        String with validation feedback (pass/fail with details)
    """
    start_time = time.time()

    # Check zip exists
    if not os.path.exists(zip_path):
        return "## ❌ BUNDLE VALIDATION FAILED\n\nBundle not found: " + zip_path

    # Check zip size
    zip_size = os.path.getsize(zip_path)
    if zip_size > MAX_FILE_SIZE_BYTES:
        return f"## ❌ BUNDLE VALIDATION FAILED\n\nBundle too large ({zip_size / 1024 / 1024:.1f} MB)"

    # Create temp workspace
    workspace = tempfile.mkdtemp(prefix="bundle_validation_")

    try:
        # Unzip
        with zipfile.ZipFile(zip_path, "r") as zf:
            file_list = zf.namelist()
            zf.extractall(workspace)

        # Run the agent loop (no timeout - let it work)
        result = _run_validator_agent(workspace, file_list, zip_path)

    except zipfile.BadZipFile:
        result = "## ❌ BUNDLE VALIDATION FAILED\n\nInvalid zip file - cannot extract"
    except Exception as e:
        result = f"## ❌ BUNDLE VALIDATION FAILED\n\nSystem error: {e!s}\n\n{traceback.format_exc()}"
    finally:
        # Cleanup workspace
        shutil.rmtree(workspace, ignore_errors=True)

    elapsed = time.time() - start_time
    result += f"\n\n---\n_Validation completed in {elapsed:.1f} seconds_"

    return result


def _scan_failed_collections(workspace):
    """Pre-scan all CSVs for failed collections and return summary"""
    import pandas as pd

    results = []
    csv_files = []

    for root, _dirs, files in os.walk(workspace):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    for csv_path in csv_files[:50]:  # Limit to first 50 CSVs
        try:
            df = pd.read_csv(csv_path, low_memory=False, nrows=5000)
            rel_path = os.path.relpath(csv_path, workspace)

            # Check for failed_collection column
            if "failed_collection" in df.columns:
                failed_rows = df[df["failed_collection"].notna() & (df["failed_collection"] != "")]
                if len(failed_rows) > 0:
                    for _, row in failed_rows.iterrows():
                        study_id = row.get("id", row.get("paper_id", row.get("study_id", "unknown")))
                        reason = str(row["failed_collection"])[:150]
                        results.append(f"  - [{rel_path}] Study '{study_id}': {reason}")

            # Also check for text containing "failed_collection"
            for col in df.columns:
                if df[col].dtype == "object":
                    mask = df[col].astype(str).str.contains("failed_collection", case=False, na=False)
                    if mask.any():
                        count = mask.sum()
                        results.append(
                            f"  - [{rel_path}] Column '{col}' has {count} rows containing 'failed_collection' text"
                        )

        except Exception as e:
            results.append(f"  - [{csv_path}] Error reading: {str(e)[:100]}")

    if results:
        return "**⚠️ FAILED EXTRACTIONS DETECTED:**\n" + "\n".join(results)
    else:
        return "**✓ No failed collections detected in CSV files.**"


def _check_bundle_structure(workspace):
    """Pre-check bundle structure and return summary"""

    expected_structure = {
        "directories": [
            ("01_data_processing", "Data processing scripts"),
            ("02_analytic_data", "Clean analytic CSV files"),
            ("03_data_dictionary", "Data dictionary documentation"),
            ("04_analysis", "Analysis code"),
            ("05_outputs", "Generated outputs (figures, tables)"),
        ],
        "files": [
            ("README.md", "Documentation and instructions"),
            ("requirements.txt", "Python dependencies"),
        ],
    }

    found = []
    missing = []

    # Check directories
    for dir_name, description in expected_structure["directories"]:
        path = os.path.join(workspace, dir_name)
        # Also check one level nested (common zip structure)
        nested = None
        for item in os.listdir(workspace):
            nested_path = os.path.join(workspace, item, dir_name)
            if os.path.isdir(nested_path):
                nested = nested_path
                break

        if os.path.isdir(path) or nested:
            found.append(f"  ✓ {dir_name}/ - {description}")
        else:
            missing.append(f"  ✗ {dir_name}/ - {description}")

    # Check files
    for file_name, description in expected_structure["files"]:
        path = os.path.join(workspace, file_name)
        nested = None
        for item in os.listdir(workspace):
            nested_path = os.path.join(workspace, item, file_name)
            if os.path.isfile(nested_path):
                nested = nested_path
                break

        if os.path.isfile(path) or nested:
            found.append(f"  ✓ {file_name} - {description}")
        else:
            missing.append(f"  ✗ {file_name} - {description}")

    result = "**BUNDLE STRUCTURE CHECK:**\n"
    if found:
        result += "Found:\n" + "\n".join(found) + "\n"
    if missing:
        result += "Missing:\n" + "\n".join(missing)
    else:
        result += "All expected components present."

    return result


def _check_effect_direction_consistency(workspace):
    """
    Two checks:
    1. Statistical outliers (z-score > 2)
    2. Values on opposite side of 1.0 from majority (semantic outliers for ratios)
    """
    import numpy as np
    import pandas as pd

    stat_results = []
    direction_results = []

    csv_files = []
    for root, _dirs, files in os.walk(workspace):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    for csv_path in csv_files[:30]:
        try:
            df = pd.read_csv(csv_path, low_memory=False, nrows=1000)
            rel_path = os.path.relpath(csv_path, workspace)

            for col in df.columns:
                if df[col].dtype not in ["float64", "int64", "float32", "int32"]:
                    continue

                values = df[col].dropna()
                if len(values) < 5:
                    continue

                mean = values.mean()
                std = values.std()

                # Check 1: Statistical outliers (z-score)
                if std > 0 and not np.isnan(std):
                    z_scores = (values - mean) / std
                    outlier_mask = np.abs(z_scores) > 2
                    outlier_indices = values[outlier_mask].index.tolist()

                    if len(outlier_indices) > 0 and len(outlier_indices) <= 5:
                        stat_results.append(f"\n**{rel_path}** column `{col}`:")
                        stat_results.append(f"  mean={mean:.3f}, std={std:.3f}")
                        stat_results.append(f"  {len(outlier_indices)} outlier(s) with |z| > 2:")

                        for idx in outlier_indices[:5]:
                            val = df.loc[idx, col]
                            z = (val - mean) / std
                            stat_results.append(f"  - row {idx}: {val:.3f} (z={z:.1f})")

                # Check 2: Opposite side of 1.0 from majority
                above_1 = (values > 1).sum()
                below_1 = (values < 1).sum()
                total = len(values)

                if above_1 > total * 0.6 and below_1 > 0 and below_1 < total * 0.3:
                    outlier_rows = df[df[col] < 1].head(5)
                    direction_results.append(f"\n**{rel_path}** column `{col}`:")
                    direction_results.append(f"  {above_1}/{total} values > 1, but {below_1} are < 1:")
                    for idx, row in outlier_rows.iterrows():
                        direction_results.append(f"  - row {idx}: {row[col]:.3f}")

                elif below_1 > total * 0.6 and above_1 > 0 and above_1 < total * 0.3:
                    outlier_rows = df[df[col] > 1].head(5)
                    direction_results.append(f"\n**{rel_path}** column `{col}`:")
                    direction_results.append(f"  {below_1}/{total} values < 1, but {above_1} are > 1:")
                    for idx, row in outlier_rows.iterrows():
                        direction_results.append(f"  - row {idx}: {row[col]:.3f}")

        except Exception:  # nosec
            # Ignore errors during bundle validation
            pass

    output = []
    if stat_results:
        output.append("**⚠️ STATISTICAL OUTLIERS (|z-score| > 2):**")
        output.extend(stat_results)
    if direction_results:
        output.append("\n**⚠️ DIRECTION OUTLIERS (opposite side of 1.0 from majority):**")
        output.extend(direction_results)

    if output:
        return "\n".join(output)
    else:
        return "**✓ No numeric outliers detected.**"


def _run_validator_agent(workspace, file_list, original_zip_path):
    """
    Run the validator agent in a loop until it signals completion.

    The agent has access to:
    1. run_python_code - Execute Python in the workspace context
    2. report_validation_result - Signal completion with pass/fail
    """

    # Run pre-checks and gather info for the agent
    try:
        failed_collection_scan = _scan_failed_collections(workspace)
    except Exception as e:
        failed_collection_scan = f"**Failed extraction scan error:** {e!s}"

    try:
        structure_check = _check_bundle_structure(workspace)
    except Exception as e:
        structure_check = f"**Structure check error:** {e!s}"

    try:
        effect_direction_check = _check_effect_direction_consistency(workspace)
    except Exception as e:
        effect_direction_check = f"**Effect direction check error:** {e!s}"

    # Tools available to the agent
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_python_code",
                "description": "Execute Python code in the bundle workspace. You have access to pandas, numpy, os, json, etc. "
                f"The workspace directory is: {workspace}\n"
                "Use this to read files, analyze data, check for issues, run the bundle's scripts, etc. "
                "Print results to see them.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_image",
                "description": "Analyze an image file (PNG, JPG, etc.) in the bundle using a vision model. "
                "Use this to check generated figures for quality, readability, correct labels, etc. "
                "Provide the full path to the image file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "Full path to the image file to analyze"},
                        "check_for": {
                            "type": "string",
                            "description": "Optional: specific things to check for (e.g., 'verify axis labels match data dictionary', 'check if forest plot shows correct direction')",
                        },
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "report_validation_result",
                "description": "Call this when you have completed validation to report your findings. "
                "This ends the validation loop.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "passed": {
                            "type": "boolean",
                            "description": "True if bundle passes validation, False if there are blocking issues",
                        },
                        "errors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of critical errors that MUST be fixed before delivery",
                        },
                        "warnings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of warnings that SHOULD be reviewed but don't block delivery",
                        },
                        "notes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Informational notes about what was checked",
                        },
                        "summary": {"type": "string", "description": "Brief summary of the validation for the PI"},
                    },
                    "required": ["passed", "errors", "warnings", "notes", "summary"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    # Initial message with context
    file_list_str = "\n".join(f"  - {f}" for f in file_list[:100])
    if len(file_list) > 100:
        file_list_str += f"\n  ... and {len(file_list) - 100} more files"

    initial_message = f"""You are validating an analytic bundle before delivery.

**Bundle location:** {workspace}

**Files in bundle:**
{file_list_str}

---

## PRE-SCAN RESULTS

{structure_check}

{failed_collection_scan}

{effect_direction_check}

---

## YOUR TASK

Investigate any issues flagged above. For numeric outliers, load the file and understand what those columns represent - outliers might indicate data coding errors.

Check:
1. Failed extractions are handled (removed + documented, or recovered)
2. Data makes sense (investigate any flagged outliers)
3. Code runs without errors
4. Documentation is complete

Use run_python_code to inspect files and run scripts.
Call report_validation_result when done."""

    messages = [{"role": "system", "content": VALIDATOR_SYSTEM_PROMPT}, {"role": "user", "content": initial_message}]

    iteration = 0
    validation_complete = False
    final_result = None
    consecutive_no_tools = 0

    logger.info("Starting validation loop...")

    while not validation_complete and iteration < MAX_ITERATIONS:
        iteration += 1

        try:
            response = client.chat.completions.create(
                model=get_config().default_model, messages=messages, tools=tools, tool_choice="auto"
            )

            assistant_message = response.choices[0].message

            # Build proper message to append
            msg_to_append = {"role": "assistant", "content": assistant_message.content or ""}
            if assistant_message.tool_calls:
                tool_calls_list = []
                for tc in assistant_message.tool_calls:
                    tc_entry = {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    # Preserve thought_signature if present (required for Gemini 3 Pro with thinking)
                    if hasattr(tc, "thought_signature") and tc.thought_signature:
                        tc_entry["thought_signature"] = tc.thought_signature
                    tool_calls_list.append(tc_entry)
                msg_to_append["tool_calls"] = tool_calls_list
            # Preserve thinking block if present
            if hasattr(assistant_message, "thinking") and assistant_message.thinking:
                msg_to_append["thinking"] = assistant_message.thinking
            messages.append(msg_to_append)

            if not assistant_message.tool_calls:
                consecutive_no_tools += 1
                logger.info(f"Iteration {iteration}: No tool calls (consecutive: {consecutive_no_tools})")

                # If too many consecutive no-tool responses, force completion
                if consecutive_no_tools >= MAX_NO_TOOL_RESPONSES:
                    logger.warning("Too many no-tool responses, forcing completion request")
                    messages.append(
                        {
                            "role": "user",
                            "content": "You must now call report_validation_result to complete validation. "
                            "Summarize what you've found so far and report your findings.",
                        }
                    )
                else:
                    # Add a nudge to use tools
                    messages.append(
                        {
                            "role": "user",
                            "content": "Please continue validation using the tools. Call run_python_code to check things, "
                            "or report_validation_result when you're done.",
                        }
                    )
                continue

            # Reset counter when tools are used
            consecutive_no_tools = 0

            # Process tool calls
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                logger.info(f"Iteration {iteration}: Calling {function_name}")

                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": f"Error parsing arguments: {e}"}
                    )
                    continue

                if function_name == "run_python_code":
                    result = _execute_python(arguments.get("code", ""), workspace)
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

                elif function_name == "analyze_image":
                    result = _analyze_image(arguments.get("image_path", ""), arguments.get("check_for", ""))
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

                elif function_name == "report_validation_result":
                    validation_complete = True
                    final_result = _format_validation_result(arguments)
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": "Validation result recorded."}
                    )
                    logger.info(f"Validation complete after {iteration} iterations")
                    break
                else:
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": f"Unknown function: {function_name}"}
                    )

        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {e}")
            messages.append(
                {"role": "user", "content": f"Error occurred: {e!s}. Please continue or report your findings."}
            )

    if not validation_complete:
        logger.warning(f"Did not complete after {iteration} iterations")
        return (
            "## ⚠️ BUNDLE VALIDATION INCOMPLETE\n\n"
            f"Agent did not complete validation after {iteration} iterations.\n"
            "Please review the bundle manually."
        )

    return final_result


def _execute_python(code, workspace):
    """Execute Python code in the workspace context (similar to PI's run_python_code)"""
    import contextlib

    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Track files before execution
    existing_files = set()
    if os.path.exists(workspace):
        existing_files = set(os.listdir(workspace))

    # Helper to show plots if matplotlib is used
    def show_plot():
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if plt.get_fignums():
                from datetime import datetime

                filename = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(filename, dpi=150, bbox_inches="tight")
                plt.close("all")
                logger.info(f"Plot saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving plot: {e}")

    # Build execution environment
    env = {
        "__builtins__": __builtins__,
        "os": os,
        "json": json,
        "workspace": workspace,
        "print": lambda *args, **kwargs: print(*args, file=stdout_capture, **kwargs),
        "show_plot": show_plot,
    }

    # Try to import common data science libraries
    try:
        import pandas as pd

        env["pd"] = pd
    except ImportError:
        pass

    try:
        import numpy as np

        env["np"] = np
    except ImportError:
        pass

    # Execute code
    original_cwd = os.getcwd()
    try:
        os.chdir(workspace)

        # Force Agg backend to avoid display issues
        try:
            import matplotlib

            matplotlib.use("Agg")
        except ImportError:
            pass

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code, env)  # nosec B102

    except Exception:
        traceback.print_exc(file=stderr_capture)
    finally:
        os.chdir(original_cwd)

    output = stdout_capture.getvalue()
    errors = stderr_capture.getvalue()

    # Detect new files created
    current_files = set()
    if os.path.exists(workspace):
        current_files = set(os.listdir(workspace))
    new_files = current_files - existing_files

    # Build result message
    result_msg = ""
    if output:
        result_msg += f"Output:\n{output}\n"
    if errors:
        result_msg += f"Errors:\n{errors}\n"

    if new_files:
        result_msg += f"\nNew files created ({len(new_files)}):\n"
        for filename in sorted(new_files):
            file_path = os.path.join(workspace, filename)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            result_msg += f"  - {filename} ({file_size / 1024:.1f} KB)\n"

    if not result_msg:
        result_msg = "Code executed successfully (no output)."

    return result_msg[:15000]  # Limit output size


def _analyze_image(image_path, check_for=""):
    """Analyze an image using a vision model"""

    if not os.path.exists(image_path):
        return f"Error: Image file not found: {image_path}"

    try:
        import base64

        from PIL import Image

        # Load the image and get metadata
        img = Image.open(image_path)
        width, height = img.size
        format_name = img.format or "PNG"
        file_size = os.path.getsize(image_path)

        # Check file size limit (10MB)
        if file_size > 10 * 1024 * 1024:
            return f"Error: Image too large ({file_size / 1024 / 1024:.1f} MB). Max 10MB."

        # Encode image as base64
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        # Build the analysis prompt
        analysis_prompt = f"Please analyze this image ({os.path.basename(image_path)})."
        if check_for:
            analysis_prompt += f"\n\nSpecifically check for: {check_for}"

        vision_messages = [
            {
                "role": "system",
                "content": "You are an expert data visualization and figure critic validating an analysis bundle. "
                "Analyze the provided image for:\n"
                "1. Quality and clarity - are labels readable? is text overlapping?\n"
                "2. Data accuracy - do axis labels, legends, and titles make sense?\n"
                "3. Scientific correctness - for plots like forest plots, is the direction/interpretation clear?\n"
                "4. Completeness - are there missing labels, legends, or annotations?\n"
                "Be concise but thorough. Flag any issues that would require fixing.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": analysis_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{format_name.lower()};base64,{img_base64}", "detail": "high"},
                    },
                ],
            },
        ]

        # Call the vision model
        vision_response = client.chat.completions.create(
            model=get_model_for_role(MODEL_VISION), messages=vision_messages, max_tokens=1000
        )

        critique = vision_response.choices[0].message.content

        # Return structured response
        response = f"**Image:** {os.path.basename(image_path)}\n"
        response += f"**Dimensions:** {width}x{height}\n"
        response += f"**Format:** {format_name}\n"
        response += f"**Size:** {round(file_size / 1024, 2)} KB\n\n"
        response += "### Vision Model Analysis:\n"
        response += critique

        return response

    except ImportError:
        return "Error: PIL (Pillow) not installed. Cannot analyze images."
    except Exception as e:
        return f"Error analyzing image: {e!s}"


def _format_validation_result(result):
    """Format the validation result from the agent"""

    lines = []

    if result["passed"]:
        lines.append("## ✅ BUNDLE VALIDATION PASSED\n")
    else:
        lines.append("## ❌ BUNDLE VALIDATION FAILED\n")

    lines.append(result.get("summary", ""))
    lines.append("")

    if result.get("errors"):
        lines.append("### ERRORS (Must Fix)\n")
        for i, err in enumerate(result["errors"], 1):
            lines.append(f"**{i}.** {err}")
        lines.append("")

    if result.get("warnings"):
        lines.append("### WARNINGS (Should Review)\n")
        for i, warn in enumerate(result["warnings"], 1):
            lines.append(f"**{i}.** {warn}")
        lines.append("")

    if result.get("notes"):
        lines.append("### NOTES\n")
        for note in result["notes"]:
            lines.append(f"- {note}")

    return "\n".join(lines)


# Tool schema for PI to call
def validate_bundle_tool_schema():
    return {
        "type": "function",
        "function": {
            "strict": True,
            "name": "validate_analytic_bundle",
            "description": "Validate an analytic bundle before delivery. This runs an AI agent that thoroughly checks "
            "the bundle for data quality issues, code errors, documentation completeness, and more. "
            "The agent can execute Python code to inspect files and run tests. "
            "ALWAYS run this before delivering a bundle to the user. Takes up to 20 minutes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "zip_path": {"type": "string", "description": "Path to the analysis bundle zip file to validate"}
                },
                "additionalProperties": False,
                "required": ["zip_path"],
            },
        },
    }
