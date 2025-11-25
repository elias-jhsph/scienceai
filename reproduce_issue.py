import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from scienceai.llm import async_client, client
    print("Successfully imported clients")
    import asyncio
    
    async def test():
        print("Testing async client with proxy env var...")
        os.environ["HTTP_PROXY"] = "http://127.0.0.1:8080"
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8080"
        
        # Re-initialize client to pick up env vars
        from openai import AsyncOpenAI
        try:
            # We need to re-import or re-init because client might be created at module level
            # But llm.py creates it at module level.
            # So we should create a NEW client here to test.
            
            # Note: we need a key. llm.py loads it.
            from scienceai.llm import openai_key
            client = AsyncOpenAI(api_key=openai_key)
            
            await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}]
            )
        except Exception as e:
            print(f"Async client error: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(test())

except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
