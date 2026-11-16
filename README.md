# Toss Up

Can't decide? Give Toss Up two to four options and what matters to you. It researches them on the web and commits to ONE pick, with reasons and sources.

It is a single AI agent with two real tools:
- **search_web**: web search (Tavily)
- **read_page**: reads one full page when search snippets are thin (Tavily)

The AI model is Google Gemini. The app is built with Streamlit.

Safety limits: max 8 steps per run, 4 searches, 2 page reads, capped answer length, 5 runs per visitor session, and a human approval step before any outside tool is called.

## Files
- `agent.py`: the agent (loop, tools, limits)
- `app.py`: the web page (approval gate, live trace, trace download)
- `eval_harness.py`: 13 test cases with known answers; reports pass@1 and pass^k
- `test_offline.py`: quick logic checks that need no keys
- `requirements.txt`: the packages to install
- `.env`: your secret keys (YOU create this; never upload it)

## One-time setup
1. Install Python (3.10 or newer) from python.org. On Windows, tick "Add Python to PATH".
2. Put the project files in a folder called `tossup` and open that folder in VS Code (File > Open Folder). Open a terminal (Terminal > New Terminal).
3. Get two free keys:
   - Gemini: aistudio.google.com/apikey (use a personal Google account)
   - Tavily: tavily.com (sign up, copy the key from the dashboard)
4. Create a file named exactly `.env` in the folder (not `.env.txt`) with these two lines, no quotes and no spaces:

       GEMINI_API_KEY=paste-your-gemini-key-here
       TAVILY_API_KEY=paste-your-tavily-key-here

5. Install the packages:

       python -m pip install -r requirements.txt

## Run it
Check the logic (no keys needed; should end with ALL TESTS PASSED):

    python test_offline.py

Start the app:

    python -m streamlit run app.py

Leave "Test mode" unchecked for real searches. Click "Decide for me", check the inputs shown, then click "Approve and run". You can download the full execution trace afterwards.

## Evaluate it
    python eval_harness.py

This runs every test case 3 times with fixed search results (no Tavily credits used) and prints:
- **pass@1**: the share of all runs that passed
- **pass^k**: the share of cases where ALL k runs passed (reliability)

It saves `eval_results.json` and `failures.md`. Read `failures.md` to see why each failed run failed. If failures mention 429 or quota, that is the free-tier rate limit; wait a few minutes and rerun.

## Put it online (public link)
1. Create a GitHub repository and upload the project files. Do NOT upload `.env`.
2. Go to share.streamlit.io, sign in with GitHub, click Create app, choose the repository, and set `app.py` as the main file.
3. In Advanced settings, paste your keys into Secrets:

       GEMINI_API_KEY = "your-key"
       TAVILY_API_KEY = "your-key"

4. Deploy and copy the link. Test it on your phone with Wi-Fi off.
5. The app sleeps when nobody visits. Open the link just before class and before you submit.

## Before you share the link
- Use fresh keys that have never appeared in a screenshot or chat.
- Connect nothing real: the agent only reads the web and recommends. It cannot send, buy, or delete anything.
- Prompt-injection testing is allowed only on links posted in the Agent Arcade thread.

## Troubleshooting
- `streamlit is not recognized`: use `python -m streamlit run app.py`.
- `No module named ...`: run `python -m pip install -r requirements.txt` again.
- `KeyError: GEMINI_API_KEY`: your `.env` is misnamed or in the wrong folder.
- `404 NOT_FOUND ... model`: Google retired the model name. In `agent.py` line near the top, set `MODEL_NAME = "gemini-flash-latest"`.
- 429 or quota errors: free-tier limit; wait a minute and retry.
- VS Code says the file is newer: do not click Overwrite. Close the tab without saving and reopen the file.
