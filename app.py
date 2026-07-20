from backend import (
    chatbot,
    get_all_threads,
    ingest_rag_document
)

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)

from langgraph.types import Command

import streamlit as st
import uuid
import tempfile
import os


# Generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())


# Add a new thread ID to the conversation list
def add_thread(thread_id):

    # Prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# Create a completely new chat conversation
def reset_chat():

    # Generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state["message_history"] = []

    # ========================= HITL ADDED =========================
    # Clear any pending human approval request
    st.session_state["pending_hitl"] = None
    # =============================================================

    # ========================= TITLE FEATURE ADDED =========================
    # NOTE: We do NOT add this new thread to the sidebar list here.
    # A brand-new empty chat has no messages yet, so there is nothing
    # meaningful to title it with. Like ChatGPT, it will only appear
    # in the sidebar once the user sends its first message
    # (see the add_thread() call further down, in the "if user_input:" block).
    # =========================================================================



# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    # Get the saved state for the selected thread
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get("messages", [])


# ========================= TITLE FEATURE ADDED =========================

# Return a short, readable title for a thread instead of showing its raw UUID.
# The title is the thread's MOST RECENT user message (not the first one),
# shortened if it's long, so the sidebar reflects what the conversation is
# currently about. If the thread has no messages yet, a placeholder is used.
def get_thread_title(thread_id):

    # Reuse the existing function to load this thread's saved messages
    messages = load_conversation(thread_id)

    # Maximum characters to show before shortening with "…"
    MAX_TITLE_LENGTH = 24

    # Walk the messages in REVERSE order, so we find the most recent
    # user message first instead of the very first one
    for message in reversed(messages):

        if isinstance(message, HumanMessage) and message.content:

            title = message.content.strip()

            # Keep the sidebar tidy by shortening long messages
            if len(title) > MAX_TITLE_LENGTH:
                title = title[:MAX_TITLE_LENGTH].rstrip() + "…"

            return title

    # No messages yet in this thread (e.g. it was just created)
    return f"New conversation ({str(thread_id)[:8]}…)"

# =========================================================================


# ========================= HITL helper functions =========================

def get_pending_interrupt(thread_id):
    """
    Return the first unresolved LangGraph interrupt for a thread.

    Returns:
        The pending Interrupt object, or None.
    """

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        # Read the current checkpoint state
        state_snapshot = chatbot.get_state(config)

        # Some LangGraph versions expose interrupts directly
        direct_interrupts = getattr(
            state_snapshot,
            "interrupts",
            ()
        ) or ()

        if direct_interrupts:
            return direct_interrupts[0]

        # Other LangGraph versions store interrupts inside tasks
        tasks = getattr(
            state_snapshot,
            "tasks",
            ()
        ) or ()

        for task in tasks:

            task_interrupts = getattr(
                task,
                "interrupts",
                ()
            ) or ()

            if task_interrupts:
                return task_interrupts[0]

    except Exception:

        # A newly created thread may not have a checkpoint yet
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):
    """
    Save the pending interrupt information inside Streamlit state.
    """

    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value)
    }


def sync_pending_interrupt(thread_id):
    """
    Synchronize Streamlit HITL state with the LangGraph checkpoint.

    This allows a pending approval request to reappear after:
    - a Streamlit rerun
    - a browser refresh
    - switching between conversations
    """

    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:

        save_pending_interrupt(
            thread_id,
            pending_interrupt
        )

    else:

        current_pending = st.session_state.get(
            "pending_hitl"
        )

        if (
            current_pending is not None
            and current_pending.get("thread_id") == thread_id
        ):
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision):
    """
    Resume an interrupted LangGraph execution.

    Args:
        decision:
            "yes" approves the stock purchase.
            "no" rejects the stock purchase.
    """

    pending_hitl = st.session_state.get(
        "pending_hitl"
    )

    if not pending_hitl:

        st.warning(
            "There is no pending action to approve or reject."
        )

        return

    # Get the thread that originally triggered the interrupt
    interrupted_thread_id = pending_hitl["thread_id"]

    # The same thread ID must be used when resuming
    resume_config = {
        "configurable": {
            "thread_id": interrupted_thread_id
        },
        "metadata": {
            "thread_id": interrupted_thread_id
        },
        "run_name": "hitl_resume_trace",
    }

    try:

        # Display the resumed response
        with st.chat_message("assistant"):

            status_holder = {
                "box": st.status(
                    "🔄 Resuming the requested action...",
                    expanded=True
                )
            }

            def resumed_ai_only_stream():

                # Resume the graph with the human decision
                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):

                    # Update tool execution status
                    if isinstance(
                        message_chunk,
                        ToolMessage
                    ):

                        tool_name = getattr(
                            message_chunk,
                            "name",
                            "tool"
                        )

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                    # Stream only assistant-generated text
                    if isinstance(
                        message_chunk,
                        AIMessage
                    ):

                        if message_chunk.content:
                            yield message_chunk.content

            # Display the streamed final answer
            resumed_ai_message = st.write_stream(
                resumed_ai_only_stream()
            )

            # Check whether another interrupt occurred
            next_interrupt = get_pending_interrupt(
                interrupted_thread_id
            )

            if next_interrupt is not None:

                save_pending_interrupt(
                    interrupted_thread_id,
                    next_interrupt
                )

                status_holder["box"].update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False
                )

            else:

                # No more pending approval
                st.session_state["pending_hitl"] = None

                status_holder["box"].update(
                    label="✅ Action completed",
                    state="complete",
                    expanded=False
                )

        # Save the assistant response in Streamlit UI history
        if resumed_ai_message:

            st.session_state["message_history"].append({
                "role": "assistant",
                "content": resumed_ai_message
            })

        # Rerun so the response appears in normal chat order
        st.rerun()

    except Exception as error:

        st.error(
            f"Could not resume the requested action: {error}"
        )


# ========================= Page configuration =========================

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖"
)

# ========================= TITLE POSITION ADDED =========================
# Streamlit adds empty space above the page content by default.
# This trims that gap so the title sits higher on the page.
st.markdown(
    "<style>.block-container { padding-top: 1.5rem; }</style>",
    unsafe_allow_html=True
)
# =========================================================================

# Display the main application title
st.title("Agentic Chatbot with LangGraph")

# ========================= CAPABILITIES BADGES ADDED =========================
# Show what this agent can actually do, as styled pill badges that are
# ALSO clickable — clicking one sends a real example message for that
# capability, using the exact same message flow as typing in the chat box.

CAPABILITY_BADGES = [
    ("📄", "Documents", "How do I upload a PDF, and what can you tell me about it once it's uploaded?"),
    ("🌐", "Web Search", "Search the web for the latest AI news today."),
    ("🧮", "Calculations", "Calculate 245 * 12 + sqrt(144) for me."),
    ("📈", "Stock Prices", "What is the current stock price of AAPL?"),
    ("💰", "Purchases", "Buy 5 shares of AAPL."),
    ("🌦️", "Weather", "What is the current weather in Nagpur, India?"),
]

# Holds a prompt queued by clicking a badge, so it can be picked up
# by the normal message-handling code further down the script
if "queued_prompt" not in st.session_state:
    st.session_state["queued_prompt"] = None

# CSS that restyles each badge's underlying st.button to look like a
# small rounded pill instead of a default rectangular Streamlit button.
# Each badge button sits inside st.container(key="cap_badge_N"), which
# Streamlit renders as a div with class "st-key-cap_badge_N" — that
# class is what lets this CSS target only these specific buttons.
st.markdown(
    """
    <style>
    div[data-testid="column"]:has([class*="st-key-cap_badge"]) {
        width: fit-content !important;
        flex: 0 0 auto !important;
        min-width: 0 !important;
    }

    [class*="st-key-cap_badge"] button {
        background-color: #1c2129 !important;
        border: 1px solid #2e3542 !important;
        border-radius: 999px !important;
        padding: 6px 14px !important;
        font-size: 0.78rem !important;
        color: #c9d1d9 !important;
        white-space: nowrap !important;
        width: max-content !important;
    }
    [class*="st-key-cap_badge"] button:hover {
        border-color: #58a6ff !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Lay the badges out in one row, left-aligned, each as its own
# clickable button that queues its prompt when clicked.
badge_column_widths = [len(label) + 6 for _, label, _ in CAPABILITY_BADGES]
badge_columns = st.columns(badge_column_widths, gap="medium")

for index, (col, (icon, label, prompt)) in enumerate(zip(badge_columns, CAPABILITY_BADGES)):

    with col:
        with st.container(key=f"cap_badge_{index}"):

            if st.button(f"{icon} {label}", key=f"cap_badge_btn_{index}"):
                st.session_state["queued_prompt"] = prompt
                st.rerun()

st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
# ===============================================================================


# Create message_history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# Create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()


# ========================= TITLE FEATURE ADDED =========================

# Cache of thread_id -> title, so we don't have to re-read the
# checkpoint database for every thread on every single rerun
if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

# =========================================================================


# ========================= HITL ADDED =========================

# Store the currently pending human approval request
if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

# =============================================================


# ========================= HITL ADDED =========================

# Recover pending approval after page refresh or rerun
sync_pending_interrupt(
    st.session_state["thread_id"]
)

# =============================================================


# ========================= Sidebar threading feature =========================

# Display the sidebar title
st.sidebar.title("My Conversations")


# Create a button for starting a new conversation
if st.sidebar.button("New Chat", use_container_width=True):

    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()


# Display all conversation threads in reverse order
for thread_id in st.session_state["chat_threads"][::-1]:

    # ========================= TITLE FEATURE ADDED =========================

    if thread_id not in st.session_state["thread_titles"]:
        st.session_state["thread_titles"][thread_id] = get_thread_title(thread_id)

    thread_title = st.session_state["thread_titles"][thread_id]

    # =========================================================================

    if st.sidebar.button(
        thread_title,
        key=thread_id,
        use_container_width=True
    ):

        # Set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # Load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        temp_messages = []

        for message in messages:

            if isinstance(message, HumanMessage):
                role = "user"

            elif isinstance(message, AIMessage):
                role = "assistant"

            else:
                continue

            temp_messages.append({
                "role": role,
                "content": message.content
            })

        st.session_state["message_history"] = temp_messages

        # ========================= HITL ADDED =========================

        sync_pending_interrupt(thread_id)

        # =============================================================

        st.rerun()


# ========================= Main chat interface =========================

# ========================= AUTO CAPABILITY INTRO ADDED =========================
# If this is a brand-new conversation with no messages yet, automatically
# show an assistant welcome bubble that explains what the agent can do —
# so the user understands its capabilities WITHOUT having to send a
# message first.
#
# IMPORTANT: this bubble is UI-only. It is never appended to
# message_history and never sent to the LangGraph backend, so it does
# not get saved as part of the thread, does not affect the sidebar
# title, and disappears on its own the moment a real message exists.
if not st.session_state["message_history"]:

    with st.chat_message("assistant"):
        st.markdown(
            "👋 Hi! I'm an agentic chatbot. Here's what I can help you with:\n\n"
            "- 📄 **Documents** — upload a PDF and ask me questions about it\n"
            "- 🌐 **Web Search** — I can search the web for the latest information\n"
            "- 🧮 **Calculations** — I can work out math problems for you\n"
            "- 📈 **Stock Prices** — ask me for the current price of any stock\n"
            "- 💰 **Purchases** — I can buy stocks for you, with your approval\n"
            "- 🌦️ **Weather** — ask me for the current weather in any city\n\n"
            "Tap one of the badges above, or just type a message below to get started."
        )
# ===============================================================================

# Display all messages from the currently selected conversation
for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.text(message["content"])


# ========================= HITL approval interface =========================

pending_hitl = st.session_state.get(
    "pending_hitl"
)

current_thread_has_pending_hitl = (
    pending_hitl is not None
    and pending_hitl.get("thread_id")
    == st.session_state["thread_id"]
)


if current_thread_has_pending_hitl:

    st.warning(
        "🧑 Human approval required\n\n"
        f"{pending_hitl['prompt']}"
    )

    approve_column, reject_column = st.columns(2)

    with approve_column:

        if st.button(
            "✅ Approve Purchase",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",
            use_container_width=True
        ):

            resume_hitl_execution("yes")

    with reject_column:

        if st.button(
            "❌ Reject Purchase",
            key=f"reject_{st.session_state['thread_id']}",
            use_container_width=True
        ):

            resume_hitl_execution("no")


# ========================= Fixed chat input with PDF upload =========================

submission = st.chat_input(
    "Type here",
    accept_file=True,
    file_type=["pdf"],
    disabled=current_thread_has_pending_hitl
)


user_input = None

# ========================= CAPABILITIES BADGES ADDED =========================
if st.session_state.get("queued_prompt"):
    user_input = st.session_state["queued_prompt"]
    st.session_state["queued_prompt"] = None
# ===============================================================================


if submission:

    user_input = submission.text

    uploaded_files = submission.files

    if uploaded_files:

        uploaded_pdf = uploaded_files[0]

        temporary_file_path = None

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temporary_file:

                temporary_file.write(
                    uploaded_pdf.getvalue()
                )

                temporary_file_path = temporary_file.name

            with st.spinner(
                f"Processing {uploaded_pdf.name}..."
            ):

                ingest_rag_document(
                    temporary_file_path
                )

            st.toast(
                f"{uploaded_pdf.name} processed successfully.",
                icon="✅"
            )

        except Exception as error:

            st.error(
                f"PDF processing failed: {error}"
            )

        finally:

            if (
                temporary_file_path
                and os.path.exists(temporary_file_path)
            ):
                os.remove(temporary_file_path)


if user_input:

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    # ========================= TITLE FEATURE ADDED =========================

    add_thread(st.session_state["thread_id"])

    st.session_state["thread_titles"].pop(st.session_state["thread_id"], None)

    # =========================================================================

    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }

    with st.chat_message("assistant"):

        status_holder = {
            "box": None
        }

        def ai_only_stream():

            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):

                if isinstance(
                    message_chunk,
                    ToolMessage
                ):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )

                    if status_holder["box"] is None:

                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …",
                            expanded=True
                        )

                    else:

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(
                    message_chunk,
                    AIMessage
                ):
                    yield message_chunk.content

            # ========================= HITL ADDED =========================

            pending_interrupt = get_pending_interrupt(
                st.session_state["thread_id"]
            )

            if pending_interrupt is not None:

                save_pending_interrupt(
                    st.session_state["thread_id"],
                    pending_interrupt
                )

                yield (
                    "\n\n⚠️ This stock purchase requires your approval. "
                    "Use the Approve Purchase or Reject Purchase "
                    "button below."
                )

            # =============================================================

        ai_message = st.write_stream(
            ai_only_stream()
        )

        if status_holder["box"] is not None:

            if get_pending_interrupt(
                st.session_state["thread_id"]
            ) is not None:

                status_holder["box"].update(
                    label="⏸️ Waiting for human approval",
                    state="complete",
                    expanded=False
                )

            else:

                status_holder["box"].update(
                    label="✅ Tool finished",
                    state="complete",
                    expanded=False
                )

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })

    # ========================= HITL ADDED =========================

    if (
        st.session_state.get("pending_hitl") is not None
        and st.session_state["pending_hitl"].get("thread_id")
        == st.session_state["thread_id"]
    ):
        st.rerun()

    # =============================================================