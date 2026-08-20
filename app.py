import os
import streamlit as st
from config import settings
from export.exporter import Exporter
from ingestion.chunker import DocumentChunker
from ingestion.loader import DocumentLoader
from rag.rag_pipeline import RAGPipeline
from study.study_agent import StudyAgent
from utils.helpers import allowed_file_types
from storage.workspace_manager import WorkspaceManager
from study.ai_study_agent import AIStudyAgent

st.set_page_config(
    page_title="AI Study Assistant",
    layout="wide"
)

st.title("AI Study Assistant")
st.caption("Hybrid RAG | Neo4j | Qdrant | Groq")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "study_agent" not in st.session_state:
    st.session_state.study_agent = None

if "loader" not in st.session_state:
    st.session_state.loader = DocumentLoader()

if "chunker" not in st.session_state:
    st.session_state.chunker = DocumentChunker()

if "exporter" not in st.session_state:
    st.session_state.exporter = Exporter()

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_debug" not in st.session_state:
    st.session_state.last_debug = None

if "workspace_manager" not in st.session_state:
    st.session_state.workspace_manager = WorkspaceManager()

if "workspace" not in st.session_state:
    st.session_state.workspace = None

if "ai_study_agent" not in st.session_state:
    st.session_state.ai_study_agent = None

workspace_manager = st.session_state.workspace_manager
pipeline = st.session_state.pipeline
study_agent = st.session_state.study_agent
loader = st.session_state.loader
chunker = st.session_state.chunker
exporter = st.session_state.exporter


with st.sidebar:
    st.header("Workspace")

    workspaces = workspace_manager.list_workspaces()
    mode = st.radio(
        "Mode",
        [
            "Open Workspace",
            "Create Workspace"
        ]
    )

    debug_mode = st.checkbox(
        "Retrieval Debug",
        value=False
    )

    if mode == "Open Workspace":

        if workspaces:
            workspace_name = st.selectbox(
                "Workspace",
                workspaces
            )

            open_button = st.button(
                "Open Workspace",
                use_container_width=True
            )

        else:
            st.info("No workspaces found.")
            workspace_name = ""
            open_button = False

        uploaded_files = st.file_uploader(
            "Add more documents (optional)",
            type=allowed_file_types(),
            accept_multiple_files=True
        )

        add_button = st.button(
            "Add Documents",
            use_container_width=True
        )

        create_button = False

    else:

        workspace_name = st.text_input(
            "Workspace Name"
        )

        uploaded_files = st.file_uploader(
            "Choose files",
            type=allowed_file_types(),
            accept_multiple_files=True
        )

        create_button = st.button(
            "Create Workspace",
            use_container_width=True
        )

        open_button = False
        add_button = False

if open_button:

    workspace_path = workspace_manager.get_workspace_path(
        workspace_name
    )

    st.session_state.pipeline = RAGPipeline(
        workspace_path,
        workspace_name
    )

    pipeline = st.session_state.pipeline

    st.session_state.study_agent = StudyAgent(
        pipeline.retriever
    )

    st.session_state.ai_study_agent = AIStudyAgent(
        retriever=pipeline.retriever,
        llm=pipeline.llm
    )

    st.session_state.documents_loaded = True
    st.session_state.workspace = workspace_name
    st.session_state.chat_history = workspace_manager.load_chat(
        workspace_name
    )

    st.success("Workspace opened successfully.")

if create_button:

    if not workspace_name.strip():
        st.warning("Enter a workspace name.")
        st.stop()

    if not uploaded_files:
        st.warning("Please upload at least one document.")
        st.stop()

    if not workspace_manager.exists(workspace_name):
        workspace_manager.create_workspace(workspace_name)

    workspace_path = workspace_manager.get_workspace_path(
        workspace_name
    )

    st.session_state.pipeline = RAGPipeline(
        workspace_path,
        workspace_name
    )

    pipeline = st.session_state.pipeline

    st.session_state.study_agent = StudyAgent(
        pipeline.retriever
    )

    st.session_state.ai_study_agent = AIStudyAgent(
        retriever=pipeline.retriever,
        llm=pipeline.llm
    )

    study_agent = st.session_state.study_agent
    documents = []

    progress = st.progress(0)
    status = st.empty()

    for index, uploaded_file in enumerate(uploaded_files):
        status.write(
            f"Processing {uploaded_file.name}..."
        )

        upload_folder = workspace_manager.get_upload_path(
            workspace_name
         )

        temp_path = upload_folder / uploaded_file.name

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        document = loader.load(temp_path)

        if document is not None:
            documents.append(document)

            workspace_manager.add_document(
                workspace_name,
                uploaded_file.name
            )

        progress.progress(
            (index + 1) / len(uploaded_files)
        )

    if not documents:
        st.error("No documents could be loaded.")
        st.stop()

    status.write("Chunking documents...")

    chunks = chunker.chunk_documents(
        documents
    )

    if not chunks:
        st.error("No chunks were created.")
        st.stop()

    status.write("Building knowledge base...")
    pipeline.ingest_documents(
        chunks
    )

    st.session_state.documents_loaded = True

    progress.empty()
    status.empty()

    st.success(
         f"Successfully processed {len(uploaded_files)} document(s)."
    )

    st.session_state.workspace = workspace_name

if add_button:

    if not uploaded_files:
        st.warning("Upload at least one document.")
        st.stop()

    workspace_path = workspace_manager.get_workspace_path(
        workspace_name
    )

    if st.session_state.pipeline is None or (
        st.session_state.workspace != workspace_name
    ):

        st.session_state.pipeline = RAGPipeline(
            workspace_path,
            workspace_name
        )

    pipeline = st.session_state.pipeline

    documents = []

    for uploaded_file in uploaded_files:
        upload_folder = workspace_manager.get_upload_path(
            workspace_name
        )

        temp_path = upload_folder / uploaded_file.name

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        document = loader.load(temp_path)

        if document is not None:
            documents.append(document)

            workspace_manager.add_document(
                workspace_name,
                uploaded_file.name
            )

    if documents:

        chunks = chunker.chunk_documents(
            documents
        )

        pipeline.ingest_documents(
            chunks
        )

        st.success("Documents added successfully.")

if "workspace" not in st.session_state:
    st.session_state.workspace = None

if (
    workspace_name
    and workspace_name != st.session_state.workspace
):
    st.session_state.workspace = workspace_name
    st.session_state.chat_history = workspace_manager.load_chat(
        workspace_name
    )

chat_tab, study_tab, agent_tab = st.tabs(
    [
        "Chat",
        "Study Tools",
        "AI Study Agent"
    ]
)

with chat_tab:
    if st.session_state.documents_loaded:
        st.divider()

        question = st.text_input(
            "Ask a question about your documents"
        )

        ask = st.button(
            "Ask",
            use_container_width=True
        )

        if ask:

            if not question.strip():
                st.warning("Please enter a question.")

            else:
                with st.spinner("Thinking..."):

                    result = pipeline.ask(
                        question,
                        debug=debug_mode
                    )

                if debug_mode:
                    response = result["response"]
                    st.session_state.last_debug = result["debug"]
                else:
                    response = result
                    st.session_state.last_debug = None

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": response.answer
                    }
                )

                workspace_manager.save_chat(
                    workspace_name,
                    st.session_state.chat_history
                )

        if st.session_state.chat_history:
            st.divider()

            for i, chat in enumerate(
                reversed(st.session_state.chat_history)
            ):
                with st.chat_message("user"):
                    st.write(chat["question"])

                with st.chat_message("assistant"):
                    st.write(chat["answer"])
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        txt = exporter.export_temp(
                            chat["answer"],
                            "txt",
                            title="Answer"
                        )
                        with open(txt, "rb") as file:
                            st.download_button(
                                "TXT",
                                file,
                                file_name="answer.txt",
                                key=f"txt1_{i}"
                            )

                    with col2:
                        docx = exporter.export_temp(
                            chat["answer"],
                            "docx",
                            title="Answer"
                        )
                        with open(docx, "rb") as file:
                            st.download_button(
                                "DOCX",
                                file,
                                file_name="answer.docx",
                                key=f"docx1_{i}"
                            )

                    with col3:
                        pdf = exporter.export_temp(
                            chat["answer"],
                            "pdf",
                            title="Answer"
                        )
                        with open(pdf, "rb") as file:
                            st.download_button(
                                "PDF",
                                file,
                                file_name="answer.pdf",
                                key=f"pdf1_{i}"
                            )

    else:
        st.info(
            "Upload and process documents to start chatting"
        )

if debug_mode and st.session_state.last_debug is not None:
    debug = st.session_state.last_debug

    st.divider()
    st.subheader("Retrieval Debug")

    with st.expander("Primary Chunks", expanded=True):
        for chunk in debug["primary_chunks"]:
            st.write(f"**{chunk.source}** (Page {chunk.page})")
            st.caption(chunk.text[:300] + "...")

    with st.expander("Graph Chunks"):
        for chunk in debug["graph_chunks"]:
            st.write(f"**{chunk.source}** (Page {chunk.page})")
            st.caption(chunk.text[:300] + "...")

    with st.expander("Final Context"):
        for chunk in debug["context_chunks"]:
            st.write(f"**{chunk.source}** (Page {chunk.page})")
            st.caption(chunk.text[:300] + "...")

with study_tab:
    st.divider()
    st.header("Study Tools")

    topic = st.text_input(
        "Topic",
        placeholder="Enter a topic from your uploaded documents"
    )

    study_option = st.selectbox(
        "Select Tool",
        [
            "study_plan",
            "notes",
            "explain",
            "flashcards",
            "quiz",
            "practice_questions"
        ]
    )

    count = st.number_input(
        "Number of Items",
        min_value=5,
        max_value=30,
        value=10
    )

    generate = st.button(
        "Generate",
        use_container_width=True
    )

    if generate:
        if not st.session_state.documents_loaded:
            st.warning(
                "Upload documents first."
            )
        elif st.session_state.study_agent is None:
            st.warning(
                "Please process your documents first."
            )
        elif not topic.strip():
            st.warning(
                "Enter a topic."
            )

        else:
            study_agent = st.session_state.study_agent
            with st.spinner("Generating..."):

                result = study_agent.execute(
                    action=study_option,
                    topic=topic,
                    count=count
                )

            st.subheader(
                study_option.replace("_", " ").title()
            )

            st.write(result)
            col1, col2, col3 = st.columns(3)

            with col1:
                txt = exporter.export_temp(
                    result,
                    "txt",
                    title=study_option.replace("_", " ").title(),
                    topic=topic
                )
                with open(txt, "rb") as file:
                    st.download_button(
                        "Download TXT",
                        file,
                        file_name=f"{study_option}.txt",
                        key=f"txt2_{i}"
                    )

            with col2:
                docx = exporter.export_temp(
                    result,
                    "docx",
                    title=study_option.replace("_", " ").title(),
                    topic=topic
                )
                with open(docx, "rb") as file:
                    st.download_button(
                        "Download DOCX",
                        file,
                        file_name=f"{study_option}.docx",
                        key=f"docx2_{i}"
                    )

            with col3:
                pdf = exporter.export_temp(
                    result,
                    "pdf",
                    title=study_option.replace("_", " ").title(),
                    topic=topic
                )
                with open(pdf, "rb") as file:
                    st.download_button(
                        "Download PDF",
                        file,
                        file_name=f"{study_option}.pdf",
                        key=f"pdf2_{i}"
                    )

    st.divider()
    st.markdown(
        """
        <center>
            <h4>AI Study Assistant</h4>
            <p>
            Hybrid RAG • Neo4j • Qdrant • Groq
            </p>
        </center>
        """,
        unsafe_allow_html=True
    )

with agent_tab:
    st.subheader(" AI Study Agent")

    if not st.session_state.documents_loaded:
        st.warning("Please upload and process documents first.")
        st.stop()

    goal = st.text_area(
        "What do you want to achieve?",
        placeholder=(
            "Examples:\n"
            "- Help me prepare for my DBMS interview\n"
            "- I have an OS exam tomorrow\n"
            "- Teach me transformers\n"
            "- Revise the important topics"
        )
    )

    if st.button("Start AI Study Agent"):
        if not goal.strip():
            st.warning("Please enter your goal.")
            st.stop()

        with st.spinner("Planning..."):
            result = (
                st.session_state.ai_study_agent.run(goal)
            )

        st.success("Plan Created")
        st.subheader("Plan")
        st.json(result["plan"])
        st.subheader("Results")

        for output in result["results"]:
            st.markdown(
                f"### {output['tool'].replace('_',' ').title()}"
            )

            if "error" in output:
                st.error(output["error"])

            else:
                st.markdown(output["result"])

        download_text = "# AI Study Agent\n\n"
        download_text += f"## Goal\n{goal}\n\n"
        download_text += "## Plan\n"

        for step in result["plan"]:
            download_text += (
                f"- {step['tool'].replace('_',' ').title()}\n"
            )

        download_text += "\n## Results\n\n"

        for output in result["results"]:
            download_text += (
                f"### {output['tool'].replace('_',' ').title()}\n\n"
            )

            if "error" in output:
                download_text += output["error"]
            else:
                download_text += output["result"]

            download_text += "\n\n"

            st.divider()
            st.subheader("Download Complete Study Session")

            col1, col2, col3 = st.columns(3)

            with col1:
                txt = exporter.export_temp(
                    download_text,
                    "txt",
                    title="AI Study Agent",
                    topic=goal
                )
                with open(txt, "rb") as file:

                    st.download_button(
                        "TXT",
                        file,
                        file_name="ai_study_session.txt",
                        key=f"txt3_{i}"
                    )

            with col2:
                docx = exporter.export_temp(
                    download_text,
                    "docx",
                    title="AI Study Agent",
                    topic=goal
                )
                with open(docx, "rb") as file:

                    st.download_button(
                        "DOCX",
                        file,
                        file_name="ai_study_session.docx",
                        key=f"docx3_{i}"
                    )

            with col3:
                pdf = exporter.export_temp(
                    download_text,
                    "pdf",
                    title="AI Study Agent",
                    topic=goal
                )
                with open(pdf, "rb") as file:

                    st.download_button(
                        "PDF",
                        file,
                        file_name="ai_study_session.pdf",
                        key=f"pdf3_{i}"
                    )