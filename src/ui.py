"""Small presentation helpers for the Streamlit PPE demonstration."""

import streamlit as st


def show_decision(decisions):
    """Render frame status and each worker's required PPE checks."""
    if not decisions:
        st.error("🔴 ACCESS DENIED")
        st.warning("No person detected. Access denied for safety.")
        return

    frame_allowed = all(decision.allowed for decision in decisions)
    if frame_allowed:
        st.success("🟢 ACCESS ALLOWED")
        st.write("All required PPE detected.")
    else:
        st.error("🔴 ACCESS DENIED")
        st.write("Worker is missing required PPE.")

    for index, decision in enumerate(decisions, start=1):
        st.markdown(f"**PERSON {index}**")
        for name in ("helmet", "vest", "boots"):
            mark = "✓" if decision.items[name].detected else "✗"
            st.write(f"{name.title()} {mark}")
        st.write(decision.message)
        if decision.missing:
            missing = ", ".join(item.title() for item in decision.missing)
            st.write(f"Missing: {missing}")


def show_device(device_info):
    """Show the active compute device in the sidebar."""
    st.sidebar.subheader("Inference device")
    st.sidebar.write(f"Device: {device_info['label']}")
    if device_info["cuda_available"]:
        st.sidebar.write(f"GPU: {device_info['gpu']}")
    else:
        st.sidebar.write("GPU: Not available")


def show_camera_unavailable():
    """Fail closed when the browser cannot provide camera frames."""
    st.error("🔴 ACCESS DENIED")
    st.warning("Camera unavailable. Please allow camera access or upload an image.")