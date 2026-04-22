import streamlit as st
from services.analysis import generate_response


# INPUT SECTION
st.subheader("Input Configuration")

dataset_url = st.text_input("Dataset URL")
target_column = st.text_input("Target Column")
sensitive_feature = st.text_input("Sensitive Feature")

# ACTION
if st.button("Analyze Bias"):

    # Validate input
    if not dataset_url or not target_column or not sensitive_feature:
        st.error("Please fill all fields")
    else:
        with st.spinner("Analyzing dataset..."):

            result = generate_response(
                dataset_url,
                target_column,
                sensitive_feature
            )
        # OUTPUT SECTION
        st.subheader("Analysis Result")

        if "error" in result:
            st.error(result["error"])

        else:
            st.success(f"Verdict: {result['verdict']}")
            st.metric("Bias Score", f"{result['biasScore']}%")

            # Metrics
            st.subheader("Metrics")
            st.write(result["metrics"])

            # Group Stats
            st.subheader("Group Comparison")
            st.write(result["groupStats"])

            # Feature Importance
            st.subheader(" Important Features")
            st.write(result["topFeatures"])

            # Suggestions
            st.subheader("Fix Suggestions")
            for s in result["fixSuggestions"]:
                st.warning(s["suggestion"])