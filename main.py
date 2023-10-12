import streamlit as st
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal
import os
import pandas as pd

st.set_page_config(
    page_title="LIDA: Automatic Generation of Visualizations and Infographics",
    page_icon="📊",
)

st.write("# LIDA: Automatic Generation of Visualizations and Infographics using Large Language Models 📊")

st.sidebar.write("## Setup")

openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    openai_key = st.sidebar.text_input("Enter OpenAI API key:")
    if openai_key:
        display_key = openai_key[:2] + "*" * (len(openai_key) - 5) + openai_key[-3:]
        st.sidebar.write(f"Current key: {display_key}")
    else:
        st.sidebar.write("Please enter OpenAI API key.")
else:
    display_key = openai_key[:2] + "*" * (len(openai_key) - 5) + openai_key[-3:]
    st.sidebar.write(f"OpenAI API key loaded from environment variable: {display_key}")


st.markdown(
    """
    LIDA is a library for generating data visualizations and data-faithful infographics.
    LIDA is grammar agnostic (will work with any programming language and visualization
    libraries e.g. matplotlib, seaborn, altair, d3 etc) and works with multiple large language
    model providers (OpenAI, Azure OpenAI, PaLM, Cohere, Huggingface). Details on the components
    of LIDA are described in the [paper here](https://arxiv.org/abs/2303.02927) and in this
    tutorial [notebook](notebooks/tutorial.ipynb). See the project page [here](https://microsoft.github.io/lida/) for updates!.

   This demo shows how to use the LIDA python api with Streamlit. [More](/about).
""")

if openai_key:
    # select model from gpt-4 , gpt-3.5-turbo, gpt-3.5-turbo-16k
    st.sidebar.write("## Text Generation Model")
    models = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
    selected_model = st.sidebar.selectbox(
        'Choose a model',
        options=models,
        index=1
    )
    # set use_cache in sidebar
    use_cache = st.sidebar.checkbox("Use cache", value=True)
    # select a dataset  or upload a dataset
    st.sidebar.write("## Data Summarization")

    st.sidebar.write("### Choose a dataset")
    datasets = {
        "Cars": "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv",
        "Weather": "https://raw.githubusercontent.com/uwdata/draco/master/data/weather.json",
    }

    selected_dataset = st.sidebar.selectbox(
        'Choose a dataset',
        options=list(datasets.keys())
    )


selected_dataset = datasets[selected_dataset] if selected_dataset in datasets else None

if selected_dataset == "Upload your own":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        import pandas as pd
        import os
        data = pd.read_csv(uploaded_file)
        data.to_csv('temp.csv')
        selected_dataset = os.path.abspath('temp.csv')

        st.sidebar.write("Uploaded file path: ", selected_dataset)


st.sidebar.write("### Choose a summarization method")
summarization_methods = ["default", "llm", "columns"]

selected_method = st.sidebar.selectbox("Choose a method", options=summarization_methods)


if openai_key and selected_dataset and selected_method:
    lida = Manager(text_gen=llm("openai", api_key=openai_key))
    textgen_config = TextGenerationConfig(
        n=1,
        temperature=0.5,
        model=selected_model,
        use_cache=use_cache)

    st.write("## Summary")

    summary = lida.summarize(
        selected_dataset,
        summary_method=selected_method,
        textgen_config=textgen_config)

    print(summary.keys())

    if "dataset_description" in summary:
        st.write(summary["dataset_description"])

    if "fields" in summary:
        fields = summary["fields"]
        nfields = []
        for field in fields:
            flatted_fields = {}
            flatted_fields["column"] = field["column"]
            # flatted_fields["dtype"] = field["dtype"]
            for row in field["properties"].keys():
                if row != "samples":
                    flatted_fields[row] = field["properties"][row]
                else:
                    flatted_fields[row] = str(field["properties"][row])
            # flatted_fields = {**flatted_fields, **field["properties"]}
            nfields.append(flatted_fields)
        nfields_df = pd.DataFrame(nfields)
        st.write(nfields_df)
    else:
        st.write(str(summary))

    if summary:
        st.sidebar.write("### Goal Selection")

        num_goals = st.sidebar.slider(
            "Number of goals to generate",
            min_value=1,
            max_value=10,
            value=4)
        own_goal = st.sidebar.checkbox("Add Your Own Goal")

        goals = lida.goals(summary, n=num_goals, textgen_config=textgen_config)
        st.write(f"## Goals ({len(goals)})")

        default_goal = goals[0].question
        goal_questions = [goal.question for goal in goals]

        if own_goal:
            user_goal = st.sidebar.text_input("Describe Your Goal")

            if user_goal:

                new_goal = Goal(question=user_goal, visualization=user_goal, rationale="")
                goals.append(new_goal)
                goal_questions.append(new_goal.question)

        selected_goal = st.selectbox('Choose a generated goal', options=goal_questions, index=0)

        # st.markdown("### Selected Goal")
        selected_goal_index = goal_questions.index(selected_goal)
        st.write(goals[selected_goal_index])

        selected_goal_object = goals[selected_goal_index]

        if selected_goal_object:
            st.sidebar.write("## Visualization Library")
            visualization_libraries = ["seaborn", "matplotlib", "plotly"]

            selected_library = st.sidebar.selectbox(
                'Choose a visualization library',
                options=visualization_libraries,
                index=0
            )

            # Update the visualization generation call to use the selected library.
            st.write("## Visualizations")

            # slider for number of visualizations
            num_visualizations = st.sidebar.slider(
                "Number of visualizations to generate",
                min_value=1,
                max_value=10,
                value=2)

            textgen_config = TextGenerationConfig(
                n=num_visualizations, temperature=0.2,
                model=selected_model,
                use_cache=use_cache)

            visualizations = lida.visualize(
                summary=summary,
                goal=selected_goal_object,
                textgen_config=textgen_config,
                library=selected_library)

            viz_titles = [f'Visualization {i+1}' for i in range(len(visualizations))]

            selected_viz_title = st.selectbox('Choose a visualization', options=viz_titles, index=0)

            selected_viz = visualizations[viz_titles.index(selected_viz_title)]

            if selected_viz.raster:
                from PIL import Image
                import io
                import base64

                imgdata = base64.b64decode(selected_viz.raster)
                img = Image.open(io.BytesIO(imgdata))
                st.image(img, caption=selected_viz_title, use_column_width=True)

            st.write("### Visualization Code")
            st.code(selected_viz.code)
